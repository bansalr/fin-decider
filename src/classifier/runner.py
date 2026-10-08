"""Run orchestration: identity, manifest, concurrency, resume, cost guard."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from . import results
from .adapters import make_adapter
from .adapters.base import ClassificationAdapter, ClassificationContext, FatalProviderError
from .config import Config, check_credentials
from .dedupe import load_corpus
from .inputs import build_input
from .ontology import Ontology, load_ontology
from .provenance import environment_info, git_info, make_run_id, sha256_obj, utc_now
from .templates import load_template
from .traversal import classify_article

# Operational knobs that do not change any classification; excluded from identity hashes.
NON_MATERIAL_MODEL_FIELDS = {"concurrency", "retries", "timeout_seconds", "price_per_m_input_tokens", "expected_model_returned", "endpoint"}
NON_MATERIAL_RUN_FIELDS = {"allow_dirty_git", "label"}
NON_MATERIAL_OUTPUT_FIELDS = {"directory", "shard_rows"}
SNAPSHOT_FIELDS = ("id", "type", "context_window", "released", "owned_by")


SHARED_SECTIONS = ("benchmark", "dataset", "corpus", "ontology", "input", "hierarchy")


def material_config(cfg: Config, model: str) -> dict[str, Any]:
    """Everything that can change this model's classifications: the shared sections,
    this model's settings, and its template. Other models never affect a run's identity."""
    d = cfg.model_dump()
    m = d["models"][model]
    return {k: d[k] for k in SHARED_SECTIONS} | {
        "run": {k: v for k, v in d["run"].items() if k not in NON_MATERIAL_RUN_FIELDS},
        "output": {k: v for k, v in d["output"].items() if k not in NON_MATERIAL_OUTPUT_FIELDS},
        "model": {"key": model} | {k: v for k, v in m.items() if k not in NON_MATERIAL_MODEL_FIELDS},
        "template": d["templates"][m["template"]],
    }


def model_config_hash(cfg: Config, model: str) -> str:
    return sha256_obj(material_config(cfg, model)["model"])


def shared_inputs_hash(cfg: Config, onto_sha: str, corpus_hash: str, scope: dict) -> str:
    """Identical across models classified on the same inputs; use it to pair runs for comparison."""
    d = cfg.model_dump()
    return sha256_obj({k: d[k] for k in SHARED_SECTIONS} | {"ontology_sha": onto_sha, "corpus": corpus_hash,
                                                             "scope": scope})


@dataclass
class RunContext:
    cfg: Config
    onto: Ontology
    corpus: pl.DataFrame
    corpus_manifest: dict[str, Any]
    templates: dict[str, Any]
    model: str
    run_id: str
    run_hash: str
    run_dir: Path
    git: dict[str, Any]
    scope: dict[str, Any]
    dry_run: bool = False


def prepare_run(cfg: Config, *, model: str, limit: int | None, repo_root: Path = Path(".")) -> RunContext:
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    if onto.version != cfg.ontology.version:
        raise ValueError(f"ontology version {onto.version} != configured {cfg.ontology.version}")
    if model not in cfg.models:
        raise ValueError(f"unknown model {model!r}")
    tname = cfg.models[model].template
    t = cfg.templates[tname]
    templates = {tname: load_template(t.path, t.sha256, t.kind)}
    templates[tname].check_ontology(onto)
    corpus, cman = load_corpus(Path(cfg.corpus.directory))
    if cman["dataset_sha256"] != cfg.dataset.file_sha256:
        raise ValueError("corpus was prepared from a different dataset file; rerun `classify prepare`")
    if cman["dedupe"] != cfg.corpus.dedupe.model_dump() or cman["normalization_version"] != cfg.corpus.normalization_version:
        raise ValueError("corpus was prepared with different dedupe/normalization settings; rerun `classify prepare`")

    corpus = corpus.sort("article_id")
    scope = {"kind": "full_corpus", "articles": corpus.height}
    if limit is not None:
        corpus = corpus.head(limit)
        scope = {"kind": "engineering_first_n_by_article_id", "articles": corpus.height, "limit": limit}

    git = git_info(repo_root)
    if git["dirty"] and not cfg.run.allow_dirty_git:
        raise RuntimeError("git working tree is dirty; commit first or set run.allow_dirty_git for engineering runs")

    run_hash_short, run_hash = make_run_id(
        resolved_config=material_config(cfg, model) | {"scope": scope},
        ontology_hash=onto.sha256,
        dataset_hash=cfg.dataset.file_sha256,
        corpus_manifest_hash=cman["corpus_manifest_hash"],
        template_hashes={k: t.sha256 for k, t in templates.items()},
        git_commit=git["commit"],
    )
    run_id = f"{model}-{run_hash_short}"
    run_dir = Path(cfg.output.directory) / run_id
    return RunContext(cfg, onto, corpus, cman, templates, model, run_id, run_hash, run_dir, git, scope)


async def model_snapshot(cfg: Config, model: str) -> dict[str, Any]:
    """The Gateway's catalog entry for the model at run start (versions are not pinnable)."""
    m = cfg.models[model]
    import os

    if m.provider == "cloudflare_workers_ai":
        return await _workers_ai_snapshot(m)
    if m.provider == "local_server":
        return await _local_server_snapshot(m)
    if m.provider != "vercel_ai_gateway":
        return {"id": m.model, "provider": m.provider}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{m.endpoint}/v1/models", headers={"Authorization": f"Bearer {os.environ[m.api_key_env]}"})
        r.raise_for_status()
    entries = [e for e in r.json().get("data", []) if e.get("id") == m.model]
    if not entries:
        raise FatalProviderError(f"model {m.model} not listed by the gateway")
    e = entries[0]
    return {k: e.get(k) for k in SNAPSHOT_FIELDS} | {"pricing": e.get("pricing")}


async def _local_server_snapshot(m) -> dict[str, Any]:
    """Self-hosted server: pinned weight revisions, quantization, the server's own /health
    report, and the accelerator it runs on. Endpoint is operational and not hashed."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{m.endpoint}/health")
        r.raise_for_status()
        health = r.json()
    snap = {"id": m.model, "provider": m.provider, "revisions": m.revisions, "quantization": m.quantization,
            "health": {k: v for k, v in health.items() if k not in ("status", "uptime_s", "requests")}}
    try:
        import torch

        if torch.cuda.is_available():
            snap["accelerator"] = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                                   "torch": torch.__version__}
    except ImportError:
        pass
    return snap


async def _workers_ai_snapshot(m) -> dict[str, Any]:
    """Workers AI catalog entry plus the open-weights repo revision, for the record.
    Neither pins the served version (spec §0.3); a change between sessions is fatal."""
    import os

    account, token = os.environ[m.account_id_env], os.environ[m.api_key_env]
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{m.endpoint}/accounts/{account}/ai/models/search",
                             params={"search": m.model.rsplit("/", 1)[-1]},
                             headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        entries = [e for e in (r.json().get("result") or []) if e.get("name") == m.model]
        if not entries:
            raise FatalProviderError(f"model {m.model} not listed by Workers AI")
        e = entries[0]
        # The open-weights revision is recorded for reference only (it does not pin what
        # Workers AI serves), so a Hugging Face outage must not block the run.
        try:
            hf = await client.get("https://huggingface.co/api/models/Cloudflare/" + m.model.rsplit("/", 1)[-1])
            hf_revision = hf.json().get("sha") if hf.status_code == 200 else f"unavailable (HTTP {hf.status_code})"
        except httpx.HTTPError as err:
            hf_revision = f"unavailable ({type(err).__name__})"
    snap = {"id": m.model, "provider": m.provider,
            "catalog": {k: e.get(k) for k in ("id", "name", "created_at", "task")},
            "properties": e.get("properties")}
    return snap | {"hf_revision": hf_revision}


def _manifest_path(rc: RunContext) -> Path:
    return rc.run_dir / "manifest.json"


def load_or_init_manifest(rc: RunContext) -> dict[str, Any]:
    p = _manifest_path(rc)
    if p.exists():
        man = json.loads(p.read_text(encoding="utf-8"))
        if man["run_hash"] != rc.run_hash:
            raise RuntimeError("existing manifest belongs to a different run identity")
        return man
    cfg = rc.cfg
    return {
        "run_id": rc.run_id,
        "run_hash": rc.run_hash,
        "benchmark": cfg.benchmark.model_dump(),
        "run_label": cfg.run.label,
        "scope": rc.scope,
        "git": rc.git,
        "environment": environment_info(),
        "dataset": {
            "repository": cfg.dataset.repository,
            "revision": cfg.dataset.revision,
            "file": cfg.dataset.file,
            "sha256": cfg.dataset.file_sha256,
        },
        "corpus": {
            "manifest_hash": rc.corpus_manifest["corpus_manifest_hash"],
            "stats": rc.corpus_manifest["stats"],
            "dedupe": rc.corpus_manifest["dedupe"],
        },
        "ontology": {"id": rc.onto.id, "version": rc.onto.version, "sha256": rc.onto.sha256},
        "template_hashes": {k: t.sha256 for k, t in rc.templates.items()},
        "config_sha256": sha256_obj(material_config(cfg, rc.model)),
        "resolved_config": material_config(cfg, rc.model),
        "shared_inputs_sha256": shared_inputs_hash(cfg, rc.onto.sha256, rc.corpus_manifest["corpus_manifest_hash"],
                                                   rc.scope),
        "dry_run": rc.dry_run,
        "skipped_by_routing": "omitted",
        "row_semantics": "article x model x node x child; per-call usage fields repeat per child row, aggregate by prediction_key",
        "models": {},
        "created_at": utc_now(),
    }


def save_manifest(rc: RunContext, man: dict[str, Any]) -> None:
    rc.run_dir.mkdir(parents=True, exist_ok=True)
    tmp = _manifest_path(rc).with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(_manifest_path(rc))


def _log(rc: RunContext, msg: str) -> None:
    line = f"{utc_now()} {msg}"
    print(line, file=sys.stderr, flush=True)
    rc.run_dir.mkdir(parents=True, exist_ok=True)
    with open(rc.run_dir / "run.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


async def run_model(rc: RunContext, model: str, *, resume: bool, max_cost_usd: float | None,
                    adapter: ClassificationAdapter | None = None, progress_every: int = 500) -> dict[str, Any]:
    cfg = rc.cfg
    if model != rc.model:
        raise ValueError(f"run context was prepared for {rc.model}, not {model}")
    mcfg = cfg.models[model]
    if not mcfg.enabled:
        raise ValueError(f"model {model} is disabled in config")
    if adapter is None:
        check_credentials(cfg, [model])
        if mcfg.provider in ("cloudflare_workers_ai", "local_server") and not mcfg.expected_model_returned:
            raise RuntimeError(f"{model}: set expected_model_returned from `classify probe -m {model}` before running")

    man = load_or_init_manifest(rc)
    existing_work = any(results.work_dir(rc.run_dir, model).glob("part-*.parquet"))
    if existing_work and not resume:
        raise RuntimeError(f"run {rc.run_id} already has results for {model}; pass --resume")

    snapshot = await model_snapshot(cfg, model) if adapter is None else {"id": mcfg.model, "provider": "test"}
    snap_hash = sha256_obj({k: v for k, v in snapshot.items() if k != "pricing"})
    mman = man["models"].get(model)
    if mman and mman["catalog_snapshot_sha256"] != snap_hash and not mcfg.allow_returned_model_mismatch:
        raise FatalProviderError(f"{model}: gateway catalog entry changed since this run started: {snapshot}")
    mman = mman or {
        "requested": mcfg.model,
        "provider_lock": list(mcfg.provider_lock),
        "model_config_hash": model_config_hash(cfg, model),
        "catalog_snapshot": snapshot,
        "catalog_snapshot_sha256": snap_hash,
        "returned_versions": [],
        "routing_providers": [],
        "started_at": utc_now(),
    }
    mman["status"] = "RUNNING"
    man["models"][model] = mman
    save_manifest(rc, man)

    cache, complete, attempts = results.load_cache(rc.run_dir, model) if resume else ({}, set(), {})
    todo = [r for r in rc.corpus.iter_rows(named=True) if r["article_id"] not in complete]
    _log(rc, f"[{model}] run {rc.run_id}: {len(todo)} articles to do, {len(complete)} already complete, "
             f"{len(cache)} cached decisions")

    adapter = adapter or make_adapter(model, mcfg)
    template = rc.templates[mcfg.template]
    ctx = ClassificationContext(model_id=mcfg.model, template=template)
    ontology_hash = rc.onto.sha256
    mch = mman["model_config_hash"]
    node_hashes = {nid: rc.onto.node_hash(nid) for nid in rc.onto.nodes}
    writer = results.ShardWriter(rc.run_dir, model, cfg.output.shard_rows)
    static = {
        "run_id": rc.run_id,
        "model_id": model,
        "model_version": mcfg.model,
        "task_root": rc.onto.roots[0],
        "config_hash": man["config_sha256"],
        "ontology_hash": ontology_hash,
        "dataset_hash": cfg.dataset.file_sha256,
        "corpus_manifest_hash": rc.corpus_manifest["corpus_manifest_hash"],
        "template_hash": template.sha256,
        "code_commit": rc.git["commit"],
    }

    # A model-specific technical limit tightens the shared cap; never loosens it.
    input_cfg = cfg.input
    if mcfg.max_input_chars and mcfg.max_input_chars < cfg.input.max_chars:
        input_cfg = cfg.input.model_copy(update={"max_chars": mcfg.max_input_chars})
    stats = {"articles": 0, "complete": 0, "incomplete": 0, "calls": 0, "cost_usd": 0.0, "input_tokens": 0}
    returned, providers = set(mman["returned_versions"]), set(mman["routing_providers"])
    it = iter(todo)
    stop = asyncio.Event()
    t0 = time.monotonic()

    async def worker() -> None:
        while not stop.is_set():
            try:
                row = next(it)
            except StopIteration:
                return
            art = build_input(row["article_id"], row["headline"], row["article"], input_cfg)

            def key_fn(node_id: str, _h=art.input_hash) -> str:
                return sha256_obj([mcfg.model, _h, node_hashes[node_id], ontology_hash, mch, template.sha256])

            outcome = await classify_article(art, rc.onto, adapter, ctx, cfg.hierarchy, key_fn, cache)
            attempt = attempts.get(art.article_id, 0) + 1
            now = utc_now()
            per_article = {
                "article_id": art.article_id,
                "original_chars": art.original_chars,
                "submitted_chars": art.submitted_chars,
                "truncated": art.truncated,
                "truncation_strategy": art.truncation_strategy,
                "attempt": attempt,
                "created_at": now,
            }
            rows = [static | per_article | r for r in outcome.rows]
            writer.add_article(rows)

            stats["articles"] += 1
            stats["complete" if outcome.complete else "incomplete"] += 1
            stats["calls"] += outcome.calls_made
            seen_keys = set()
            for r in outcome.rows:
                if r["status"] == "SUCCESS" and not r["reused_from_cache"] and r["prediction_key"] not in seen_keys:
                    seen_keys.add(r["prediction_key"])
                    stats["cost_usd"] += r["provider_cost_usd"] or 0.0
                    stats["input_tokens"] += r["input_tokens"] or 0
                if r["model_returned"]:
                    returned.add(r["model_returned"])
                if r["routing_provider"]:
                    providers.add(r["routing_provider"])

            if max_cost_usd is not None and stats["cost_usd"] >= max_cost_usd and not stop.is_set():
                _log(rc, f"[{model}] cost guard reached (${stats['cost_usd']:.4f} >= ${max_cost_usd}); stopping")
                stop.set()
            if stats["articles"] % progress_every == 0:
                el = time.monotonic() - t0
                rate = stats["articles"] / el if el else 0.0
                eta_h = (len(todo) - stats["articles"]) / rate / 3600 if rate else float("inf")
                _log(rc, f"[{model}] {stats['articles']}/{len(todo)} articles, {rate:.2f}/s, ETA {eta_h:.1f}h, "
                         f"calls {stats['calls']}, tokens {stats['input_tokens']}, cost ${stats['cost_usd']:.4f}, "
                         f"incomplete {stats['incomplete']}")

    fatal: BaseException | None = None
    try:
        tasks = [asyncio.create_task(worker()) for _ in range(mcfg.concurrency)]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            if t.exception() is not None:
                fatal = t.exception()
        if fatal:
            stop.set()
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        writer.flush()
        await adapter.aclose()
        complete_after = results.complete_articles(rc.run_dir, model)
        finished = len(complete_after) == rc.corpus.height
        session_s = round(time.monotonic() - t0, 1)
        mman["elapsed_s_total"] = round(mman.get("elapsed_s_total", 0.0) + session_s, 1)
        mman["sessions"] = mman.get("sessions", 0) + 1
        mman.update(
            {
                "returned_versions": sorted(returned),
                "routing_providers": sorted(providers),
                "status": "COMPLETE" if finished and not fatal else "INCOMPLETE",
                "complete_articles": len(complete_after),
                "last_session": stats | {"elapsed_s": session_s, "ended_at": utc_now(),
                                         "fatal": repr(fatal) if fatal else None},
            }
        )
        if finished and not fatal:
            mman["completed_at"] = utc_now()
            results.compact(rc.run_dir, model)
        save_manifest(rc, man)
        _log(rc, f"[{model}] session end: {stats}; status {mman['status']}")
    if fatal:
        raise fatal
    return stats
