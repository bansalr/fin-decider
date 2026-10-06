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
NON_MATERIAL_MODEL_FIELDS = {"concurrency", "retries", "timeout_seconds"}
NON_MATERIAL_RUN_FIELDS = {"allow_dirty_git", "label"}
SNAPSHOT_FIELDS = ("id", "type", "context_window", "released", "owned_by")


def material_config(cfg: Config) -> dict[str, Any]:
    d = cfg.model_dump()
    d["run"] = {k: v for k, v in d["run"].items() if k not in NON_MATERIAL_RUN_FIELDS}
    d["models"] = {
        name: {k: v for k, v in m.items() if k not in NON_MATERIAL_MODEL_FIELDS}
        for name, m in d["models"].items()
        if m["enabled"]
    }
    return d


def model_config_hash(cfg: Config, model: str) -> str:
    return sha256_obj(material_config(cfg)["models"][model])


@dataclass
class RunContext:
    cfg: Config
    onto: Ontology
    corpus: pl.DataFrame
    corpus_manifest: dict[str, Any]
    templates: dict[str, Any]
    run_id: str
    run_hash: str
    run_dir: Path
    git: dict[str, Any]
    scope: dict[str, Any]


def prepare_run(cfg: Config, *, limit: int | None, repo_root: Path = Path(".")) -> RunContext:
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    if onto.version != cfg.ontology.version:
        raise ValueError(f"ontology version {onto.version} != configured {cfg.ontology.version}")
    templates = {name: load_template(t.path, t.sha256) for name, t in cfg.templates.items()}
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

    run_id, run_hash = make_run_id(
        resolved_config=material_config(cfg) | {"scope": scope},
        ontology_hash=onto.sha256,
        dataset_hash=cfg.dataset.file_sha256,
        corpus_manifest_hash=cman["corpus_manifest_hash"],
        template_hashes={k: t.sha256 for k, t in templates.items()},
        git_commit=git["commit"],
    )
    run_dir = Path(cfg.output.directory) / run_id
    return RunContext(cfg, onto, corpus, cman, templates, run_id, run_hash, run_dir, git, scope)


async def model_snapshot(cfg: Config, model: str) -> dict[str, Any]:
    """The Gateway's catalog entry for the model at run start (versions are not pinnable)."""
    m = cfg.models[model]
    if m.provider != "vercel_ai_gateway":
        return {"id": m.model, "provider": m.provider}
    import os

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{m.endpoint}/v1/models", headers={"Authorization": f"Bearer {os.environ[m.api_key_env]}"})
        r.raise_for_status()
    entries = [e for e in r.json().get("data", []) if e.get("id") == m.model]
    if not entries:
        raise FatalProviderError(f"model {m.model} not listed by the gateway")
    e = entries[0]
    return {k: e.get(k) for k in SNAPSHOT_FIELDS} | {"pricing": e.get("pricing")}


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
        "config_sha256": sha256_obj(material_config(cfg)),
        "resolved_config": material_config(cfg),
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
    mcfg = cfg.models[model]
    if not mcfg.enabled:
        raise ValueError(f"model {model} is disabled in config")
    if adapter is None:
        check_credentials(cfg, [model])

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
            art = build_input(row["article_id"], row["headline"], row["article"], cfg.input)

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
        mman.update(
            {
                "returned_versions": sorted(returned),
                "routing_providers": sorted(providers),
                "status": "COMPLETE" if finished and not fatal else "INCOMPLETE",
                "complete_articles": len(complete_after),
                "last_session": stats | {"elapsed_s": round(time.monotonic() - t0, 1), "ended_at": utc_now(),
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
