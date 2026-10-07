"""Classifier CLI. No evaluation or adjudication commands belong here."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from .config import check_credentials, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
CONFIG = typer.Option(Path("configs/classification.yaml"), "--config", "-c")
SET = typer.Option(None, "--set", help="Config override, e.g. models.jev.concurrency=8 (repeatable)")


def _setup(config: Path, overrides: Optional[list[str]] = None):
    load_dotenv(override=False)
    return load_config(config, overrides)


@app.command()
def validate(config: Path = CONFIG, check_dataset: bool = typer.Option(True), overrides: Optional[list[str]] = SET) -> None:
    """Validate config, ontology, templates, dataset hash, and credentials."""
    from .dataset import verify
    from .ontology import load_ontology
    from .templates import load_template

    cfg = _setup(config, overrides)
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    typer.echo(f"ontology ok: {onto.id} v{onto.version}, {len(onto.nodes)} nodes")
    for name, t in cfg.templates.items():
        load_template(t.path, t.sha256, t.kind).check_ontology(onto)
        typer.echo(f"template ok: {name}")
    if check_dataset:
        path = Path(cfg.dataset.local_dir) / cfg.dataset.file
        verify(path, cfg.dataset.file_sha256)
        typer.echo(f"dataset ok: {path}")
    enabled = [m for m, c in cfg.models.items() if c.enabled]
    check_credentials(cfg, enabled)
    typer.echo(f"credentials ok for: {', '.join(enabled)}")


@app.command()
def prepare(config: Path = CONFIG, workers: int = typer.Option(8)) -> None:
    """Fetch + verify the pinned dataset, ingest, deduplicate, and freeze the canonical corpus."""
    from .dataset import fetch, ingest
    from .dedupe import dedupe, write_corpus

    cfg = _setup(config)
    t0 = time.monotonic()
    path = fetch(cfg.dataset)
    df = ingest(path)
    typer.echo(f"ingested {df.height} rows in {time.monotonic() - t0:.0f}s")
    canonical, clusters, stats = dedupe(df, cfg.corpus.dedupe, cfg.run.seed, workers=workers)
    man = write_corpus(Path(cfg.corpus.directory), canonical, clusters, stats, cfg.corpus.dedupe,
                       cfg.corpus.normalization_version, cfg.dataset.file_sha256, cfg.run.seed)
    typer.echo(json.dumps(man["stats"], indent=2))
    typer.echo(f"corpus_manifest_hash: {man['corpus_manifest_hash']}  ({time.monotonic() - t0:.0f}s)")


@app.command()
def probe(config: Path = CONFIG, model: str = typer.Option(..., "--model", "-m")) -> None:
    """One live decision on a toy article: checks request shape, returned model, routing, usage."""
    from .adapters import make_adapter
    from .adapters.base import ArticleInput, ClassificationContext
    from .ontology import load_ontology
    from .templates import load_template

    cfg = _setup(config)
    mcfg = cfg.models[model]
    check_credentials(cfg, [model])
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    tmpl = load_template(cfg.templates[mcfg.template].path, cfg.templates[mcfg.template].sha256, cfg.templates[mcfg.template].kind)
    text = ("Goldman Sachs and JPMorgan led a $5 billion bond sale for Oracle, and JPMorgan provided a "
            "$2 billion bridge loan to fund the acquisition.")
    art = ArticleInput("probe", text, "probe", len(text), len(text), False, "head")
    root = onto.node(onto.roots[0])

    async def go():
        adapter = make_adapter(model, mcfg)
        try:
            return await adapter.classify(art, root, onto.children(root.id), ClassificationContext(mcfg.model, tmpl))
        finally:
            await adapter.aclose()

    d = asyncio.run(go())
    typer.echo(json.dumps({"scores": d.scores, "model_returned": d.model_returned, "routing_provider": d.routing_provider,
                           "latency_ms": round(d.latency_ms or 0), "input_tokens": d.input_tokens,
                           "output_tokens": d.output_tokens, "cost_usd": d.cost_usd, "attempts": d.attempts,
                           "generation_id": d.generation_id}, indent=2))


@app.command()
def run(
    config: Path = CONFIG,
    model: Optional[list[str]] = typer.Option(None, "--model", "-m", help="Model key(s); default: all enabled"),
    resume: bool = typer.Option(False, "--resume"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Engineering run on the first N canonical articles"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Use the offline fake adapter; no network"),
    max_cost_usd: Optional[float] = typer.Option(None, "--max-cost-usd"),
    progress_every: int = typer.Option(500, "--progress-every"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Results location (not part of run identity)"),
    overrides: Optional[list[str]] = SET,
    show: int = typer.Option(0, "--show", help="Print the traversal tree for the first N articles"),
    dev_eval: bool = typer.Option(True, "--dev-eval/--no-dev-eval", help="Score the synthetic dev set in the report"),
    reports_dir: Path = typer.Option(Path("reports"), "--reports-dir"),
) -> None:
    """Classify the canonical corpus (or the first N articles) with the selected models, then report."""
    from .adapters.fake import FakeAdapter
    from .runner import prepare_run, run_model

    sets = list(overrides or [])
    if output_dir is not None:
        sets.append(f"output.directory={output_dir}")
    cfg = _setup(config, sets)
    models = model or [m for m, c in cfg.models.items() if c.enabled]
    finished = []
    for m in models:
        rc = prepare_run(cfg, model=m, limit=limit)
        if dry_run:
            rc.run_dir = rc.run_dir.with_name(rc.run_dir.name + "_dryrun")
            rc.run_id = rc.run_dir.name
            rc.dry_run = True
        typer.echo(f"{m}: run_id {rc.run_id} -> {rc.run_dir}")
        adapter = FakeAdapter(m) if dry_run else None
        asyncio.run(run_model(rc, m, resume=resume, max_cost_usd=max_cost_usd, adapter=adapter,
                              progress_every=progress_every))
        finished.append(rc)
    # Reports run after all models so cross-model agreement sees every run of this invocation.
    for rc in finished:
        _report(rc, show=show, dev_eval=dev_eval, reports_dir=reports_dir)


def _report(rc, *, show: int, dev_eval: bool, reports_dir: Path) -> None:
    """Downstream of classification: reads finished results, prints and saves the report."""
    from report import devset, summary
    from report.render import render, render_tree

    from .adapters import make_adapter
    from .adapters.base import ClassificationContext
    from .adapters.fake import FakeAdapter

    m, mcfg = rc.model, rc.cfg.models[rc.model]
    if show:
        rows = summary._read_rows(rc.run_dir, m)
        h = rc.corpus.select("article_id", "headline")
        typer.echo(f"\n── {m}: first {show} articles (✓ positive at {rc.cfg.hierarchy.classification.positive_threshold}, "
                   f"→ traversed at {rc.cfg.hierarchy.traversal.threshold})")
        for aid, headline in h.head(show).iter_rows():
            typer.echo(render_tree(rows, aid, headline, rc.cfg.hierarchy.classification.positive_threshold,
                                   rc.cfg.hierarchy.traversal.threshold))
    dev = None
    if dev_eval:
        async def go():
            adapter = FakeAdapter(m) if rc.dry_run else make_adapter(m, mcfg)
            try:
                ctx = ClassificationContext(mcfg.model, rc.templates[mcfg.template])
                return await devset.evaluate(adapter, ctx, rc.onto)
            finally:
                await adapter.aclose()

        dev = asyncio.run(go())
    rep = summary.build(rc.run_dir, m, rc.onto, dev)
    typer.echo("\n" + render(rep))
    j, _ = summary.write(rep, reports_dir)
    typer.echo(f"report saved: {j} (+ .md)")


@app.command()
def report(
    config: Path = CONFIG,
    model: str = typer.Option(..., "--model", "-m"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    show: int = typer.Option(0, "--show"),
    dev_eval: bool = typer.Option(False, "--dev-eval/--no-dev-eval"),
    reports_dir: Path = typer.Option(Path("reports"), "--reports-dir"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    overrides: Optional[list[str]] = SET,
) -> None:
    """Re-print the report for an existing run (same config, model and --limit as the run)."""
    from .runner import prepare_run

    sets = list(overrides or [])
    if output_dir is not None:
        sets.append(f"output.directory={output_dir}")
    cfg = _setup(config, sets)
    rc = prepare_run(cfg, model=model, limit=limit)
    if dry_run:
        rc.run_dir = rc.run_dir.with_name(rc.run_dir.name + "_dryrun")
        rc.run_id, rc.dry_run = rc.run_dir.name, True
    if not (rc.run_dir / "manifest.json").exists():
        raise RuntimeError(f"no run at {rc.run_dir}")
    _report(rc, show=show, dev_eval=dev_eval, reports_dir=reports_dir)


@app.command()
def compact(config: Path = CONFIG, run_id: str = typer.Option(...), model: str = typer.Option(..., "--model", "-m")) -> None:
    """Write deduplicated final results for a model (also done automatically when a run completes)."""
    from . import results

    cfg = _setup(config)
    path = results.compact(Path(cfg.output.directory) / run_id, model)
    typer.echo(str(path))


def main() -> None:
    from .adapters.base import FatalProviderError

    try:
        app(standalone_mode=False)
    except FatalProviderError as e:
        typer.secho(f"FATAL: {e}", fg="red", err=True)
        raise SystemExit(2)
    except (RuntimeError, ValueError) as e:
        typer.secho(f"ERROR: {e}", fg="red", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
