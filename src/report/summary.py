"""End-of-run summary built from a finished run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from classifier import results
from classifier.ontology import Ontology

TOP_LEAVES = 15


def _read_rows(run_dir: Path, model: str) -> pl.DataFrame:
    final = run_dir / "results" / f"model={model}"
    files = sorted(final.glob("part-*.parquet"))
    if files:
        return pl.read_parquet(files)
    return results.latest_attempts(results.read_work(run_dir, model))


def _calls(df: pl.DataFrame) -> pl.DataFrame:
    """One row per API request (rows repeat per child); cached decisions excluded."""
    return (df.filter(~pl.col("reused_from_cache"))
              .unique(subset=["article_id", "prediction_key"], keep="first", maintain_order=True))


def ops_section(df: pl.DataFrame, manifest: dict, model: str) -> dict[str, Any]:
    m = manifest["models"][model]
    articles = df["article_id"].n_unique()
    ok_articles = df.group_by("article_id").agg((pl.col("status") == "SUCCESS").all().alias("ok"))["ok"].sum()
    calls = _calls(df)
    ok = calls.filter(pl.col("status") == "SUCCESS")
    lat = ok["latency_ms"].drop_nulls()
    elapsed = m.get("elapsed_s_total") or (m.get("last_session") or {}).get("elapsed_s") or 0.0
    cost = float(ok["provider_cost_usd"].fill_null(0).sum())
    tokens_in = int(ok["input_tokens"].fill_null(0).sum())
    out = {
        "articles": articles,
        "articles_complete": int(ok_articles),
        "api_calls": calls.height,
        "calls_per_article": calls.height / articles if articles else None,
        "calls_by_status": dict(calls.group_by("status").len().iter_rows()),
        "retried_calls": int((calls["attempts"].fill_null(1) > 1).sum()),
        "input_tokens": tokens_in,
        "output_tokens": int(ok["output_tokens"].fill_null(0).sum()),
        "input_tokens_per_call": tokens_in / ok.height if ok.height else None,
        "cost_usd": cost,
        "cost_is_estimate": manifest["resolved_config"]["model"].get("provider") == "cloudflare_workers_ai",
        "elapsed_s": elapsed,
        "articles_per_s": articles / elapsed if elapsed else None,
        "latency_ms": {"p50": lat.quantile(0.5), "p95": lat.quantile(0.95), "max": lat.max()} if lat.len() else None,
    }
    corpus_total = (manifest.get("corpus") or {}).get("stats", {}).get("canonical_articles")
    scope = manifest.get("scope", {})
    if scope.get("kind") != "full_corpus" and corpus_total and articles:
        out["projection_full_corpus"] = {
            "articles": corpus_total,
            "cost_usd": cost / articles * corpus_total,
            "hours_at_same_concurrency": (elapsed / articles * corpus_total / 3600) if elapsed else None,
        }
    return out


def label_section(df: pl.DataFrame, onto: Ontology) -> dict[str, Any]:
    ok = df.filter(pl.col("status") == "SUCCESS")
    n = df["article_id"].n_unique()
    root = onto.roots[0]
    pos = ok.filter(pl.col("selected_positive"))

    def share(child_ids: list[str]) -> dict[str, float]:
        counts = dict(pos.filter(pl.col("child_id").is_in(child_ids)).group_by("child_id")
                      .agg(pl.col("article_id").n_unique()).iter_rows())
        return {c: counts.get(c, 0) / n for c in child_ids}

    arms = list(onto.node("relevance.gsib_activity").children) if "relevance.gsib_activity" in onto.nodes else []
    reached = dict(ok.filter(pl.col("node_id").is_in(arms)).group_by("node_id")
                   .agg(pl.col("article_id").n_unique()).iter_rows())
    # Business-activity leaves only; the relevance classes have their own row.
    leaves = [nid for nid, nd in onto.nodes.items() if nd.is_leaf and nd.parent != root]
    leaf_pos = pos.filter(pl.col("child_id").is_in(leaves))
    per_article = leaf_pos.group_by("article_id").len()
    top = (leaf_pos.group_by("child_id").agg(pl.col("article_id").n_unique().alias("k"))
           .sort("k", descending=True).head(TOP_LEAVES))
    root_scores = ok.filter(pl.col("node_id") == root)["score"]
    hist = [0] * 10
    for s in root_scores.to_list():
        hist[min(9, int(s * 10))] += 1
    return {
        "articles": n,
        "relevance": share(list(onto.node(root).children)),
        "business_arm_positive": share(arms),
        "business_arm_reached": {a: reached.get(a, 0) / n for a in arms},
        "top_leaves": {c: k / n for c, k in top.iter_rows()},
        "mean_positive_leaves": per_article["len"].sum() / n if n else 0.0,
        "share_zero_leaves": 1 - per_article.height / n if n else None,
        "root_score_histogram": hist,
    }


def _positives(df: pl.DataFrame) -> dict[str, set[str]]:
    pos = df.filter((pl.col("status") == "SUCCESS") & pl.col("selected_positive"))
    out = {a: set() for a in df["article_id"].unique().to_list()}
    for a, c in pos.select("article_id", "child_id").iter_rows():
        out[a].add(c)
    return out


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement_section(run_dir: Path, manifest: dict, model: str, df: pl.DataFrame, onto: Ontology) -> dict[str, Any]:
    root = onto.roots[0]
    root_kids = list(onto.node(root).children)
    leaves = {nid for nid, nd in onto.nodes.items() if nd.is_leaf and nd.parent != root}
    mine = _positives(df)
    out: dict[str, Any] = {}
    for other_dir in sorted(run_dir.parent.iterdir()):
        mf = other_dir / "manifest.json"
        if other_dir == run_dir or not mf.exists():
            continue
        om = json.loads(mf.read_text(encoding="utf-8"))
        if om.get("shared_inputs_sha256") != manifest.get("shared_inputs_sha256"):
            continue
        if bool(om.get("dry_run")) != bool(manifest.get("dry_run")):
            continue
        for omodel in om.get("models", {}):
            if omodel == model:
                continue
            theirs = _positives(_read_rows(other_dir, omodel))
            common = sorted(set(mine) & set(theirs))
            if not common:
                continue
            rel = {}
            for c in root_kids:
                a = [int(c in mine[x]) for x in common]
                b = [int(c in theirs[x]) for x in common]
                rel[c] = {"agreement": sum(x == y for x, y in zip(a, b)) / len(common), "kappa": cohen_kappa(a, b)}
            jac = []
            for x in common:
                la, lb = mine[x] & leaves, theirs[x] & leaves
                jac.append(1.0 if not la and not lb else len(la & lb) / len(la | lb))
            out[omodel] = {"run_id": om["run_id"], "articles_compared": len(common), "relevance": rel,
                           "leaf_jaccard_mean": sum(jac) / len(jac)}
    return out


def build(run_dir: Path, model: str, onto: Ontology, devset: dict | None = None) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    df = _read_rows(run_dir, model)
    return {
        "run_id": manifest["run_id"],
        "model": model,
        "model_version": manifest["models"][model]["requested"],
        "dry_run": bool(manifest.get("dry_run")),
        "scope": manifest.get("scope"),
        "ontology": manifest.get("ontology"),
        "ops": ops_section(df, manifest, model),
        "labels": label_section(df, onto),
        "agreement": agreement_section(run_dir, manifest, model, df, onto),
        "devset": {k: v for k, v in devset.items() if k != "detail"} if devset else None,
    }


def write(report: dict, reports_dir: Path) -> tuple[Path, Path]:
    from .render import render

    reports_dir.mkdir(parents=True, exist_ok=True)
    j = reports_dir / f"{report['run_id']}.json"
    md = reports_dir / f"{report['run_id']}.md"
    j.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md.write_text(render(report), encoding="utf-8")
    return j, md
