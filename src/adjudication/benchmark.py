"""Score a classification run on a benchmark corpus against the corpus's provenance.

Provenance rows carry `group` and `expect` (JSON {node_id: 0|1}, fixed in each source module before
any run). Reported per group: share of items positive for each top-level node and business arm; and,
over all items with an expectation for a node, precision/recall/accuracy of the run on that node.
Several source rows can share one canonical article (exact duplicates); each row is scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from . import queue

REPORT_NODES = ["relevance.gsib_activity", "relevance.market_context_only", "relevance.neither",
                "business.markets", "business.lending", "business.treasury_alm"]


def score(run_dir: Path, model: str, corpus_dir: Path) -> dict:
    prov = pl.read_parquet(corpus_dir / "provenance.parquet")
    canon = set(pl.read_parquet(corpus_dir / "canonical.parquet")["article_id"].to_list())
    clusters = pl.read_parquet(corpus_dir / "clusters.parquet").select("article_id", "canonical_article_id")
    to_canon = dict(clusters.iter_rows())
    pos = queue.positives(run_dir, model)
    groups: dict[str, dict] = {}
    conf: dict[str, list[int]] = {}
    missing = 0
    for r in prov.iter_rows(named=True):
        aid = to_canon.get(r["article_id"], r["article_id"])
        if aid not in pos:
            missing += 1
            continue
        got = pos[aid]
        g = groups.setdefault(r["group"] or "(none)", {"n": 0, **{n: 0 for n in REPORT_NODES}})
        g["n"] += 1
        for n in REPORT_NODES:
            g[n] += int(n in got)
        for node, y in json.loads(r["expect"] or "{}").items():
            c = conf.setdefault(node, [0, 0, 0, 0])  # tp, fp, fn, tn
            p = int(node in got)
            c[0 if (y and p) else 1 if (p and not y) else 2 if (y and not p) else 3] += 1
    per_group = {k: {"n": v["n"], **{n: v[n] / v["n"] for n in REPORT_NODES}} for k, v in sorted(groups.items())}
    expectations = {}
    for node, (tp, fp, fn, tn) in sorted(conf.items()):
        expectations[node] = {
            "items": tp + fp + fn + tn, "expected_positive": tp + fn, "expected_negative": fp + tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "accuracy": (tp + tn) / (tp + fp + fn + tn), "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        }
    return {"canonical_articles": len(canon), "scored_rows": prov.height - missing, "missing_rows": missing,
            "per_group": per_group, "expectations": expectations}


def render(s: dict) -> str:
    f = lambda x: "  n/a" if x is None else f"{x:5.3f}"  # noqa: E731
    lines = [f"scored {s['scored_rows']:,} source rows ({s['missing_rows']:,} without results)", "",
             "per group (share positive):",
             f"  {'group':34s} {'n':>6s}  " + "  ".join(n.split('.')[-1][:10].rjust(10) for n in REPORT_NODES)]
    for g, v in s["per_group"].items():
        lines.append(f"  {g[:34]:34s} {v['n']:6d}  " + "  ".join(f"{v[n]:10.1%}" for n in REPORT_NODES))
    if s["expectations"]:
        lines += ["", "against pre-registered expectations:",
                  f"  {'node':34s} {'items':>7s} {'exp+':>6s} {'exp-':>6s}  precision recall  accuracy  FP rate"]
        for n, e in s["expectations"].items():
            lines.append(f"  {n:34s} {e['items']:7d} {e['expected_positive']:6d} {e['expected_negative']:6d}  "
                         f"{f(e['precision'])}    {f(e['recall'])}  {f(e['accuracy'])}    {f(e['false_positive_rate'])}")
    return "\n".join(lines)
