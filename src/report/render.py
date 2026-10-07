"""Plain-text rendering of run reports and per-article traversal trees."""

from __future__ import annotations

from typing import Any

import polars as pl

BAR = "▏▎▍▌▋▊▉█"


def _pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def _num(x, fmt="{:,.0f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def _short(node_id: str) -> str:
    return node_id.split(".", 1)[1] if "." in node_id else node_id


def _hist(h: list[int]) -> str:
    top = max(h) or 1
    return " ".join(BAR[min(7, int(8 * v / top))] if v else "·" for v in h)


def render(r: dict[str, Any]) -> str:
    o, lab = r["ops"], r["labels"]
    tag = "  [DRY RUN · fake scores, format only]" if r["dry_run"] else ""
    lines = [f"══ {r['model']} ({r['model_version']}) · run {r['run_id']}{tag}",
             f"   scope: {r['scope']}  · ontology {r['ontology']['version']}", ""]

    lines.append("── Cost, time, throughput")
    est = " (estimated)" if o["cost_is_estimate"] else ""
    lat = o["latency_ms"] or {}
    lines += [
        f"   articles {o['articles']:,} ({o['articles_complete']:,} complete) · API calls {o['api_calls']:,}"
        f" ({_num(o['calls_per_article'], '{:.2f}')}/article) · retried {o['retried_calls']}",
        f"   calls by status: {o['calls_by_status']}",
        f"   tokens in {o['input_tokens']:,} ({_num(o['input_tokens_per_call'])}/call) · out {o['output_tokens']:,}",
        f"   cost ${o['cost_usd']:.4f}{est} · wall time {o['elapsed_s']:.1f}s · {_num(o['articles_per_s'], '{:.2f}')} articles/s",
        f"   latency p50 {_num(lat.get('p50'))} ms · p95 {_num(lat.get('p95'))} ms · max {_num(lat.get('max'))} ms",
    ]
    if "projection_full_corpus" in o:
        p = o["projection_full_corpus"]
        lines.append(f"   → full corpus ({p['articles']:,} articles): ${p['cost_usd']:,.2f}{est}, "
                     f"{_num(p['hours_at_same_concurrency'], '{:.1f}')} h at this concurrency")
    lines.append("")

    lines.append("── Label distribution (share of articles, score ≥ 0.5)")
    lines.append("   relevance:     " + "  ".join(f"{_short(k)} {_pct(v)}" for k, v in lab["relevance"].items()))
    lines.append("   business arm:  " + "  ".join(
        f"{_short(k)} {_pct(v)} (reached {_pct(lab['business_arm_reached'][k]).strip()})"
        for k, v in lab["business_arm_positive"].items()))
    lines.append(f"   positive activity leaves per article: {lab['mean_positive_leaves']:.2f} · articles with none: "
                 f"{_pct(lab['share_zero_leaves']).strip()}")
    if lab["top_leaves"]:
        lines.append("   top leaves:    " + "  ".join(f"{k} {_pct(v).strip()}" for k, v in lab["top_leaves"].items()))
    lines.append(f"   root-score histogram 0→1: {_hist(lab['root_score_histogram'])}  {lab['root_score_histogram']}")
    lines.append("")

    lines.append("── Cross-model agreement (descriptive; same articles, ontology, thresholds)")
    if not r["agreement"]:
        lines.append("   no comparable runs yet")
    for other, a in r["agreement"].items():
        rel = "  ".join(f"{_short(k)} {_pct(v['agreement']).strip()} κ={_num(v['kappa'], '{:.2f}')}"
                        for k, v in a["relevance"].items())
        lines.append(f"   vs {other:6s} ({a['articles_compared']:,} articles): {rel} · leaf Jaccard "
                     f"{a['leaf_jaccard_mean']:.2f}")
    lines.append("")

    lines.append("── Dev-set accuracy (synthetic cases; balanced accuracy at 0.5)")
    d = r["devset"]
    if not d:
        lines.append("   skipped")
    else:
        ov = d["overall"]
        lines.append(f"   overall {_num(ov['balanced_accuracy'], '{:.3f}')} (recall {_num(ov['recall'], '{:.2f}')}, "
                     f"specificity {_num(ov['specificity'], '{:.2f}')}) on {ov['n']} checks from {d['cases']} cases"
                     f" · cost ${d['cost_usd']:.4f}" + (f" · errors {d['errors']}" if d["errors"] else ""))
        lines.append("   by level: " + "  ".join(
            f"{k} {_num(v['balanced_accuracy'], '{:.3f}')} (n={v['n']})" for k, v in d["by_level"].items() if v["n"]))
        for m in d["worst_misses"]:
            lines.append(f"   miss: {m['case']:7s} {m['node']:42s} expected {m['y']} got {m['p']:.2f}")
    return "\n".join(lines) + "\n"


def render_tree(rows: pl.DataFrame, article_id: str, headline: str, pos_t: float, trav_t: float) -> str:
    """Visited nodes for one article with every child's score. ✓ positive, → traversed."""
    df = rows.filter(pl.col("article_id") == article_id).sort(["node_depth", "node_id", "child_id"])
    lines = [f"■ {headline[:100]}", f"  {article_id}"]
    for (node_id, depth), g in df.group_by(["node_id", "node_depth"], maintain_order=True):
        lines.append(f"  {'  ' * depth}{node_id}")
        for r in g.iter_rows(named=True):
            if r["status"] != "SUCCESS":
                lines.append(f"  {'  ' * depth}  ✗ {r['child_id']}  [{r['status']}]")
                continue
            mark = ("✓" if r["selected_positive"] else " ") + ("→" if r["selected_for_traversal"] else " ")
            lines.append(f"  {'  ' * depth}  {mark} {r['score']:.2f}  {r['child_id']}")
    return "\n".join(lines)
