"""Compare yes/no (boolean) and graded-relevance runs on the Reuters-21578 R(10) native task.

Experiment tool for the graded scoring study (native_benchmarks/reuters_top10/GRADED.md). For each run it
reports, on the same stories:
    macro F1 at the config threshold (0.50)
    macro F1 at a single threshold tuned on a separate tuning set (--tune-run-* / --tune-first), if given
    per-topic and macro ROC AUC and average precision (threshold-free ranking quality)
    false positives at 0.50, and at matched recall: for each topic, the FPs each run needs to reach the
    recall the reference run (the first one) gets at 0.50

Stories are scored per source row, as in score_native_reuters.py (exact duplicates share their canonical
story's scores). Select stories with --first N (the --limit N dev sample) or --skip-first N (hold-out).

    .venv/bin/python scripts/compare_graded_reuters.py --corpus corpus_native/reuters_top10_train --first 1000 \\
        boolean=classification_results_native/jev-16bc31eb63e4f789 graded=classification_results_native/jev-...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

TOP10 = ("earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn")


def load_scores(run_dir: Path) -> dict[tuple[str, str], float]:
    model = next(iter(json.loads((run_dir / "manifest.json").read_text())["models"]))
    files = sorted((run_dir / "results" / f"model={model}").glob("*.parquet")) or \
        sorted((run_dir / "_work" / f"model={model}").glob("*.parquet"))
    df = pl.read_parquet(files).filter(pl.col("status") == "SUCCESS").unique(["article_id", "child_id"], keep="last")
    return {(a, c.removeprefix("topic.")): s for a, c, s in df.select("article_id", "child_id", "score").iter_rows()}


def rows(corpus: Path, first: int, skip_first: int) -> list[tuple[str, set[str]]]:
    """-> [(canonical article_id, gold topics)] per source row in the selected slice."""
    ids = pl.read_parquet(corpus / "canonical.parquet", columns=["article_id"]).sort("article_id")["article_id"]
    if first:
        keep = set(ids.head(first).to_list())
    else:
        keep = set(ids.slice(skip_first).to_list())
    to_canon = dict(pl.read_parquet(corpus / "clusters.parquet").select("article_id", "canonical_article_id").iter_rows())
    prov = pl.read_parquet(corpus / "provenance.parquet").select("article_id", "label")
    return [(to_canon.get(a, a), set(lab.split(";"))) for a, lab in prov.iter_rows() if to_canon.get(a, a) in keep]


def auc(y: list[int], s: list[float]) -> float:
    d = pl.DataFrame({"y": y, "s": s}).with_columns(pl.col("s").rank("average").alias("r"))
    P = sum(y)
    N = len(y) - P
    return float((d.filter(pl.col("y") == 1)["r"].sum() - P * (P + 1) / 2) / (P * N)) if P and N else float("nan")


def avg_precision(y: list[int], s: list[float]) -> float:
    pairs = sorted(zip(s, y), key=lambda t: -t[0])
    P = sum(y)
    tp = 0
    ap = 0.0
    for i, (_, yi) in enumerate(pairs, 1):
        if yi:
            tp += 1
            ap += tp / i
    return ap / P if P else float("nan")


def f1_at(y: list[int], s: list[float], t: float) -> tuple[float, int, int, int]:
    tp = sum(1 for a, b in zip(y, s) if a and b >= t)
    fp = sum(1 for a, b in zip(y, s) if not a and b >= t)
    fn = sum(1 for a, b in zip(y, s) if a and b < t)
    return (2 * tp / (2 * tp + fp + fn) if tp else 0.0), tp, fp, fn


def fp_at_recall(y: list[int], s: list[float], recall: float) -> int:
    """Fewest false positives at which recall >= `recall` (scanning thresholds from the top)."""
    P = sum(y)
    need = recall * P - 1e-9
    tp = fp = 0
    for score, yi in sorted(zip(s, y), key=lambda t: -t[0]):
        if tp >= need:
            break
        tp += yi
        fp += 1 - yi
    return fp


def vectors(rws, scores, topic):
    y = [int(topic in g) for _, g in rws]
    s = [scores[(a, topic)] for a, _ in rws]
    return y, s


def macro_f1(rws, scores, t):
    return sum(f1_at(*vectors(rws, scores, tp), t)[0] for tp in TOP10) / len(TOP10)


def best_threshold(rws, scores):
    grid = [i / 100 for i in range(1, 100)]
    return max(grid, key=lambda t: macro_f1(rws, scores, t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--skip-first", type=int, default=0)
    ap.add_argument("--tune-corpus", default=None, help="corpus for threshold tuning (default: --corpus)")
    ap.add_argument("--tune-first", type=int, default=0, help="tune thresholds on the first N stories of the tune corpus")
    ap.add_argument("--tune", nargs="*", default=[], help="name=run_dir per run, the runs to tune each threshold on")
    ap.add_argument("--json", default=None)
    ap.add_argument("runs", nargs="+", help="name=run_dir; the first is the reference for matched recall")
    a = ap.parse_args()

    runs = {n: load_scores(Path(d)) for n, d in (r.split("=", 1) for r in a.runs)}
    rws = [r for r in rows(Path(a.corpus), a.first, a.skip_first)
           if all((r[0], t) in sc for sc in runs.values() for t in TOP10)]
    tuned = {}
    if a.tune:
        tc = Path(a.tune_corpus or a.corpus)
        for n, d in (r.split("=", 1) for r in a.tune):
            sc = load_scores(Path(d))
            trw = [r for r in rows(tc, a.tune_first, 0) if all((r[0], t) in sc for t in TOP10)]
            tuned[n] = best_threshold(trw, sc)
    ref = next(iter(runs))
    ref_recall = {t: (lambda y, s: f1_at(y, s, 0.5)[1] / max(1, sum(y)))(*vectors(rws, runs[ref], t)) for t in TOP10}

    out = {"stories": len(rws), "runs": {}}
    for n, sc in runs.items():
        per = {}
        for t in TOP10:
            y, s = vectors(rws, sc, t)
            f1, tp, fp, fn = f1_at(y, s, 0.5)
            per[t] = {"support": sum(y), "f1@0.5": f1, "fp@0.5": fp, "fn@0.5": fn, "auc": auc(y, s),
                      "ap": avg_precision(y, s), "fp@ref_recall": fp_at_recall(y, s, ref_recall[t])}
        r = {"per_topic": per, "macro_f1@0.5": macro_f1(rws, sc, 0.5),
             "macro_auc": sum(v["auc"] for v in per.values()) / 10, "macro_ap": sum(v["ap"] for v in per.values()) / 10,
             "fp@0.5": sum(v["fp@0.5"] for v in per.values()), "fn@0.5": sum(v["fn@0.5"] for v in per.values()),
             "fp@ref_recall": sum(v["fp@ref_recall"] for v in per.values()),
             "oracle_threshold": best_threshold(rws, sc)}
        r["macro_f1@oracle"] = macro_f1(rws, sc, r["oracle_threshold"])
        if n in tuned:
            r["tuned_threshold"] = tuned[n]
            r["macro_f1@tuned"] = macro_f1(rws, sc, tuned[n])
        out["runs"][n] = r

    pc = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    print(f"{len(rws):,} stories (source rows); reference for matched recall: {ref}\n")
    print(f"{'run':14s} {'F1@0.5':>7s} {'F1@tuned':>9s} {'(t)':>5s} {'AUC':>6s} {'AP':>6s} {'FP@0.5':>7s} {'FN@0.5':>7s} {'FP@ref-recall':>14s}")
    for n, r in out["runs"].items():
        tf = f"{pc(r['macro_f1@tuned'])}  {r['tuned_threshold']:.2f}" if "tuned_threshold" in r else "      -     "
        print(f"{n:14s} {pc(r['macro_f1@0.5']):>7s} {tf:>15s} {pc(r['macro_auc']):>6s} {pc(r['macro_ap']):>6s} "
              f"{r['fp@0.5']:7d} {r['fn@0.5']:7d} {r['fp@ref_recall']:14d}")
    print("\nper topic: F1@0.5 / AP / FP@0.5 / FP at reference recall")
    print(f"{'topic':10s}" + "".join(f"{n:>28s}" for n in runs))
    for t in TOP10:
        cells = []
        for n in runs:
            v = out["runs"][n]["per_topic"][t]
            cells.append(f"{pc(v['f1@0.5'])} {pc(v['ap'])} {v['fp@0.5']:5d} {v['fp@ref_recall']:5d}")
        print(f"{t:10s}" + "".join(f"{c:>28s}" for c in cells))
    print("\n(oracle thresholds, tuned on these same stories, for reference only: "
          + ", ".join(f"{n} {r['oracle_threshold']:.2f} -> {pc(r['macro_f1@oracle'])}" for n, r in out["runs"].items()) + ")")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
