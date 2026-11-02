"""Graded scoring study, part 3: G-SIB arms on benchmarks with known answers.

Known answers for relevance.gsib_activity (from each benchmark's provenance `group`, fixed before any run):
    fomc               every statement "no" (central bank)
    credit_agreements  every agreement "yes" (bank lending); also business.lending "yes"
    edgar_gsib         bank filers "yes", non-bank controls "no"
    wiki_industry      Banking "yes"; Financial services left out (ambiguous); other industries "no"

Per arm: share flagged in each class at 0.5, AUC where both classes exist, and over all four pooled: AUC, F1 at
0.5, F1 at the cutoff tuned on the 300 Claude-labelled Bloomberg articles (--tune name=run, a transfer test),
and false positives at the recall the first arm gets at 0.5.

    .venv/bin/python scripts/compare_graded_bench.py --runs runs.json [--tune jev=... jev_v2a=... jev_graded=...]
runs.json = {"fomc": {"jev": "<run dir>", "jev_v2a": "...", "jev_graded": "..."}, "edgar": {...}, ...}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import polars as pl

CORPUS = {"fomc": "corpus_bench/fomc", "edgar": "corpus_bench/edgar_gsib",
          "credit_agreements": "corpus_bench/credit_agreements", "wiki_industry": "corpus_bench/wiki_industry"}


def truth(name: str, group: str) -> int | None:
    if name == "fomc":
        return 0
    if name == "credit_agreements":
        return 1
    if name == "edgar":
        return 1 if group.startswith("bank:") else 0
    return 1 if group == "Banking" else (None if group == "Financial services" else 0)


def scores(run_dir: str, node: str) -> dict[str, float]:
    model = next(iter(json.loads((Path(run_dir) / "manifest.json").read_text())["models"]))
    files = sorted((Path(run_dir) / "results" / f"model={model}").glob("*.parquet"))
    df = (pl.scan_parquet(files).filter((pl.col("status") == "SUCCESS") & (pl.col("child_id") == node))
          .select("article_id", "score").collect())
    return dict(df.iter_rows())


def auc(y, s):
    d = pl.DataFrame({"y": y, "s": s}).with_columns(pl.col("s").rank("average").alias("r"))
    P = sum(y)
    N = len(y) - P
    return float((d.filter(pl.col("y") == 1)["r"].sum() - P * (P + 1) / 2) / (P * N)) if P and N else float("nan")


def f1(y, s, t):
    tp = sum(a and b >= t for a, b in zip(y, s))
    fp = sum((not a) and b >= t for a, b in zip(y, s))
    fn = sum(a and b < t for a, b in zip(y, s))
    return (2 * tp / (2 * tp + fp + fn) if tp else 0.0), tp, fp, fn


def fp_at_recall(y, s, rec):
    need = rec * sum(y) - 1e-9
    tp = fp = 0
    for _, yi in sorted(zip(s, y), key=lambda t: -t[0]):
        if tp >= need:
            break
        tp += yi
        fp += 1 - yi
    return fp


def tuned_cutoff(run_dir: str) -> float:
    key = {r["label_id"]: r["article_id"] for r in csv.DictReader(open("labeling/queue_v1_key.csv"))}
    lab = {key[r["label_id"]]: int(r["gsib_activity"] == "y") for r in csv.DictReader(open("labeling/claude_labels_v1.csv"))}
    sc = scores(run_dir, "relevance.gsib_activity")
    ids = sorted(lab)
    y = [lab[a] for a in ids]
    s = [sc.get(a, 0.0) for a in ids]
    return max([i / 100 for i in range(1, 100)], key=lambda t: f1(y, s, t)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tune", nargs="*", default=[])
    a = ap.parse_args()
    runs = json.loads(Path(a.runs).read_text())
    arms = list(next(iter(runs.values())))
    cut = {n: tuned_cutoff(d) for n, d in (t.split("=", 1) for t in a.tune)}

    pooled = {arm: ([], []) for arm in arms}
    print("## Per benchmark (share flagged G-SIB activity at 0.5)\n")
    print("| benchmark | class | n | " + " | ".join(arms) + " |\n|---|---|---|" + "---|" * len(arms))
    aucs = {}
    lend = {}
    for name, by_arm in runs.items():
        prov = pl.read_parquet(Path(CORPUS[name]) / "provenance.parquet").select("article_id", "group").unique("article_id")
        lab = {aid: truth(name, g or "") for aid, g in prov.iter_rows()}
        lab = {k: v for k, v in lab.items() if v is not None}
        sc = {arm: scores(d, "relevance.gsib_activity") for arm, d in by_arm.items()}
        ids = sorted(k for k in lab if all(k in sc[arm] for arm in arms))
        for cls in (1, 0):
            cid = [i for i in ids if lab[i] == cls]
            if cid:
                cells = [f"{sum(sc[arm][i] >= 0.5 for i in cid) / len(cid):.1%}" for arm in arms]
                print(f"| {name} | {'yes' if cls else 'no'} | {len(cid):,} | " + " | ".join(cells) + " |")
        for arm in arms:
            y = [lab[i] for i in ids]
            s = [sc[arm][i] for i in ids]
            aucs.setdefault(name, {})[arm] = auc(y, s)
            pooled[arm][0].extend(y)
            pooled[arm][1].extend(s)
        if name == "credit_agreements":
            for arm, d in by_arm.items():
                ls = scores(d, "business.lending")
                lend[arm] = sum(ls.get(i, 0.0) >= 0.5 for i in ids) / len(ids)
    print("\n## AUC where both classes exist\n")
    print("| benchmark | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for name, v in aucs.items():
        if not any(x != x for x in v.values()):
            print(f"| {name} | " + " | ".join(f"{v[arm]:.3f}" for arm in arms) + " |")
    ref = arms[0]
    y0, s0 = pooled[ref]
    rec = f1(y0, s0, 0.5)[1] / sum(y0)
    print(f"\n## All four pooled ({len(y0):,} items, {sum(y0):,} yes)\n")
    print("| arm | AUC | F1 at 0.5 | FP at 0.5 | FN at 0.5 | cutoff tuned on 300 labelled | F1 at that cutoff | FP at matched recall |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        y, s = pooled[arm]
        f, tp, fp, fn = f1(y, s, 0.5)
        tc = cut.get(arm)
        ft = f"{tc:.2f} | {f1(y, s, tc)[0]:.3f}" if tc is not None else "- | -"
        print(f"| {arm} | {auc(y, s):.3f} | {f:.3f} | {fp:,} | {fn:,} | {ft} | {fp_at_recall(y, s, rec):,} |")
    print(f"\n(matched recall = {ref}'s at 0.5: {rec:.3f})")
    if lend:
        print("\nCredit agreements flagged business.lending at 0.5: " + ", ".join(f"{k} {v:.1%}" for k, v in lend.items()))


if __name__ == "__main__":
    main()
