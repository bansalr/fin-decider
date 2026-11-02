"""Graded scoring study, part 2: compare G-SIB runs against the 300-article Claude-labelled queue.

Reference labels: labeling/claude_labels_v1.csv (gitignored; one careful labeller, not ground truth), keyed via
labeling/queue_v1_key.csv. Reports, per run, for relevance.gsib_activity: ROC AUC, F1 at 0.5, F1 with a cutoff
tuned by 2-fold cross-validation, and false positives at a matched recall; paired bootstrap CIs against the
first run. Then business lines (markets / lending / treasury_alm) and activity categories (leaves) on the
articles Claude labelled G-SIB activity.

    .venv/bin/python scripts/compare_graded_gsib.py jev=classification_results/jev-... graded=classification_results/jev_graded-...
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import polars as pl

BL = {"markets": "business.markets", "lending": "business.lending", "treasury_alm": "business.treasury_alm"}


def load(run_dir: str, ids: list[str]) -> dict[tuple[str, str], float]:
    model = next(iter(json.loads((Path(run_dir) / "manifest.json").read_text())["models"]))
    files = sorted((Path(run_dir) / "results" / f"model={model}").glob("*.parquet"))
    df = (pl.scan_parquet(files).filter((pl.col("status") == "SUCCESS") & pl.col("article_id").is_in(ids))
          .select("article_id", "child_id", "score").collect())
    return {(a, c): s for a, c, s in df.select("article_id", "child_id", "score").iter_rows()}


def auc(y, s):
    d = pl.DataFrame({"y": y, "s": s}).with_columns(pl.col("s").rank("average").alias("r"))
    P = sum(y); N = len(y) - P
    return float((d.filter(pl.col("y") == 1)["r"].sum() - P * (P + 1) / 2) / (P * N)) if P and N else float("nan")


def prf(y, s, t):
    tp = sum(a and b >= t for a, b in zip(y, s)); fp = sum((not a) and b >= t for a, b in zip(y, s))
    fn = sum(a and b < t for a, b in zip(y, s))
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), fp


def best_t(y, s):
    return max([i / 100 for i in range(1, 100)], key=lambda t: prf(y, s, t)[2])


def cv_f1(y, s, ids):
    """2-fold: tune the cutoff on one half (by article-id hash), apply to the other; pooled F1."""
    fold = [int(a[-1], 16) % 2 for a in ids]
    pred = [0] * len(y)
    for k in (0, 1):
        tr = [i for i in range(len(y)) if fold[i] != k]
        t = best_t([y[i] for i in tr], [s[i] for i in tr])
        for i in range(len(y)):
            if fold[i] == k:
                pred[i] = int(s[i] >= t)
    return prf(y, pred, 0.5)[2]


def fp_at_recall(y, s, rec):
    need = rec * sum(y) - 1e-9; tp = fp = 0
    for sc, yi in sorted(zip(s, y), key=lambda t: -t[0]):
        if tp >= need:
            break
        tp += yi; fp += 1 - yi
    return fp


def main() -> None:
    key = {r["label_id"]: r for r in csv.DictReader(open("labeling/queue_v1_key.csv"))}
    lab = {key[r["label_id"]]["article_id"]: r for r in csv.DictReader(open("labeling/claude_labels_v1.csv"))}
    ids = sorted(lab)
    runs = {n: load(d, ids) for n, d in (r.split("=", 1) for r in sys.argv[1:])}
    y = [int(lab[a]["gsib_activity"] == "y") for a in ids]
    ref = next(iter(runs))
    sc = {n: [r.get((a, "relevance.gsib_activity"), 0.0) for a in ids] for n, r in runs.items()}
    ref_recall = prf(y, sc[ref], 0.5)[1]
    print(f"G-SIB activity on {len(ids)} labelled articles ({sum(y)} Claude-positive); matched recall = {ref}'s at 0.5 ({ref_recall:.2f})\n")
    print(f"| run | AUC | precision@0.5 | recall@0.5 | F1@0.5 | F1, 2-fold tuned cutoff | FP@0.5 | FP at matched recall |\n|---|---|---|---|---|---|---|---|")
    for n, s in sc.items():
        p, r, f, fp = prf(y, s, 0.5)
        print(f"| {n} | {auc(y, s):.3f} | {p:.2f} | {r:.2f} | {f:.2f} | {cv_f1(y, s, ids):.2f} | {fp} | {fp_at_recall(y, s, ref_recall)} |")
    random.seed(0)
    print(f"\npaired bootstrap vs {ref} (1,000 resamples of articles), 95% CI:")
    for n in list(runs)[1:]:
        da, df = [], []
        for _ in range(1000):
            ix = [random.randrange(len(y)) for _ in y]
            yy = [y[i] for i in ix]
            if 0 < sum(yy) < len(yy):
                da.append(auc(yy, [sc[n][i] for i in ix]) - auc(yy, [sc[ref][i] for i in ix]))
                df.append(fp_at_recall(yy, [sc[n][i] for i in ix], ref_recall) - fp_at_recall(yy, [sc[ref][i] for i in ix], ref_recall))
        da.sort(); df.sort(); k = len(da)
        print(f"  {n}: AUC diff {auc(y, sc[n]) - auc(y, sc[ref]):+.3f} [{da[int(.025 * k)]:+.3f}, {da[int(.975 * k)]:+.3f}];"
              f" FP-at-matched-recall diff [{df[int(.025 * k)]:+d}, {df[int(.975 * k)]:+d}]")

    pos = [a for a in ids if lab[a]["gsib_activity"] == "y"]
    print(f"\nBusiness lines on the {len(pos)} Claude-positive articles (score 0 where the model did not descend):\n")
    print("| run | " + " | ".join(f"{b} AUC / F1@0.5" for b in BL) + " |\n|---|---|---|---|")
    for n, r in runs.items():
        cells = []
        for b, node in BL.items():
            yb = [int(lab[a][b] == "y") for a in pos]; sb = [r.get((a, node), 0.0) for a in pos]
            cells.append(f"{auc(yb, sb):.3f} / {prf(yb, sb, 0.5)[2]:.2f}")
        print(f"| {n} | " + " | ".join(cells) + " |")
    from classifier.ontology import load_ontology
    onto = load_ontology("gsib_basel_ontology_v1.1.json")
    leaves = sorted(nid for nid, nd in onto.nodes.items() if not onto.children(nid)
                    and nid.split(".")[0] in ("markets", "lending", "treasury"))
    print(f"\nActivity categories (leaves) on the same {len(pos)} articles, pooled over {len(leaves)} leaves:\n")
    print("| run | leaf AUC (pooled) | precision@0.5 | recall@0.5 | F1@0.5 |\n|---|---|---|---|---|")
    for n, r in runs.items():
        yl, sl = [], []
        for a in pos:
            gold = {x for x in lab[a]["leaves"].split(";") if x}
            for c in leaves:
                yl.append(int(c in gold)); sl.append(r.get((a, c), 0.0))
        p, rc, f, _ = prf(yl, sl, 0.5)
        print(f"| {n} | {auc(yl, sl):.3f} | {p:.2f} | {rc:.2f} | {f:.2f} |")


if __name__ == "__main__":
    main()
