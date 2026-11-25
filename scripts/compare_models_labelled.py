"""Score several Bloomberg G-SIB runs against the 300-article labelled queue, node by node.

Reference labels: labeling/claude_labels_v1.csv (gitignored), keyed via labeling/queue_v1_key.csv. The queue is a
stratified sample of the 14,699 articles all hosted models ran (strata: split / all_yes / all_no), so each label
carries a stratum weight; weighted results estimate the 14,699-article population (and, since that slice is an
effectively random slice of the hash-ordered corpus, the whole corpus).

Reports, per run:
  - G-SIB activity: weighted precision / recall / F1 with stratified-bootstrap 95% intervals, accuracy, AUC
  - top-level and business-line nodes: weighted precision / recall / F1
  - the 31 activity categories pooled over all 300 articles (unweighted counts)

    .venv/bin/python scripts/compare_models_labelled.py Jev=classification_results/jev-... Mo=classification_results/mo-...
A run path may be a run directory (results/ or _work/ parquet is found automatically).
"""

from __future__ import annotations

import csv
import glob
import json
import random
import sys
from pathlib import Path

import polars as pl

from classifier.ontology import load_ontology

TOP = ["relevance.gsib_activity", "relevance.market_context_only", "relevance.neither"]
LINES = ["business.markets", "business.lending", "business.treasury_alm"]


def files(run: str) -> list[str]:
    model = next(iter(json.loads((Path(run) / "manifest.json").read_text())["models"]))
    return glob.glob(f"{run}/results/model={model}/*.parquet") or glob.glob(f"{run}/_work/model={model}/*.parquet")


def main() -> None:
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    key = {r["label_id"]: r for r in csv.DictReader(open("labeling/queue_v1_key.csv"))}
    lab = {key[r["label_id"]]["article_id"]: r for r in csv.DictReader(open("labeling/claude_labels_v1.csv"))}
    ids = sorted(lab)
    w = {a: float(key[lab[a]["label_id"]]["weight"]) for a in ids}
    stratum = {a: key[lab[a]["label_id"]]["stratum"] for a in ids}
    onto = load_ontology("gsib_basel_ontology_v1.1.json")
    leaves = [n for n in onto.nodes if not onto.children(n) and n.split(".")[0] in ("markets", "lending", "treasury")]

    def gold(a, node):
        r = lab[a]
        if node in TOP:
            return r[node.split(".")[1]] == "y"
        if node in LINES:
            return r["gsib_activity"] == "y" and r[node.split(".")[1]] == "y"
        return node in {x for x in r["leaves"].split(";") if x}

    pos, score = {}, {}
    for name, run in runs.items():
        d = (pl.scan_parquet(files(run)).filter(pl.col("article_id").is_in(ids) & (pl.col("status") == "SUCCESS"))
             .select("article_id", "child_id", "score", "selected_positive").unique(["article_id", "child_id"]).collect())
        missing = set(ids) - set(d["article_id"].unique().to_list())
        if missing:
            print(f"! {name}: {len(missing)} labelled articles have no results")
        pos[name] = {(a, c) for a, c, _, p in d.iter_rows() if p}
        score[name] = {(a, c): s for a, c, s, _ in d.iter_rows()}

    def prf(name, node, idx=None, weighted=True):
        tp = fp = fn = tn = 0.0
        for i in (idx if idx is not None else range(len(ids))):
            a = ids[i]
            ww = w[a] if weighted else 1.0
            y, p = gold(a, node), (a, node) in pos[name]
            tp += ww * (y and p); fp += ww * ((not y) and p); fn += ww * (y and not p); tn += ww * ((not y) and not p)
        P = tp / (tp + fp) if tp + fp else float("nan")
        R = tp / (tp + fn) if tp + fn else float("nan")
        F = 2 * P * R / (P + R) if tp else 0.0
        return P, R, F, (tp + tn) / (tp + fp + fn + tn)

    def auc(name, node):
        y = [int(gold(a, node)) for a in ids]
        s = [score[name].get((a, node), 0.0) for a in ids]
        d = pl.DataFrame({"y": y, "s": s}).with_columns(pl.col("s").rank("average").alias("r"))
        P = sum(y); N = len(y) - P
        return float((d.filter(pl.col("y") == 1)["r"].sum() - P * (P + 1) / 2) / (P * N))

    strata = {}
    for i, a in enumerate(ids):
        strata.setdefault(stratum[a], []).append(i)
    rng = random.Random(0)
    boots = [[i for s in strata.values() for i in (rng.choice(s) for _ in s)] for _ in range(1000)]
    f2 = lambda x: "—" if x != x else f"{x:.2f}"  # noqa: E731

    print("## G-SIB activity, estimated corpus-wide (weighted; 95% CI from a stratified bootstrap)\n")
    print("| model | precision | recall | F1 | accuracy | AUC (unweighted) |\n|---|---|---|---|---|---|")
    for name in runs:
        P, R, F, A = prf(name, TOP[0])
        bs = [prf(name, TOP[0], b) for b in boots]
        ci = lambda k: sorted(x[k] for x in bs if x[k] == x[k])  # noqa: E731
        c = [ci(k) for k in range(4)]
        q = lambda k: f"[{c[k][int(.025 * len(c[k]))]:.2f}–{c[k][int(.975 * len(c[k])) - 1]:.2f}]"  # noqa: E731
        print(f"| {name} | {P:.2f} {q(0)} | {R:.2f} {q(1)} | {F:.2f} {q(2)} | {A:.2f} | {auc(name, TOP[0]):.3f} |")

    print("\n## Unweighted, on the 300-article sample (G-SIB activity)\n")
    print("| model | precision | recall | F1 |\n|---|---|---|---|")
    for name in runs:
        P, R, F, _ = prf(name, TOP[0], weighted=False)
        print(f"| {name} | {P:.2f} | {R:.2f} | {F:.2f} |")

    print("\n## Per node, weighted (precision / recall / F1)\n")
    print("| node | " + " | ".join(runs) + " |\n|---|" + "---|" * len(runs))
    for node in TOP + LINES:
        print(f"| {onto.nodes[node].name} | " + " | ".join("{} / {} / **{}**".format(*map(f2, prf(n, node)[:3])) for n in runs) + " |")

    print("\n## 31 activity categories pooled (all 300 articles, unweighted)\n")
    print("| model | correct | false positives | missed | precision | recall | F1 |\n|---|---|---|---|---|---|---|")
    for name in runs:
        tp = sum(gold(a, n) and (a, n) in pos[name] for a in ids for n in leaves)
        fp = sum((not gold(a, n)) and (a, n) in pos[name] for a in ids for n in leaves)
        fn = sum(gold(a, n) and (a, n) not in pos[name] for a in ids for n in leaves)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn)
        print(f"| {name} | {tp} | {fp} | {fn} | {P:.2f} | {R:.2f} | {2 * P * R / (P + R) if P + R else 0:.2f} |")


if __name__ == "__main__":
    main()
