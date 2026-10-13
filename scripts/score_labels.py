"""Score model runs against human labels.

    .venv/bin/python scripts/score_labels.py --labels labeling/queue_v1_labelled.csv \
        [--labels2 labeling/queue_v1_labelled_b.csv] --run jev=classification_results/jev-...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adjudication import queue, score
from classifier.config import load_config
from classifier.ontology import load_ontology

ROOT = Path(__file__).resolve().parents[1]


def fmt(x):
    return "  n/a" if x is None else f"{x:5.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--labels2", default=None, help="second labeller's file, for agreement")
    ap.add_argument("--key", default=str(ROOT / "labeling/queue_v1_key.csv"))
    ap.add_argument("--run", action="append", required=True, metavar="MODEL=RUN_DIR")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs/classification.yaml")
    onto = load_ontology(ROOT / cfg.ontology.path, cfg.ontology.sha256)
    leaf_ids = {n for n, nd in onto.nodes.items() if nd.is_leaf and nd.parent != onto.roots[0]}
    key = score.read_key(Path(args.key))
    labels = score.read_labels(Path(args.labels))
    print(f"{len(labels)} labelled articles")
    if args.labels2:
        ag = score.annotator_agreement(labels, score.read_labels(Path(args.labels2)))
        print(f"\nlabeller agreement on {ag['items']} articles:")
        for n, v in ag.items():
            if isinstance(v, dict):
                print(f"  {n:34s} agreement {fmt(v['agreement'])}  kappa {fmt(v['kappa'])}")
        print(f"  to adjudicate (top-level disagreements): {', '.join(ag['disagreements']) or 'none'}")
    for spec in args.run:
        model, d = spec.split("=", 1)
        s = score.score(labels, key, queue.positives(Path(d), model), leaf_ids)
        print(f"\n{model} ({d}), {s['_scored']} scored, {s['_missing_predictions']} not in run; weighted to population:")
        print(f"  {'':34s} precision recall   f1   support")
        for n, v in s.items():
            if not n.startswith("_"):
                print(f"  {n:34s} {fmt(v['precision'])}   {fmt(v['recall'])}  {fmt(v['f1'])}  {v['support']:8.0f}")


if __name__ == "__main__":
    main()
