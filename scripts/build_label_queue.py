"""Build the blinded human-labelling queue (spec §2: downstream of classification).

    .venv/bin/python scripts/build_label_queue.py --run jev=classification_results/jev-... \
        --run d1=classification_results/d1-... --run clef=classification_results/clef-...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from adjudication import queue
from classifier.config import load_config
from classifier.ontology import load_ontology

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="MODEL=RUN_DIR")
    ap.add_argument("--split", type=int, default=150)
    ap.add_argument("--all-yes", type=int, default=50)
    ap.add_argument("--all-no", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--name", default="queue_v1")
    ap.add_argument("--out", default=str(ROOT / "labeling"))
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs/classification.yaml")
    onto = load_ontology(ROOT / cfg.ontology.path, cfg.ontology.sha256)
    runs = {m: Path(d) for m, d in (r.split("=", 1) for r in args.run)}
    corpus = pl.read_parquet(ROOT / cfg.corpus.directory / "canonical.parquet")
    sizes = {"split": args.split, "all_yes": args.all_yes, "all_no": args.all_no}
    picked, meta = queue.build(runs, corpus, sizes, args.seed)
    meta["ontology_version"] = onto.version
    paths = queue.write(picked, meta, Path(args.out), args.name)
    (Path(args.out) / "GUIDE.md").write_text(queue.guide(onto), encoding="utf-8")
    print(f"{len(picked)} articles from {meta['common_articles']:,} commonly scored; population per stratum "
          f"{meta['strata_population']}")
    print(f"queue: {paths['queue']}\nkey (keep from labellers): {paths['key']}\nguide: {Path(args.out) / 'GUIDE.md'}")


if __name__ == "__main__":
    main()
