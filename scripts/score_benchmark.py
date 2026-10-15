"""Score a run on a benchmark corpus against its pre-registered expectations and groups.

    .venv/bin/python scripts/score_benchmark.py -c configs/bench_fomc.yaml --run classification_results/jev-...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adjudication import benchmark
from classifier.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("-m", "--model", default=None, help="default: the run's only model")
    ap.add_argument("--json", default=None, help="also write the scores as JSON")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run = Path(a.run)
    model = a.model or next(iter(json.loads((run / "manifest.json").read_text())["models"]))
    s = benchmark.score(run, model, Path(cfg.corpus.directory))
    print(f"{cfg.benchmark.name} · {model} · {run.name}\n")
    print(benchmark.render(s))
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
