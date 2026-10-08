"""Developer tooling: compare candidate decision templates on the synthetic dev set.

Not part of the classifier pipeline; writes nothing to classification_results/.
Uses report.devset, the same evaluation the end-of-run report runs.

Selection rule (pre-declared, spec §0.8): the template with the highest MINIMUM
balanced accuracy at 0.5 across the decision models wins; ties go to fewer tokens.
Models whose credentials are missing are skipped and listed.

    .venv/bin/python scripts/template_check.py --templates boolean_v1 boolean_v2a boolean_v2c
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from classifier.adapters import make_adapter
from classifier.adapters.base import ClassificationContext
from classifier.config import load_config
from classifier.ontology import load_ontology
from classifier.provenance import sha256_file
from classifier.templates import load_template
from report import devset

ROOT = Path(__file__).resolve().parents[1]
# Dev cases with two or more positive children under the same parent (multi-label nodes).
MULTI = {"rel-01", "lend-02"}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/classification.yaml"))
    ap.add_argument("--ontology", default=str(ROOT / "gsib_basel_ontology_v1.1.json"))
    ap.add_argument("--cases", default=None, help="default: the dev cases matching the ontology version")
    ap.add_argument("--templates", nargs="+", default=["boolean_v1", "boolean_v2a", "boolean_v2b", "boolean_v2c"])
    ap.add_argument("--models", nargs="+", default=["jev", "laya", "d1", "clef"])
    ap.add_argument("--out", default=None, help="optional JSON dump of all scores")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5],
                    help="also report balanced accuracy at these cutoffs (choice scores are shares, not independent)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_config(args.config)
    onto = load_ontology(args.ontology)
    cases = devset.load_cases(args.cases) if args.cases else devset.cases_for(onto)

    models, skipped = [], []
    for m in args.models:
        envs = [e for e in (cfg.models[m].api_key_env, cfg.models[m].account_id_env) if e]
        (models if all(os.environ.get(e) for e in envs) else skipped).append(m)
    if skipped:
        print(f"skipping (credentials missing): {skipped}")

    results: dict[str, dict[str, dict]] = {}
    for tname in args.templates:
        path = ROOT / f"templates/decision/{tname}.txt"
        tmpl = load_template(path, sha256_file(path), "choice" if tname.startswith("choice") else "boolean")
        tmpl.check_ontology(onto)
        results[tname] = {}
        for m in models:
            mcfg = cfg.models[m]
            adapter = make_adapter(m, mcfg)
            try:
                results[tname][m] = await devset.evaluate(adapter, ClassificationContext(mcfg.model, tmpl), onto, cases)
            finally:
                await adapter.aclose()

    print(f"\n{len(cases)} cases\n")
    print(f"{'template':14s} " + " ".join(f"{m:>8s}" for m in models) + "      min   tokens/request")
    ranked = []
    for tname, per in results.items():
        accs = [per[m]["overall"]["balanced_accuracy"] for m in models]
        tok = sum(per[m]["input_tokens"] / max(1, per[m]["requests"]) for m in models) / max(1, len(models))
        ranked.append((min(accs), -tok, tname))
        errs = {m: per[m]["errors"] for m in models if per[m]["errors"]}
        print(f"{tname:14s} " + " ".join(f"{a:8.3f}" for a in accs) + f"   {min(accs):6.3f}   {tok:8.0f}"
              + (f"   errors: {errs}" if errs else ""))
    best = max(ranked)[2]
    spend = sum(per[m]["cost_usd"] for per in results.values() for m in models)
    print(f"\nselected by rule (max of min balanced accuracy, tie -> fewer tokens): {best}")
    print(f"total spend: ${spend:.4f}")
    for m in models:
        print(f"\n{m}: worst misses with {best}")
        for x in results[best][m]["worst_misses"]:
            print(f"   {x['case']:7s} {x['node']:40s} y={x['y']} p={x['p']:.2f}")
    for t in args.thresholds:
        if t == 0.5:
            continue
        print(f"\nbalanced accuracy at cutoff {t}:")
        for tname, per in results.items():
            accs = [devset.rates([(x["y"], x["p"]) for x in per[m]["detail"]], t)["balanced_accuracy"] for m in models]
            print(f"{tname:14s} " + " ".join(f"{a:8.3f}" for a in accs) + f"   min {min(accs):.3f}")
    print("\nrecall on multi-label cases (2+ true children under one node) at 0.5:")
    for tname, per in results.items():
        accs = []
        for m in models:
            pos = [x for x in per[m]["detail"] if x["case"] in MULTI and x["y"] == 1]
            accs.append(sum(x["p"] >= 0.5 for x in pos) / len(pos) if pos else float("nan"))
        print(f"{tname:14s} " + " ".join(f"{a:8.2f}" for a in accs))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    asyncio.run(main())
