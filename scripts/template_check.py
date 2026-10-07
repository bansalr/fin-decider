"""Developer tooling: compare candidate decision templates on the synthetic dev set.

Not part of the classifier pipeline; writes nothing to classification_results/.
Each request mirrors production exactly: for a (case, parent node) pair, one request
asks a boolean question for every child of that parent, and the expected children
are read from the answers.

Selection rule (pre-declared, spec §0): the template with the highest MINIMUM
balanced accuracy at 0.5 across the decision models wins; ties go to the shorter
template.

    .venv/bin/python scripts/template_check.py --ontology gsib_basel_ontology_v1.1.json \
        --templates boolean_v1 boolean_v2a boolean_v2b --models jev laya d1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from classifier.adapters.base import ArticleInput, ClassificationContext, ProviderFailure
from classifier.adapters.gateway import GatewayAdapter
from classifier.config import load_config
from classifier.ontology import load_ontology
from classifier.provenance import sha256_file, sha256_text
from classifier.templates import load_template

ROOT = Path(__file__).resolve().parents[1]


def balanced_accuracy(pairs: list[tuple[int, float]], t: float = 0.5) -> float:
    pos = [p >= t for y, p in pairs if y == 1]
    neg = [p < t for y, p in pairs if y == 0]
    parts = [sum(x) / len(x) for x in (pos, neg) if x]
    return sum(parts) / len(parts) if parts else float("nan")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/classification.yaml"))
    ap.add_argument("--ontology", default=str(ROOT / "gsib_basel_ontology_v1.1.json"))
    ap.add_argument("--cases", default=str(ROOT / "dev/template_cases_v1.jsonl"))
    ap.add_argument("--templates", nargs="+", default=["boolean_v1", "boolean_v2a", "boolean_v2b"])
    ap.add_argument("--models", nargs="+", default=["jev", "laya", "d1"])
    ap.add_argument("--out", default=None, help="optional JSON dump of all scores")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_config(args.config)
    onto = load_ontology(args.ontology)
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]

    # (case, parent) -> expected {child: label}
    jobs: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for c in cases:
        for child, y in c["expect"].items():
            jobs[(c["id"], onto.node(child).parent)][child] = y
    texts = {c["id"]: c["text"] for c in cases}
    n_pos = sum(y for c in cases for y in c["expect"].values())
    n_all = sum(len(c["expect"]) for c in cases)
    print(f"{len(cases)} cases, {n_all} labelled pairs ({n_pos} positive), {len(jobs)} requests per model×template\n")

    results: dict[str, dict[str, dict]] = {}
    for tname in args.templates:
        tcfg = cfg.templates.get(tname)
        path = tcfg.path if tcfg else f"templates/decision/{tname}.txt"
        tmpl = load_template(ROOT / path, sha256_file(ROOT / path), "boolean")
        tmpl.check_ontology(onto)
        results[tname] = {}
        for m in args.models:
            mcfg = cfg.models[m]
            adapter = GatewayAdapter(m, mcfg)
            ctx = ClassificationContext(mcfg.model, tmpl)
            sem = asyncio.Semaphore(8)

            async def one(key, expected):
                cid, parent = key
                text = texts[cid]
                art = ArticleInput(cid, text, sha256_text(text), len(text), len(text), False, "head")
                async with sem:
                    try:
                        d = await adapter.classify(art, onto.node(parent), onto.children(parent), ctx)
                    except ProviderFailure as e:
                        return key, expected, None, str(e)
                return key, expected, d, None

            out = await asyncio.gather(*[one(k, v) for k, v in jobs.items()])
            await adapter.aclose()
            pairs, errors, tokens, cost, detail = [], 0, 0, 0.0, []
            for (cid, parent), expected, d, err in out:
                if d is None:
                    errors += 1
                    continue
                tokens += d.input_tokens or 0
                cost += d.cost_usd or 0.0
                for child, y in expected.items():
                    pairs.append((y, d.scores[child]))
                    detail.append({"case": cid, "node": child, "y": y, "p": d.scores[child]})
            results[tname][m] = {
                "bal_acc": balanced_accuracy(pairs),
                "errors": errors,
                "tokens_per_request": tokens / max(1, len(out) - errors),
                "cost": cost,
                "detail": detail,
            }

    models = args.models
    print(f"{'template':14s} " + " ".join(f"{m:>8s}" for m in models) + "      min   tok/req(avg)")
    ranked = []
    for tname, per in results.items():
        accs = [per[m]["bal_acc"] for m in models]
        tok = sum(per[m]["tokens_per_request"] for m in models) / len(models)
        ranked.append((min(accs), -tok, tname))
        print(f"{tname:14s} " + " ".join(f"{a:8.3f}" for a in accs) + f"   {min(accs):6.3f}   {tok:8.0f}"
              + ("   errors: " + str({m: per[m]['errors'] for m in models}) if any(per[m]['errors'] for m in models) else ""))
    best = max(ranked)
    total_cost = sum(per[m]["cost"] for per in results.values() for m in models)
    print(f"\nselected by rule (max of min balanced accuracy, tie -> fewer tokens): {best[2]}")
    print(f"total spend: ${total_cost:.4f}")

    # Worst misses for the selected template, per model.
    for m in models:
        misses = [d for d in results[best[2]][m]["detail"] if (d["p"] >= 0.5) != bool(d["y"])]
        misses.sort(key=lambda d: -abs(d["p"] - 0.5))
        print(f"\n{m}: {len(misses)} misses with {best[2]}")
        for d in misses[:6]:
            print(f"   {d['case']:7s} {d['node']:40s} y={d['y']} p={d['p']:.2f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
