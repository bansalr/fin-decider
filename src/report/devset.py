"""Synthetic dev-set evaluation, shared by end-of-run reports and scripts/template_check.py.

Each request mirrors production: for a (case, parent node) pair, one call asks a
question for every child of that parent; the labelled children are scored.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from classifier.adapters.base import ArticleInput, ClassificationAdapter, ClassificationContext, ProviderFailure
from classifier.ontology import Ontology
from classifier.provenance import sha256_text

DEV = Path(__file__).resolve().parents[2] / "dev"
# Dev cases are labelled with node IDs, so each ontology version has its own file.
# v2 is v1 remapped through v1.1's merged_from (a merged node is positive if any source was).
CASES_BY_ONTOLOGY = {"1.0.0": DEV / "template_cases_v1.jsonl", "1.1.0": DEV / "template_cases_v2.jsonl"}
DEFAULT_CASES = CASES_BY_ONTOLOGY["1.1.0"]


def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def cases_for(onto: Ontology) -> list[dict]:
    if onto.version not in CASES_BY_ONTOLOGY:
        raise ValueError(f"no dev cases for ontology {onto.version}")
    return load_cases(CASES_BY_ONTOLOGY[onto.version])


def level_of(onto: Ontology, child_id: str) -> str:
    node = onto.node(child_id)
    parent = onto.node(node.parent) if node.parent else None
    if parent is not None and parent.parent is None:
        return "relevance"
    if parent is not None and parent.id == "relevance.gsib_activity":
        return "business arm"
    return "leaf" if node.is_leaf else "family"


def rates(pairs: list[tuple[int, float]], t: float = 0.5) -> dict[str, float | int | None]:
    pos = [p >= t for y, p in pairs if y == 1]
    neg = [p < t for y, p in pairs if y == 0]
    rec = sum(pos) / len(pos) if pos else None
    spec = sum(neg) / len(neg) if neg else None
    parts = [x for x in (rec, spec) if x is not None]
    return {"n": len(pairs), "positives": len(pos), "balanced_accuracy": sum(parts) / len(parts) if parts else None,
            "recall": rec, "specificity": spec}


async def evaluate(adapter: ClassificationAdapter, ctx: ClassificationContext, onto: Ontology,
                   cases: list[dict] | None = None, concurrency: int = 8) -> dict[str, Any]:
    cases = cases if cases is not None else cases_for(onto)
    jobs: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for c in cases:
        for child, y in c["expect"].items():
            jobs[(c["id"], onto.node(child).parent)][child] = y
    text = {c["id"]: c["text"] for c in cases}
    sem = asyncio.Semaphore(concurrency)

    async def one(key):
        cid, parent = key
        art = ArticleInput(cid, text[cid], sha256_text(text[cid]), len(text[cid]), len(text[cid]), False, "head")
        async with sem:
            try:
                return key, await adapter.classify(art, onto.node(parent), onto.children(parent), ctx), None
            except ProviderFailure as e:
                return key, None, f"{e.status}/{e.error_code}"

    out = await asyncio.gather(*[one(k) for k in jobs])
    detail, errors, cost, tokens = [], [], 0.0, 0
    for (cid, _parent), d, err in out:
        if d is None:
            errors.append(err)
            continue
        cost += d.cost_usd or 0.0
        tokens += d.input_tokens or 0
        for child, y in jobs[(cid, _parent)].items():
            detail.append({"case": cid, "node": child, "level": level_of(onto, child), "y": y, "p": d.scores[child]})
    pairs = [(x["y"], x["p"]) for x in detail]
    by_level = {lvl: rates([(x["y"], x["p"]) for x in detail if x["level"] == lvl])
                for lvl in ("relevance", "business arm", "family", "leaf")}
    misses = sorted((x for x in detail if (x["p"] >= 0.5) != bool(x["y"])), key=lambda x: -abs(x["p"] - 0.5))
    return {"cases": len(cases), "requests": len(jobs), "errors": errors, "overall": rates(pairs),
            "by_level": by_level, "worst_misses": misses[:5], "cost_usd": cost, "input_tokens": tokens,
            "detail": detail}
