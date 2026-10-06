"""Generic hierarchical traversal (spec §11, §13, §14).

The ontology graph drives everything; adapters only score the children of the
node they are handed. Traversal and positive thresholds are independent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .adapters.base import (
    ArticleInput,
    ClassificationAdapter,
    ClassificationContext,
    NodeDecision,
    ProviderFailure,
)
from .config import Hierarchy
from .ontology import Ontology


@dataclass(frozen=True)
class NodeVisit:
    node_id: str
    depth: int
    path_score: float


@dataclass
class ArticleOutcome:
    rows: list[dict[str, Any]]
    calls_made: int
    complete: bool  # every visited decision succeeded


async def classify_article(
    article: ArticleInput,
    onto: Ontology,
    adapter: ClassificationAdapter,
    ctx: ClassificationContext,
    hierarchy: Hierarchy,
    key_fn: Callable[[str], str],
    cache: dict[str, NodeDecision] | None = None,
) -> ArticleOutcome:
    """Breadth-first traversal of one article. `key_fn(node_id)` returns the prediction key."""
    cache = cache or {}
    trav = hierarchy.traversal
    pos_t = hierarchy.classification.positive_threshold
    queue = deque(NodeVisit(r, 0, 1.0) for r in onto.roots)
    visited = 0
    calls = 0
    complete = True
    rows: list[dict[str, Any]] = []

    while queue:
        visit = queue.popleft()
        node = onto.node(visit.node_id)
        if node.is_leaf:
            continue  # leaves are recorded as child rows of their parent; no call
        children = onto.children(node.id)
        key = key_fn(node.id)
        visited += 1

        decision = cache.get(key)
        from_cache = decision is not None
        if decision is None:
            calls += 1
            try:
                decision = await adapter.classify(article, node, children, ctx)
            except ProviderFailure as f:
                complete = False
                for c in children:
                    rows.append(_row(visit, node.id, c.id, key, None, False, False, trav.threshold, pos_t, None,
                                     status=f.status, error_code=f.error_code, attempts=f.attempts))
                continue  # never treat a failure as a negative; do not descend

        for c in children:
            score = decision.scores[c.id]
            positive = score >= pos_t
            traverse = (
                score >= trav.threshold
                and not c.is_leaf
                and visit.depth + 1 < trav.max_depth
                and visited + len(queue) < trav.max_nodes_per_article
            )
            path = visit.path_score * score
            rows.append(_row(visit, node.id, c.id, key, score, positive, traverse, trav.threshold, pos_t, path,
                             decision=decision, from_cache=from_cache))
            if traverse:
                queue.append(NodeVisit(c.id, visit.depth + 1, path))

    return ArticleOutcome(rows=rows, calls_made=calls, complete=complete)


def _row(visit: NodeVisit, node_id: str, child_id: str, key: str, score: float | None, positive: bool,
         traverse: bool, trav_t: float, pos_t: float, path: float | None, *, decision: NodeDecision | None = None,
         from_cache: bool = False, status: str = "SUCCESS", error_code: str | None = None,
         attempts: int | None = None) -> dict[str, Any]:
    d = decision
    return {
        "node_id": node_id,
        "node_depth": visit.depth,
        "child_id": child_id,
        "score": score,
        "selected_positive": positive,
        "selected_for_traversal": traverse,
        "local_threshold": pos_t,
        "traversal_threshold": trav_t,
        "path_score": path,
        "prediction_key": key,
        "model_returned": d.model_returned if d else None,
        "routing_provider": d.routing_provider if d else None,
        "latency_ms": d.latency_ms if d else None,
        "input_tokens": d.input_tokens if d else None,
        "output_tokens": d.output_tokens if d else None,
        "provider_cost_usd": d.cost_usd if d else None,
        "generation_id": d.generation_id if d else None,
        "attempts": d.attempts if d else attempts,
        "reused_from_cache": from_cache,
        "status": status,
        "error_code": error_code,
    }
