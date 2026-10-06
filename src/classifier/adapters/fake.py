"""Deterministic offline adapter for tests and --dry-run. Makes no network calls."""

from __future__ import annotations

from ..ontology import OntologyNode
from ..provenance import sha256_text
from .base import ArticleInput, ClassificationAdapter, ClassificationContext, NodeDecision


class FakeAdapter(ClassificationAdapter):
    def __init__(self, model_key: str = "fake", overrides: dict[str, float] | None = None, fail_nodes: dict | None = None):
        self.model_key = model_key
        self.overrides = overrides or {}
        self.fail_nodes = fail_nodes or {}  # node_id -> ProviderFailure to raise
        self.calls: list[tuple[str, str]] = []

    async def classify(self, article: ArticleInput, node: OntologyNode, children: list[OntologyNode],
                       ctx: ClassificationContext) -> NodeDecision:
        self.calls.append((article.article_id, node.id))
        if node.id in self.fail_nodes:
            raise self.fail_nodes[node.id]
        scores = {}
        for c in children:
            if c.id in self.overrides:
                scores[c.id] = self.overrides[c.id]
            else:
                scores[c.id] = int(sha256_text(article.input_hash + c.id)[:8], 16) / 0xFFFFFFFF
        return NodeDecision(node_id=node.id, scores=scores, model_returned=ctx.model_id, routing_provider="local",
                            latency_ms=0.0, input_tokens=0, output_tokens=0, cost_usd=0.0)
