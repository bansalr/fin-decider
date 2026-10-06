"""Adapter contract (spec §12). Adapters normalize provider output into NodeDecision
and contain no hierarchy logic: they are told which node and children to score."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..ontology import OntologyNode

STATUSES = (
    "SUCCESS",
    "TIMEOUT",
    "RATE_LIMITED",
    "PROVIDER_ERROR",
    "INVALID_REQUEST",
    "MODEL_ERROR",
    "SKIPPED_BY_ROUTING",
)


@dataclass(frozen=True)
class ArticleInput:
    article_id: str
    text: str
    input_hash: str
    original_chars: int
    submitted_chars: int
    truncated: bool
    truncation_strategy: str


@dataclass(frozen=True)
class ClassificationContext:
    model_id: str
    template: Any  # templates.BooleanTemplate; adapters render questions from it

    @property
    def template_hash(self) -> str:
        return self.template.sha256


@dataclass
class NodeDecision:
    node_id: str
    scores: dict[str, float]
    model_returned: str | None = None
    routing_provider: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    attempts: int = 1
    generation_id: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)


class ProviderFailure(Exception):
    """A non-classification outcome. Never interpreted as a negative."""

    def __init__(self, status: str, error_code: str, message: str = "", attempts: int = 1):
        assert status in STATUSES and status not in ("SUCCESS", "SKIPPED_BY_ROUTING")
        super().__init__(f"{status}/{error_code}: {message}")
        self.status = status
        self.error_code = error_code
        self.attempts = attempts


class FatalProviderError(Exception):
    """Aborts the whole run: auth failure or a model/provider identity mismatch."""


class ClassificationAdapter(abc.ABC):
    model_key: str  # config key, e.g. "jev"

    @abc.abstractmethod
    async def classify(
        self, article: ArticleInput, node: OntologyNode, children: list[OntologyNode], ctx: ClassificationContext
    ) -> NodeDecision: ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
