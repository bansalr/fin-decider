"""Shared HTTP decision-model adapter.

One request per decision node, one boolean question per child, so child scores
are independent (multi-label). Every HTTP decision provider builds the questions
the same way (shared template, neutral q1..qn keys); subclasses only choose the
URL, the wire encoding, and how to decode the provider's response.

Retry policy: 429, 5xx and timeouts are retried with backoff; validation errors
fail the decision immediately; authentication errors abort the run.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import ModelCfg
from ..ontology import OntologyNode
from .base import (
    ArticleInput,
    ClassificationAdapter,
    ClassificationContext,
    FatalProviderError,
    NodeDecision,
    ProviderFailure,
)


def question_keys(children: list[OntologyNode]) -> dict[str, str]:
    """Neutral, model-agnostic question keys: q1..qn in ontology order."""
    return {f"q{i + 1}": c.id for i, c in enumerate(children)}


def body_questions(body: dict[str, Any]) -> dict[str, dict]:
    """The rendered questions inside an encoded body (all providers keep them under `questions`)."""
    return {k: dict(q, type="boolean") if q.get("type") == "noul" else q for k, q in body["questions"].items()}


@dataclass
class Decoded:
    model: str | None
    probabilities: dict[str, Any]  # question key -> raw probability value
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    generation_id: str | None
    provider: str | None


class HttpDecisionAdapter(ClassificationAdapter):
    def __init__(self, model_key: str, cfg: ModelCfg, transport: httpx.AsyncBaseTransport | None = None,
                 sleep=asyncio.sleep):
        self.model_key = model_key
        self.cfg = cfg
        key = os.environ.get(cfg.api_key_env or "")
        if not key:
            raise FatalProviderError(f"environment variable {cfg.api_key_env} is not set")
        self._client = httpx.AsyncClient(
            base_url=cfg.endpoint or "",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=cfg.timeout_seconds,
            transport=transport,
            limits=httpx.Limits(max_connections=cfg.concurrency * 2, max_keepalive_connections=cfg.concurrency * 2),
        )
        self._sleep = sleep

    # -- provider hooks ---------------------------------------------------------
    def path(self) -> str:
        raise NotImplementedError

    def encode(self, state: str, questions: dict[str, dict]) -> dict[str, Any]:
        raise NotImplementedError

    def decode(self, resp: httpx.Response, data: dict[str, Any], attempt: int) -> Decoded:
        raise NotImplementedError

    def check_identity(self, decoded: Decoded) -> None:
        """Raise FatalProviderError if the response was not produced by the configured model."""

    # -- shared machinery -------------------------------------------------------
    async def aclose(self) -> None:
        await self._client.aclose()

    def build_request(self, article: ArticleInput, node: OntologyNode, children: list[OntologyNode],
                      ctx: ClassificationContext) -> tuple[dict[str, Any], dict[str, str]]:
        keys = question_keys(children)
        by_id = {c.id: c for c in children}
        questions = {k: ctx.template.render(node, by_id[cid]) for k, cid in keys.items()}
        return self.encode(article.text, questions), keys

    async def classify(self, article: ArticleInput, node: OntologyNode, children: list[OntologyNode],
                       ctx: ClassificationContext) -> NodeDecision:
        body, keys = self.build_request(article, node, children, ctx)
        state = article.text
        shrinks = 0
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                resp = await self._client.post(self.path(), json=body)
            except httpx.TimeoutException as e:
                failure, retry_after = ProviderFailure("TIMEOUT", "timeout", str(e), attempt), None
            except httpx.TransportError as e:
                failure, retry_after = ProviderFailure("PROVIDER_ERROR", type(e).__name__, str(e), attempt), None
            else:
                latency_ms = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    d = self._parse(resp, node, keys, latency_ms, attempt)
                    d.state_chars = len(state)
                    return d
                if resp.status_code == 422 and self.cfg.shrink_on_422 and shrinks < 5 and len(state) > 200:
                    # Input over the model's technical limit: resend with 20% less text from the end.
                    shrinks += 1
                    state = state[: int(len(state) * 0.8)]
                    body = self.encode(state, body_questions(body))
                    continue
                failure, retry_after = self._http_failure(resp, attempt)

            if attempt > self.cfg.retries:
                raise failure
            delay = retry_after if retry_after is not None else min(60.0, 2 ** (attempt - 1)) * (0.5 + random.random())
            await self._sleep(delay)

    def _http_failure(self, resp: httpx.Response, attempt: int) -> tuple[ProviderFailure, float | None]:
        code = resp.status_code
        text = resp.text[:300]
        if code in (401, 403):
            raise FatalProviderError(f"authentication failed ({code}): {text}")
        if code in (400, 404, 413, 422):
            raise ProviderFailure("INVALID_REQUEST", f"http_{code}", text, attempt)
        retry_after = None
        if "retry-after" in resp.headers:
            try:
                retry_after = float(resp.headers["retry-after"])
            except ValueError:
                pass
        status = "RATE_LIMITED" if code == 429 else ("TIMEOUT" if code in (408, 504) else "PROVIDER_ERROR")
        return ProviderFailure(status, f"http_{code}", text, attempt), retry_after

    def _parse(self, resp: httpx.Response, node: OntologyNode, keys: dict[str, str], latency_ms: float,
               attempt: int) -> NodeDecision:
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderFailure("MODEL_ERROR", "invalid_json", str(e), attempt) from e
        dec = self.decode(resp, data, attempt)
        if not self.cfg.allow_returned_model_mismatch:
            self.check_identity(dec)
        if set(dec.probabilities) != set(keys):
            raise ProviderFailure("MODEL_ERROR", "answer_keys",
                                  f"expected {sorted(keys)}, got {sorted(dec.probabilities)}", attempt)
        scores: dict[str, float] = {}
        for k, child_id in keys.items():
            p = dec.probabilities[k]
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= float(p) <= 1.0:
                raise ProviderFailure("MODEL_ERROR", "bad_probability", f"{k}: {p!r}", attempt)
            scores[child_id] = float(p)
        return NodeDecision(
            node_id=node.id,
            scores=scores,
            model_returned=dec.model,
            routing_provider=dec.provider,
            latency_ms=latency_ms,
            input_tokens=dec.input_tokens,
            output_tokens=dec.output_tokens,
            cost_usd=dec.cost_usd,
            attempts=attempt,
            generation_id=dec.generation_id,
            raw=data,
        )
