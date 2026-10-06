"""Vercel AI Gateway decision adapter, shared by Jev and Laya.

One POST /v1/evaluate per decision node, one boolean question per child, so
child scores are independent (multi-label). Provider responses are normalized
into NodeDecision here and nowhere else.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
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

FALLBACK_HEADER = "x-ai-gateway-decision-fallback-triggered"


def question_keys(children: list[OntologyNode]) -> dict[str, str]:
    """Neutral, model-agnostic question keys: q1..qn in ontology order."""
    return {f"q{i + 1}": c.id for i, c in enumerate(children)}


class GatewayAdapter(ClassificationAdapter):
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

    async def aclose(self) -> None:
        await self._client.aclose()

    def build_request(self, article: ArticleInput, node: OntologyNode, children: list[OntologyNode],
                      ctx: ClassificationContext) -> tuple[dict[str, Any], dict[str, str]]:
        keys = question_keys(children)
        by_id = {c.id: c for c in children}
        questions = {k: ctx.template.render(node, by_id[cid]) for k, cid in keys.items()}
        body = {
            "model": self.cfg.model,
            "state": article.text,
            "questions": questions,
            "providerOptions": {"gateway": {"only": list(self.cfg.provider_lock)}},
        }
        return body, keys

    async def classify(self, article: ArticleInput, node: OntologyNode, children: list[OntologyNode],
                       ctx: ClassificationContext) -> NodeDecision:
        body, keys = self.build_request(article, node, children, ctx)
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                resp = await self._client.post("/v1/evaluate", json=body)
            except httpx.TimeoutException as e:
                failure = ProviderFailure("TIMEOUT", "timeout", str(e), attempt)
                retry_after = None
            except httpx.TransportError as e:
                failure = ProviderFailure("PROVIDER_ERROR", type(e).__name__, str(e), attempt)
                retry_after = None
            else:
                latency_ms = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    return self._parse(resp, node, keys, latency_ms, attempt)
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
            # Validation failures are not retryable; fail fast on this decision.
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
        if resp.headers.get(FALLBACK_HEADER) == "true":
            raise FatalProviderError(f"{self.model_key}: gateway decision fallback triggered; none is configured")
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderFailure("MODEL_ERROR", "invalid_json", str(e), attempt) from e

        returned = data.get("model")
        routing = ((data.get("providerMetadata") or {}).get("gateway") or {}).get("routing") or {}
        final_provider = routing.get("finalProvider")
        if not self.cfg.allow_returned_model_mismatch:
            if returned != self.cfg.model:
                raise FatalProviderError(f"{self.model_key}: requested {self.cfg.model}, got {returned}")
            if final_provider not in self.cfg.provider_lock:
                raise FatalProviderError(
                    f"{self.model_key}: served by {final_provider}, outside provider_lock {self.cfg.provider_lock}"
                )

        answers = data.get("answers") or {}
        if set(answers) != set(keys):
            raise ProviderFailure("MODEL_ERROR", "answer_keys", f"expected {sorted(keys)}, got {sorted(answers)}", attempt)
        scores: dict[str, float] = {}
        for k, child_id in keys.items():
            ans = answers[k] or {}
            p = ans.get("probability")
            if not isinstance(p, (int, float)) or not 0.0 <= float(p) <= 1.0:
                raise ProviderFailure("MODEL_ERROR", "bad_probability", f"{k}: {ans}", attempt)
            scores[child_id] = float(p)

        usage = data.get("usage") or {}
        gw = (data.get("providerMetadata") or {}).get("gateway") or {}
        cost = gw.get("cost")
        return NodeDecision(
            node_id=node.id,
            scores=scores,
            model_returned=returned,
            routing_provider=final_provider,
            latency_ms=latency_ms,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            cost_usd=float(cost) if cost is not None else None,
            attempts=attempt,
            generation_id=gw.get("generationId"),
            raw=data,
        )
