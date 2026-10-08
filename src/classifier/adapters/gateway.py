"""Vercel AI Gateway decision adapter (Jev, Laya, d1).

POST {endpoint}/v1/evaluate with `boolean` questions and a provider lock. The
gateway reports the model that ran, the serving provider, usage, and cost.
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import FatalProviderError, ProviderFailure
from .http_decision import Decoded, HttpDecisionAdapter, question_keys  # noqa: F401  (re-export)

FALLBACK_HEADER = "x-ai-gateway-decision-fallback-triggered"


class GatewayAdapter(HttpDecisionAdapter):
    def path(self) -> str:
        return "/v1/evaluate"

    def encode(self, state: str, questions: dict[str, dict]) -> dict[str, Any]:
        return {
            "model": self.cfg.model,
            "state": state,
            "questions": questions,
            "providerOptions": {"gateway": {"only": list(self.cfg.provider_lock)}},
        }

    def decode(self, resp: httpx.Response, data: dict[str, Any], attempt: int) -> Decoded:
        if resp.headers.get(FALLBACK_HEADER) == "true":
            raise FatalProviderError(f"{self.model_key}: gateway decision fallback triggered; none is configured")
        gw = (data.get("providerMetadata") or {}).get("gateway") or {}
        routing = gw.get("routing") or {}
        usage = data.get("usage") or {}
        answers = data.get("answers") or {}
        if not isinstance(answers, dict):
            raise ProviderFailure("MODEL_ERROR", "answers_not_object", str(answers)[:200], attempt)
        cost = gw.get("cost")
        return Decoded(
            model=data.get("model"),
            probabilities={k: (v or {}).get("probability") for k, v in answers.items()},
            answers=answers,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            cost_usd=float(cost) if cost is not None else None,
            generation_id=gw.get("generationId"),
            provider=routing.get("finalProvider"),
        )

    def check_identity(self, d: Decoded) -> None:
        if d.model != self.cfg.model:
            raise FatalProviderError(f"{self.model_key}: requested {self.cfg.model}, got {d.model}")
        if d.provider not in self.cfg.provider_lock:
            raise FatalProviderError(
                f"{self.model_key}: served by {d.provider}, outside provider_lock {self.cfg.provider_lock}"
            )
