"""Clef (Cloudflare) on Workers AI.

POST {endpoint}/accounts/{ACCOUNT_ID}/ai/run/{model}. The request carries the
same rendered questions as every other decision model; the only translation is
the boolean type name, which Workers AI calls `noul` (TypeSafe naming).

Workers AI wraps responses in {result, success, errors, messages} and reports no
dollar cost, so cost is estimated from input tokens and the configured price.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import FatalProviderError, ProviderFailure
from .http_decision import Decoded, HttpDecisionAdapter

# Field names are tolerated in this order until the live probe pins them.
_PROB_FIELDS = ("noul", "probability")
_IN_TOKEN_FIELDS = ("input_tokens", "inputTokens", "prompt_tokens")
_OUT_TOKEN_FIELDS = ("output_tokens", "outputTokens", "completion_tokens")


def _first(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k]
    return None


class ClefAdapter(HttpDecisionAdapter):
    def __init__(self, model_key, cfg, transport=None, **kw):
        account = os.environ.get(cfg.account_id_env or "")
        if not account:
            raise FatalProviderError(f"environment variable {cfg.account_id_env} is not set")
        self.account_id = account
        super().__init__(model_key, cfg, transport=transport, **kw)

    def path(self) -> str:
        return f"/accounts/{self.account_id}/ai/run/{self.cfg.model}"

    def encode(self, state: str, questions: dict[str, dict]) -> dict[str, Any]:
        wire = {}
        for k, q in questions.items():
            q = dict(q)
            if q.get("type") == "boolean":
                q["type"] = "noul"
            wire[k] = q
        return {"state": state, "questions": wire}

    def decode(self, resp: httpx.Response, data: dict[str, Any], attempt: int) -> Decoded:
        if data.get("success") is False or data.get("errors"):
            raise ProviderFailure("PROVIDER_ERROR", "workers_ai_error", str(data.get("errors"))[:300], attempt)
        result = data.get("result")
        if not isinstance(result, dict):
            raise ProviderFailure("MODEL_ERROR", "no_result", str(data)[:200], attempt)
        answers = result.get("answers") or {}
        if not isinstance(answers, dict):
            raise ProviderFailure("MODEL_ERROR", "answers_not_object", str(answers)[:200], attempt)
        usage = result.get("usage") or {}
        tin = _first(usage, _IN_TOKEN_FIELDS)
        price = self.cfg.price_per_m_input_tokens
        return Decoded(
            model=result.get("model"),
            probabilities={k: _first(v or {}, _PROB_FIELDS) for k, v in answers.items()},
            input_tokens=tin,
            output_tokens=_first(usage, _OUT_TOKEN_FIELDS),
            cost_usd=(tin * price / 1e6) if (tin is not None and price is not None) else None,
            generation_id=resp.headers.get("cf-ray"),
            provider="cloudflare-workers-ai",
        )

    def check_identity(self, d: Decoded) -> None:
        expected = self.cfg.expected_model_returned
        if expected is not None and d.model != expected:
            raise FatalProviderError(f"{self.model_key}: expected model {expected!r}, got {d.model!r}")
