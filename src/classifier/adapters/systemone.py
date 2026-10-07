"""Self-hosted open-weight decision servers that speak TypeSafe's System One API
(Strands Decider's `strands-decider serve`, CLM's `clm-serve`).

POST {endpoint}/v1/systemone with the same rendered questions as every other model;
the only translation is the boolean type name, which these servers call `noul`.
Response: {"model", "answers": {k: {"type": "noul", "noul": p}}, "usage": {...}}.
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import FatalProviderError, ProviderFailure
from .http_decision import Decoded, HttpDecisionAdapter


class SystemOneAdapter(HttpDecisionAdapter):
    def path(self) -> str:
        return "/v1/systemone"

    def encode(self, state: str, questions: dict[str, dict]) -> dict[str, Any]:
        wire = {k: dict(q, type="noul") if q.get("type") == "boolean" else dict(q) for k, q in questions.items()}
        body: dict[str, Any] = {"state": state, "questions": wire}
        if self.cfg.request_model:
            body["model"] = self.cfg.request_model
        return body

    def decode(self, resp: httpx.Response, data: dict[str, Any], attempt: int) -> Decoded:
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ProviderFailure("MODEL_ERROR", "answers_not_object", str(data)[:200], attempt)
        usage = data.get("usage") or {}
        return Decoded(
            model=data.get("model"),
            probabilities={k: (v or {}).get("noul") for k, v in answers.items()},
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_usd=0.0,  # self-hosted: no provider bill; compute time is in the report
            generation_id=None,
            provider="local_server",
        )

    def check_identity(self, d: Decoded) -> None:
        expected = self.cfg.expected_model_returned
        if d.model != expected:
            raise FatalProviderError(f"{self.model_key}: expected model {expected!r}, got {d.model!r}")
