import json

import httpx
import pytest

from classifier.adapters.base import ArticleInput, ClassificationContext, FatalProviderError, ProviderFailure
from classifier.adapters.clef import ClefAdapter
from classifier.adapters.gateway import GatewayAdapter
from classifier.ontology import load_ontology

ART = ArticleInput("a", "Bank X led a bond sale.", "h", 23, 23, False, "head")


def envelope(req, model="clef", field="noul", p=0.7, usage=None):
    q = json.loads(req.content)["questions"]
    return {
        "result": {"model": model, "answers": {k: {"type": "noul", field: p} for k in q},
                   "usage": usage if usage is not None else {"input_tokens": 1000, "output_tokens": 0}},
        "success": True, "errors": [], "messages": [],
    }


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("VERCEL_API_KEY", "v-key")


def make(cfg, handler, **upd):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    mcfg = cfg.models["clef"].model_copy(update=upd)
    return ClefAdapter("clef", mcfg, transport=httpx.MockTransport(handler), sleep=fake_sleep), sleeps


def root(cfg):
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    return onto.node("relevance"), onto.children("relevance")


async def test_request_url_headers_body(cfg, template, env):
    seen = {}

    def handler(req):
        seen.update(path=req.url.path, auth=req.headers["authorization"], body=json.loads(req.content))
        return httpx.Response(200, json=envelope(req))

    a, _ = make(cfg, handler)
    node, kids = root(cfg)
    d = await a.classify(ART, node, kids, ClassificationContext("@cf/cloudflare/clef", template))
    assert seen["path"] == "/client/v4/accounts/acct123/ai/run/@cf/cloudflare/clef"
    assert seen["auth"] == "Bearer cf-token"
    assert set(seen["body"]) == {"state", "questions"}
    assert all(q["type"] == "noul" for q in seen["body"]["questions"].values())
    assert d.scores == {k.id: 0.7 for k in kids}
    assert d.input_tokens == 1000 and d.cost_usd == pytest.approx(1000 * 0.24 / 1e6)
    assert d.routing_provider == "cloudflare-workers-ai"


async def test_same_questions_as_gateway_except_type_name(cfg, template, env):
    bodies = {}

    def gw(req):
        bodies["gw"] = json.loads(req.content)
        b = bodies["gw"]
        return httpx.Response(200, json={"model": cfg.models["jev"].model,
                                         "answers": {k: {"probability": 0.5} for k in b["questions"]},
                                         "providerMetadata": {"gateway": {"routing": {"finalProvider": "typesafe-ai"}}}})

    def cf(req):
        bodies["cf"] = json.loads(req.content)
        return httpx.Response(200, json=envelope(req))

    node, kids = root(cfg)
    g = GatewayAdapter("jev", cfg.models["jev"], transport=httpx.MockTransport(gw))
    await g.classify(ART, node, kids, ClassificationContext(cfg.models["jev"].model, template))
    c, _ = make(cfg, cf)
    await c.classify(ART, node, kids, ClassificationContext("@cf/cloudflare/clef", template))
    as_boolean = {k: dict(q, type="boolean") for k, q in bodies["cf"]["questions"].items()}
    assert as_boolean == bodies["gw"]["questions"]
    assert bodies["cf"]["state"] == bodies["gw"]["state"]


async def test_real_response_shape(cfg, template, env):
    """Verbatim shape of a live Workers AI response (2026-10-07)."""
    def handler(req):
        q = json.loads(req.content)["questions"]
        return httpx.Response(200, json={
            "result": {"model": "clef", "answers": {k: {"type": "noul", "noul": 0.9808} for k in q},
                       "usage": {"input_tokens": 250, "output_tokens": 0}},
            "success": True, "errors": [], "messages": []})

    a, _ = make(cfg, handler)
    node, kids = root(cfg)
    d = await a.classify(ART, node, kids, ClassificationContext("x", template))
    assert set(d.scores.values()) == {0.9808} and d.model_returned == "clef"
    assert d.input_tokens == 250 and d.output_tokens == 0


async def test_unknown_probability_field_is_model_error(cfg, template, env):
    a, _ = make(cfg, lambda r: httpx.Response(200, json=envelope(r, field="probability", p=0.4)))
    node, kids = root(cfg)
    with pytest.raises(ProviderFailure) as e:
        await a.classify(ART, node, kids, ClassificationContext("x", template))
    assert e.value.status == "MODEL_ERROR"


async def test_success_false_is_provider_error(cfg, template, env):
    body = {"result": None, "success": False, "errors": [{"code": 5000, "message": "boom"}], "messages": []}
    a, sleeps = make(cfg, lambda r: httpx.Response(200, json=body), retries=0)
    node, kids = root(cfg)
    with pytest.raises(ProviderFailure) as e:
        await a.classify(ART, node, kids, ClassificationContext("x", template))
    assert e.value.status == "PROVIDER_ERROR"


async def test_retry_on_429(cfg, template, env):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, text="slow") if len(calls) == 1 else httpx.Response(200, json=envelope(req))

    a, sleeps = make(cfg, handler)
    node, kids = root(cfg)
    d = await a.classify(ART, node, kids, ClassificationContext("x", template))
    assert d.attempts == 2 and len(sleeps) == 1


async def test_expected_model_mismatch_is_fatal(cfg, template, env):
    a, _ = make(cfg, lambda r: httpx.Response(200, json=envelope(r, model="clef-flash")))
    node, kids = root(cfg)
    with pytest.raises(FatalProviderError):
        await a.classify(ART, node, kids, ClassificationContext("x", template))


def test_missing_account_id(cfg, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(FatalProviderError, match="CLOUDFLARE_ACCOUNT_ID"):
        ClefAdapter("clef", cfg.models["clef"])
