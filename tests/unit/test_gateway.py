import json

import httpx
import pytest

from classifier.adapters.base import ArticleInput, ClassificationContext, FatalProviderError, ProviderFailure
from classifier.adapters.gateway import GatewayAdapter

ART = ArticleInput("a", "Bank X led a bond sale.", "h", 23, 23, False, "head")


def ok_body(req_json, model="typesafe-ai/jev", provider="typesafe-ai", p=0.8):
    return {
        "model": model,
        "answers": {k: {"type": "boolean", "probability": p} for k in req_json["questions"]},
        "usage": {"inputTokens": 100, "outputTokens": 0},
        "providerMetadata": {"gateway": {"routing": {"finalProvider": provider}, "cost": "0.0000042",
                                         "generationId": "gen_1"}},
    }


def make(cfg, handler, model="jev", monkeypatch=None):
    monkeypatch.setenv("VERCEL_API_KEY", "test-key")
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    a = GatewayAdapter(model, cfg.models[model], transport=httpx.MockTransport(handler), sleep=fake_sleep)
    return a, sleeps


def onto_root(cfg):
    from classifier.ontology import load_ontology
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    root = onto.node("relevance")
    return root, onto.children("relevance")


async def test_request_shape_and_parse(cfg, template, monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body(seen["body"]))

    a, _ = make(cfg, handler, monkeypatch=monkeypatch)
    root, children = onto_root(cfg)
    d = await a.classify(ART, root, children, ClassificationContext("typesafe-ai/jev", template))
    body = seen["body"]
    assert seen["path"] == "/v1/evaluate" and seen["auth"] == "Bearer test-key"
    assert body["model"] == "typesafe-ai/jev" and body["state"] == ART.text
    assert list(body["questions"]) == ["q1", "q2", "q3"]
    assert all(q["type"] == "boolean" for q in body["questions"].values())
    assert body["providerOptions"] == {"gateway": {"only": ["typesafe-ai"]}}
    assert set(d.scores) == {c.id for c in children} and d.input_tokens == 100
    assert d.cost_usd == pytest.approx(4.2e-6) and d.generation_id == "gen_1"
    await a.aclose()


async def test_jev_and_laya_requests_identical_except_model(cfg, template, monkeypatch):
    bodies = {}
    root, children = onto_root(cfg)
    for m, prov in (("jev", "typesafe-ai"), ("laya", "boundless"), ("d1", "liquid")):
        def handler(request, m=m, prov=prov):
            bodies[m] = json.loads(request.content)
            return httpx.Response(200, json=ok_body(bodies[m], cfg.models[m].model, prov))
        a, _ = make(cfg, handler, m, monkeypatch)
        await a.classify(ART, root, children, ClassificationContext(cfg.models[m].model, template))
        await a.aclose()
    strip = lambda b: {k: v for k, v in b.items() if k not in ("model", "providerOptions")}  # noqa: E731
    assert strip(bodies["jev"]) == strip(bodies["laya"]) == strip(bodies["d1"])


async def test_retry_on_429_then_success(cfg, template, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "2"}, text="slow down")
        return httpx.Response(200, json=ok_body(json.loads(request.content)))

    a, sleeps = make(cfg, handler, monkeypatch=monkeypatch)
    root, children = onto_root(cfg)
    d = await a.classify(ART, root, children, ClassificationContext("typesafe-ai/jev", template))
    assert d.attempts == 3 and sleeps == [2.0, 2.0]


async def test_retries_exhausted(cfg, template, monkeypatch):
    a, sleeps = make(cfg, lambda r: httpx.Response(503, text="down"), monkeypatch=monkeypatch)
    root, children = onto_root(cfg)
    with pytest.raises(ProviderFailure) as e:
        await a.classify(ART, root, children, ClassificationContext("typesafe-ai/jev", template))
    assert e.value.status == "PROVIDER_ERROR" and e.value.attempts == cfg.models["jev"].retries + 1


async def test_auth_failure_is_fatal(cfg, template, monkeypatch):
    a, sleeps = make(cfg, lambda r: httpx.Response(401, text="no"), monkeypatch=monkeypatch)
    root, children = onto_root(cfg)
    with pytest.raises(FatalProviderError):
        await a.classify(ART, root, children, ClassificationContext("typesafe-ai/jev", template))
    assert sleeps == []


async def test_400_fails_fast(cfg, template, monkeypatch):
    a, sleeps = make(cfg, lambda r: httpx.Response(400, text="bad"), monkeypatch=monkeypatch)
    root, children = onto_root(cfg)
    with pytest.raises(ProviderFailure) as e:
        await a.classify(ART, root, children, ClassificationContext("typesafe-ai/jev", template))
    assert e.value.status == "INVALID_REQUEST" and sleeps == []


@pytest.mark.parametrize("model,provider", [("other/model", "typesafe-ai"), ("typesafe-ai/jev", "someone-else")])
async def test_identity_mismatch_is_fatal(cfg, template, monkeypatch, model, provider):
    a, _ = make(cfg, lambda r: httpx.Response(200, json=ok_body(json.loads(r.content), model, provider)),
                monkeypatch=monkeypatch)
    root, children = onto_root(cfg)
    with pytest.raises(FatalProviderError):
        await a.classify(ART, root, children, ClassificationContext("typesafe-ai/jev", template))


async def test_missing_answer_is_model_error(cfg, template, monkeypatch):
    def handler(request):
        b = ok_body(json.loads(request.content))
        b["answers"].pop("q2")
        return httpx.Response(200, json=b)

    a, _ = make(cfg, handler, monkeypatch=monkeypatch)
    root, children = onto_root(cfg)
    with pytest.raises(ProviderFailure) as e:
        await a.classify(ART, root, children, ClassificationContext("typesafe-ai/jev", template))
    assert e.value.status == "MODEL_ERROR"


async def test_fallback_header_is_fatal(cfg, template, monkeypatch):
    def handler(request):
        return httpx.Response(200, json=ok_body(json.loads(request.content)),
                              headers={"x-ai-gateway-decision-fallback-triggered": "true"})

    a, _ = make(cfg, handler, monkeypatch=monkeypatch)
    root, children = onto_root(cfg)
    with pytest.raises(FatalProviderError):
        await a.classify(ART, root, children, ClassificationContext("typesafe-ai/jev", template))


async def test_shrink_on_422_resends_shorter_state(cfg, template, monkeypatch):
    sent = []

    def handler(request):
        b = json.loads(request.content)
        sent.append(len(b["state"]))
        if len(b["state"]) > 60:
            return httpx.Response(422, text="That request was rejected")
        return httpx.Response(200, json=ok_body(b, cfg.models["laya"].model, "boundless"))

    monkeypatch.setenv("VERCEL_API_KEY", "k")
    mcfg = cfg.models["laya"].model_copy(update={"shrink_on_422": True})
    a = GatewayAdapter("laya", mcfg, transport=httpx.MockTransport(handler))
    text = "x" * 100
    art = ArticleInput("a", text, "h", 100, 100, False, "head")
    root, children = onto_root(cfg)
    d = await a.classify(art, root, children, ClassificationContext(mcfg.model, template))
    assert sent == [100, 80, 64, 51] and d.state_chars == 51


async def test_422_without_shrink_is_invalid_request(cfg, template, monkeypatch):
    monkeypatch.setenv("VERCEL_API_KEY", "k")
    mcfg = cfg.models["jev"]
    a = GatewayAdapter("jev", mcfg, transport=httpx.MockTransport(lambda r: httpx.Response(422, text="no")))
    root, children = onto_root(cfg)
    with pytest.raises(ProviderFailure) as e:
        await a.classify(ART, root, children, ClassificationContext(mcfg.model, template))
    assert e.value.status == "INVALID_REQUEST"
