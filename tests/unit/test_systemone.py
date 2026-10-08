import json

import httpx
import pytest

from classifier.adapters.base import ArticleInput, ClassificationContext, FatalProviderError, ProviderFailure
from classifier.adapters.gateway import GatewayAdapter
from classifier.adapters.systemone import SystemOneAdapter
from classifier.ontology import load_ontology

ART = ArticleInput("a", "Bank X led a bond sale.", "h", 23, 23, False, "head")
MODELS = ["strands", "clm", "matilda"]


def response(req, model, p=0.7):
    q = json.loads(req.content)["questions"]
    return {"model": model, "answers": {k: {"type": "noul", "noul": p} for k in q},
            "usage": {"input_tokens": 321, "output_tokens": 0}}


def make(cfg, key, handler, **upd):
    async def no_sleep(_):
        return None

    mcfg = cfg.models[key].model_copy(update=upd)
    return SystemOneAdapter(key, mcfg, transport=httpx.MockTransport(handler), sleep=no_sleep)


def root(cfg):
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    return onto.node("relevance"), onto.children("relevance")


@pytest.mark.parametrize("key", MODELS)
async def test_request_and_decode(cfg, template, key):
    seen = {}
    expected = cfg.models[key].expected_model_returned

    def handler(req):
        seen.update(path=req.url.path, url=str(req.url), auth=req.headers.get("authorization"),
                    body=json.loads(req.content))
        return httpx.Response(200, json=response(req, expected))

    a = make(cfg, key, handler)
    node, kids = root(cfg)
    d = await a.classify(ART, node, kids, ClassificationContext(cfg.models[key].model, template))
    assert seen["path"] == "/v1/systemone" and seen["url"].startswith(cfg.models[key].endpoint)
    assert seen["auth"] is None  # no API key for local servers
    assert all(q["type"] == "noul" for q in seen["body"]["questions"].values())
    assert seen["body"].get("model") == cfg.models[key].request_model
    assert d.scores == {k.id: 0.7 for k in kids} and d.input_tokens == 321 and d.cost_usd == 0.0
    await a.aclose()


async def test_questions_identical_to_gateway_models(cfg, template, monkeypatch):
    monkeypatch.setenv("VERCEL_API_KEY", "k")
    bodies = {}

    def gw(req):
        bodies["jev"] = json.loads(req.content)
        b = bodies["jev"]
        return httpx.Response(200, json={"model": cfg.models["jev"].model,
                                         "answers": {k: {"probability": 0.5} for k in b["questions"]},
                                         "providerMetadata": {"gateway": {"routing": {"finalProvider": "typesafe-ai"}}}})

    node, kids = root(cfg)
    g = GatewayAdapter("jev", cfg.models["jev"], transport=httpx.MockTransport(gw))
    await g.classify(ART, node, kids, ClassificationContext(cfg.models["jev"].model, template))
    for key in MODELS:
        exp = cfg.models[key].expected_model_returned

        def h(req, exp=exp, key=key):
            bodies[key] = json.loads(req.content)
            return httpx.Response(200, json=response(req, exp))

        a = make(cfg, key, h)
        await a.classify(ART, node, kids, ClassificationContext(cfg.models[key].model, template))
        as_bool = {k: dict(q, type="boolean") for k, q in bodies[key]["questions"].items()}
        assert as_bool == bodies["jev"]["questions"] and bodies[key]["state"] == bodies["jev"]["state"]


async def test_wrong_model_is_fatal(cfg, template):
    a = make(cfg, "matilda", lambda r: httpx.Response(200, json=response(r, "something-else")))
    node, kids = root(cfg)
    with pytest.raises(FatalProviderError):
        await a.classify(ART, node, kids, ClassificationContext("x", template))


async def test_413_shrinks_for_matilda(cfg, template):
    sent = []

    def handler(req):
        b = json.loads(req.content)
        sent.append(len(b["state"]))
        if len(b["state"]) > 600:
            return httpx.Response(413, json={"detail": "maximum context length"})
        return httpx.Response(200, json=response(req, "matilda-jev-v1"))

    a = make(cfg, "matilda", handler)
    art = ArticleInput("a", "x" * 1000, "h", 1000, 1000, False, "head")
    node, kids = root(cfg)
    d = await a.classify(art, node, kids, ClassificationContext("x", template))
    assert sent == [1000, 800, 640, 512] and d.state_chars == 512


async def test_busy_529_is_retried(cfg, template):
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(529, headers={"retry-after": "1"}, text="busy")
        return httpx.Response(200, json=response(req, "matilda-jev-v1"))

    a = make(cfg, "matilda", handler)
    node, kids = root(cfg)
    d = await a.classify(ART, node, kids, ClassificationContext("x", template))
    assert d.attempts == 2


def test_local_server_validation(repo_root):
    import yaml
    from pydantic import ValidationError

    from classifier.config import Config

    d = yaml.safe_load((repo_root / "configs/classification.yaml").read_text())
    d["models"]["clm"]["revisions"]["Qwen/Qwen3-8B"] = "main"
    with pytest.raises(ValidationError, match="40-hex"):
        Config.model_validate(d)
    d = yaml.safe_load((repo_root / "configs/classification.yaml").read_text())
    del d["models"]["strands"]["expected_model_returned"]
    with pytest.raises(ValidationError):
        Config.model_validate(d)
