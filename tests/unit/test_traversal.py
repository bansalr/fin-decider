import pytest

from classifier.adapters.base import ArticleInput, ClassificationContext, ProviderFailure
from classifier.adapters.fake import FakeAdapter
from classifier.config import Hierarchy
from classifier.traversal import classify_article

ART = ArticleInput("art1", "text", "h1", 4, 4, False, "head")


def hier(trav=0.30, pos=0.50, max_depth=10, max_nodes=100):
    return Hierarchy.model_validate({
        "traversal": {"mode": "breadth_first", "threshold": trav, "max_depth": max_depth, "max_nodes_per_article": max_nodes},
        "classification": {"positive_threshold": pos},
        "materialize_skipped_by_routing": False,
    })


async def run(toy, template, overrides, **kw):
    fail = kw.pop("fail_nodes", None)
    cache = kw.pop("cache", None)
    adapter = FakeAdapter(overrides=overrides, fail_nodes=fail)
    out = await classify_article(ART, toy, adapter, ClassificationContext("fake", template), hier(**kw),
                                 key_fn=lambda n: f"k:{n}", cache=cache)
    return out, adapter


ALL_HIGH = {c: 0.9 for c in ["a", "b", "c", "a1", "a2", "b1", "a1x"]}


async def test_multi_branch(toy, template):
    out, adapter = await run(toy, template, ALL_HIGH)
    assert [n for _, n in adapter.calls] == ["root", "a", "b", "a1"]  # breadth-first, both branches
    assert {r["child_id"] for r in out.rows if r["selected_positive"]} == set(ALL_HIGH)
    # One row per child of every visited node.
    assert len(out.rows) == 3 + 2 + 1 + 1
    assert out.complete


async def test_low_parent_still_traversed(toy, template):
    out, adapter = await run(toy, template, ALL_HIGH | {"a": 0.42})
    rows = {r["child_id"]: r for r in out.rows}
    assert not rows["a"]["selected_positive"] and rows["a"]["selected_for_traversal"]
    assert ("art1", "a") in adapter.calls
    assert rows["a1"]["path_score"] == pytest.approx(0.42 * 0.9)


async def test_below_traversal_threshold_prunes(toy, template):
    _, adapter = await run(toy, template, ALL_HIGH | {"a": 0.29})
    assert ("art1", "a") not in adapter.calls


async def test_traversal_and_positive_thresholds_independent(toy, template):
    out, _ = await run(toy, template, ALL_HIGH | {"b": 0.6}, trav=0.7, pos=0.5)
    rows = {r["child_id"]: r for r in out.rows}
    assert rows["b"]["selected_positive"] and not rows["b"]["selected_for_traversal"]


async def test_max_depth(toy, template):
    _, adapter = await run(toy, template, ALL_HIGH, max_depth=2)
    assert ("art1", "a1") not in adapter.calls


async def test_max_nodes(toy, template):
    _, adapter = await run(toy, template, ALL_HIGH, max_nodes=2)
    assert len(adapter.calls) <= 2


async def test_failure_is_not_negative(toy, template):
    out, adapter = await run(toy, template, ALL_HIGH, fail_nodes={"a": ProviderFailure("TIMEOUT", "timeout")})
    failed = [r for r in out.rows if r["node_id"] == "a"]
    assert failed and all(r["status"] == "TIMEOUT" and r["score"] is None for r in failed)
    assert all(not r["selected_positive"] and not r["selected_for_traversal"] for r in failed)
    assert ("art1", "a1") not in adapter.calls  # no descent below a failure
    assert not out.complete


async def test_cache_reuse(toy, template):
    first, _ = await run(toy, template, ALL_HIGH)
    from classifier.adapters.base import NodeDecision
    cache = {}
    for r in first.rows:
        cache.setdefault(r["prediction_key"], NodeDecision(r["node_id"], {})).scores[r["child_id"]] = r["score"]
    out, adapter = await run(toy, template, {}, cache=cache)
    assert adapter.calls == []
    assert [r["score"] for r in out.rows] == [r["score"] for r in first.rows]
