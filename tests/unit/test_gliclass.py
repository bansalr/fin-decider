import asyncio

import pytest

from classifier.adapters.base import ArticleInput, ClassificationContext, ProviderFailure
from classifier.adapters.gliclass import GLiClassAdapter
from classifier.ontology import load_ontology
from classifier.templates import load_template


class StubBackend:
    """Scores label i as 0.1*(i+1); 'fits' text to 50 chars; records batch sizes."""

    def __init__(self, fail=False):
        self.batches = []
        self.fail = fail

    def fit(self, text, labels):
        return (text[:50], 60, len(text) > 50)

    def score_batch(self, items):
        self.batches.append(len(items))
        if self.fail:
            raise RuntimeError("cuda oom")
        return [{lab: round(0.1 * (i + 1), 3) for i, lab in enumerate(labels)} for _, labels in items]


@pytest.fixture
def setup(cfg):
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    t = cfg.templates["gliclass_label_v1"]
    tmpl = load_template(t.path, t.sha256, t.kind)
    return onto, tmpl, cfg.models["gliclass"]


def art(i, n=80):
    text = f"article {i} " + "x" * n
    return ArticleInput(f"a{i}", text, f"h{i}", len(text), len(text), False, "head")


async def test_scores_every_child_and_records_truncation(setup):
    onto, tmpl, mcfg = setup
    a = GLiClassAdapter("gliclass", mcfg, backend=StubBackend())
    root = onto.node("relevance")
    d = await a.classify(art(1), root, onto.children("relevance"), ClassificationContext(mcfg.model, tmpl))
    assert list(d.scores) == list(root.children)
    assert d.scores["relevance.neither"] == pytest.approx(0.3)
    assert d.model_truncated is True and d.model_submitted_tokens == 60
    assert d.model_returned.endswith("@" + mcfg.revision)
    await a.aclose()


async def test_labels_are_name_and_definition(setup):
    onto, tmpl, _ = setup
    c = onto.node("lending.specialised.object_finance")
    assert tmpl.render_label(c) == f"{c.name}: {c.definition}"


async def test_micro_batching_groups_concurrent_calls(setup):
    onto, tmpl, mcfg = setup
    backend = StubBackend()
    a = GLiClassAdapter("gliclass", mcfg.model_copy(update={"batch_size": 8, "batch_wait_ms": 50}), backend=backend)
    ctx = ClassificationContext(mcfg.model, tmpl)
    nodes = ["relevance", "business.markets", "business.lending"]  # different label sets in one batch
    calls = [a.classify(art(i), onto.node(n), onto.children(n), ctx) for i in range(10) for n in nodes[:1]]
    calls += [a.classify(art(i), onto.node(nodes[1 + i % 2]), onto.children(nodes[1 + i % 2]), ctx) for i in range(6)]
    out = await asyncio.gather(*calls)
    assert len(out) == 16 and all(d.scores for d in out)
    assert sum(backend.batches) == 16 and max(backend.batches) <= 8 and len(backend.batches) < 16
    await a.aclose()


async def test_partial_batch_resolves(setup):
    onto, tmpl, mcfg = setup
    a = GLiClassAdapter("gliclass", mcfg.model_copy(update={"batch_size": 64, "batch_wait_ms": 10}), backend=StubBackend())
    d = await asyncio.wait_for(
        a.classify(art(1), onto.node("relevance"), onto.children("relevance"), ClassificationContext(mcfg.model, tmpl)), 2
    )
    assert d.scores
    await a.aclose()


async def test_backend_error_fails_batch_items_as_model_error(setup):
    onto, tmpl, mcfg = setup
    a = GLiClassAdapter("gliclass", mcfg, backend=StubBackend(fail=True))
    ctx = ClassificationContext(mcfg.model, tmpl)
    results = await asyncio.gather(
        *[a.classify(art(i), onto.node("relevance"), onto.children("relevance"), ctx) for i in range(3)],
        return_exceptions=True,
    )
    assert all(isinstance(r, ProviderFailure) and r.status == "MODEL_ERROR" for r in results)
    await a.aclose()
