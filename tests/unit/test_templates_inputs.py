from classifier.config import Input
from classifier.inputs import build_input
from classifier.ontology import load_ontology


def test_render_drops_empty_lines(cfg, template):
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    root = onto.node("relevance")
    q = template.render(root, onto.node("relevance.market_context_only"))
    assert q["type"] == "boolean"
    assert "Also known as" not in q["instructions"]  # empty synonyms
    assert "Excludes: bank financing" in q["instructions"]
    assert set(q["criteria"]) == {"true", "false"}


def test_input_truncation_and_normalization(cfg):
    inp = cfg.input.model_copy(update={"max_chars": 20})
    a = build_input("id", "Head <b>line</b>", "Body   text " * 10, inp)
    assert a.text.startswith("Head line\n\nBody text")
    assert a.submitted_chars == 20 and a.truncated and a.original_chars > 20
    b = build_input("id", "Head line", "Body text " * 10, inp)
    assert a.input_hash == b.input_hash
