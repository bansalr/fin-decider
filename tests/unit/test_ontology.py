import copy

import pytest

from classifier.ontology import OntologyError, build_ontology, load_ontology


def test_real_ontology_valid(cfg):
    onto = load_ontology(cfg.ontology.path, cfg.ontology.sha256)
    assert onto.roots == ("relevance",)
    assert len(onto.nodes) == 79
    assert "business.treasury_alm" in onto.nodes
    assert all(len(onto.node(n).children) <= 20 for n in onto.nodes)


def test_hash_mismatch_fails(cfg):
    with pytest.raises(OntologyError, match="hash mismatch"):
        load_ontology(cfg.ontology.path, "f" * 64)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda n: n["a"]["children"].append("ghost"), "does not exist"),
        (lambda n: n["c"].update(children=["a2"]), "parent is|leaf has children"),
        (lambda n: n["a"]["children"].append("a1"), "duplicate child"),
        (lambda n: n["a1"].pop("definition"), "missing required field"),
        (lambda n: n["root"].update(parent="a1"), "has a parent|cycle"),
        (lambda n: n["b"].update(children=[]), "non-leaf has no children|not listed"),
    ],
)
def test_structural_errors(toy_raw, mutate, match):
    raw = copy.deepcopy(toy_raw)
    mutate(raw["nodes"])
    with pytest.raises(OntologyError, match=match):
        build_ontology(raw, "0" * 64)


def test_node_hash_changes_with_child_wording(toy_raw):
    a = build_ontology(copy.deepcopy(toy_raw), "x")
    raw = copy.deepcopy(toy_raw)
    raw["nodes"]["a2"]["definition"] = "changed"
    b = build_ontology(raw, "x")
    assert a.node_hash("a") != b.node_hash("a")
    assert a.node_hash("b") == b.node_hash("b")
