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


def test_v2_templates_need_v1_1_fields(cfg, repo_root):
    import pytest
    from classifier.provenance import sha256_file
    from classifier.templates import TemplateError, load_template

    p = repo_root / "templates/decision/boolean_v2a.txt"
    t = load_template(p, sha256_file(p), "boolean")
    with pytest.raises(TemplateError, match="lacks fields"):
        t.check_ontology(load_ontology(repo_root / "gsib_basel_ontology_v1.0.json"))
    v11 = load_ontology(repo_root / "gsib_basel_ontology_v1.1.json")
    t.check_ontology(v11)
    q = t.render(v11.node("relevance"), v11.node("relevance.gsib_activity"))
    assert q["instructions"] == v11.node("relevance.gsib_activity").yes_no_question
    assert "Not:" not in q["instructions"]
    assert q["criteria"]["false"] == v11.node("relevance.gsib_activity").criteria_dict["false"]


def test_v1_1_maps_every_v1_0_node(repo_root):
    import json

    a = load_ontology(repo_root / "gsib_basel_ontology_v1.0.json")
    b = load_ontology(repo_root / "gsib_basel_ontology_v1.1.json")
    raw = json.loads((repo_root / "gsib_basel_ontology_v1.1.json").read_text())["nodes"]
    covered = [m for n in raw.values() for m in n["merged_from"]]
    assert set(covered) == set(a.nodes)
    assert len(b.nodes) < len(a.nodes) and b.roots == a.roots
    assert all(n.yes_no_question and n.criteria for n in b.nodes.values() if n.parent)
    # Unchanged top of the hierarchy keeps its IDs.
    assert set(a.node("relevance").children) == set(b.node("relevance").children)
    assert set(a.node("relevance.gsib_activity").children) == set(b.node("relevance.gsib_activity").children)


def test_dev_cases_match_ontology_versions(repo_root):
    from report.devset import CASES_BY_ONTOLOGY, load_cases

    for version, path in CASES_BY_ONTOLOGY.items():
        onto = load_ontology(repo_root / f"gsib_basel_ontology_v{version.rsplit('.', 1)[0]}.json")
        assert onto.version == version
        for c in load_cases(path):
            assert set(c["expect"]) <= set(onto.nodes), (version, c["id"])


def test_unknown_placeholder_rejected(tmp_path):
    import pytest
    from classifier.provenance import sha256_file
    from classifier.templates import TemplateError, load_template

    p = tmp_path / "t.txt"
    p.write_text("[instructions]\n{nonsense}\n")
    with pytest.raises(TemplateError, match="unknown placeholders"):
        load_template(p, sha256_file(p), "boolean")
