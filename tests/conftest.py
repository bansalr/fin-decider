import copy
import json
from pathlib import Path

import pytest

from classifier.config import load_config
from classifier.ontology import build_ontology
from classifier.templates import load_template

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root():
    return ROOT


@pytest.fixture
def cfg():
    import os
    os.chdir(ROOT)
    return load_config(ROOT / "configs/classification.yaml")


@pytest.fixture
def template(cfg):
    t = cfg.templates["boolean_v1"]
    return load_template(ROOT / t.path, t.sha256, t.kind)


def node(name, type_, parent=None, children=(), question="Q?"):
    return {
        "name": name, "type": type_, "parent": parent, "children": list(children),
        "question": question if type_ != "leaf" else None,
        "definition": f"{name} def", "includes": [f"{name} inc"], "excludes": [], "synonyms": [],
    }


@pytest.fixture
def toy_raw():
    """root -> {a, b, c(leaf)}; a -> {a1, a2}; b -> {b1}; a1 -> {a1x}"""
    nodes = {
        "root": node("Root", "multi_label", None, ["a", "b", "c"]),
        "a": node("A", "multi_label", "root", ["a1", "a2"]),
        "b": node("B", "multi_label", "root", ["b1"]),
        "c": node("C", "leaf", "root"),
        "a1": node("A1", "multi_label", "a", ["a1x"]),
        "a2": node("A2", "leaf", "a"),
        "b1": node("B1", "leaf", "b"),
        "a1x": node("A1X", "leaf", "a1"),
    }
    return {"ontology": {"id": "toy", "version": "0.0.1", "roots": ["root"]}, "nodes": nodes}


@pytest.fixture
def toy(toy_raw):
    return build_ontology(copy.deepcopy(toy_raw), "0" * 64)
