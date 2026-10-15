"""Reuters-21578 R(10) native-task benchmark: source selection, task ontology, and scorer."""

import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from classifier.config import load_config
from classifier.ontology import load_ontology
from classifier.provenance import sha256_file
from classifier.sources import reuters21578_top10
from classifier.templates import load_template

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/native_reuters_top10.yaml"


def _scorer():
    spec = importlib.util.spec_from_file_location("score_native_reuters", ROOT / "scripts/score_native_reuters.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _archive(tmp_path, sgm: str) -> None:
    tb = io.BytesIO()
    with tarfile.open(fileobj=tb, mode="w:gz") as t:
        data = sgm.encode("latin-1")
        info = tarfile.TarInfo("reut2-000.sgm")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(tmp_path / "reuters21578.zip", "w") as z:
        z.writestr("reuters21578.tar.gz", tb.getvalue())


def test_selects_modapte_test_stories_with_a_top10_topic(tmp_path):
    sgm = (
        # kept: test, two top-10 topics plus a non-top-10 one
        '<REUTERS TOPICS="YES" LEWISSPLIT="TEST" NEWID="1"><DATE>8-APR-1987</DATE>'
        '<TOPICS><D>wheat</D><D>grain</D><D>barley</D></TOPICS>'
        '<TEXT><TITLE>WHEAT SALE</TITLE><BODY>Egypt bought wheat.</BODY></TEXT></REUTERS>'
        # kept: title-only brief (dropped by the G-SIB reuters21578 source)
        '<REUTERS TOPICS="YES" LEWISSPLIT="TEST" NEWID="2"><TOPICS><D>earn</D></TOPICS>'
        '<TEXT TYPE="BRIEF"><TITLE>XYZ CORP 1ST QTR NET</TITLE></TEXT></REUTERS>'
        # kept: unprocessed text without TITLE/BODY tags
        '<REUTERS TOPICS="YES" LEWISSPLIT="TEST" NEWID="3"><TOPICS><D>acq</D></TOPICS>'
        '<TEXT TYPE="UNPROC">ABC to buy DEF &amp; units</TEXT></REUTERS>'
        # dropped: no top-10 topic / training split / TOPICS="NO"
        '<REUTERS TOPICS="YES" LEWISSPLIT="TEST" NEWID="4"><TOPICS><D>gold</D></TOPICS><TEXT><BODY>g</BODY></TEXT></REUTERS>'
        '<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" NEWID="5"><TOPICS><D>earn</D></TOPICS><TEXT><BODY>t</BODY></TEXT></REUTERS>'
        '<REUTERS TOPICS="NO" LEWISSPLIT="TEST" NEWID="6"><TOPICS><D>earn</D></TOPICS><TEXT><BODY>n</BODY></TEXT></REUTERS>'
    )
    _archive(tmp_path, sgm)
    a, p = reuters21578_top10.ingest(tmp_path, None)
    assert p["source_id"].to_list() == ["1", "2", "3"]
    assert p["label"].to_list() == ["grain;wheat", "earn", "acq"]  # gold in TOP10 order, non-top-10 dropped
    assert json.loads(p["extra"][0])["all_topics"] == ["wheat", "grain", "barley"]
    assert a["article"][1] == "" and a["headline"][1] == "XYZ CORP 1ST QTR NET"
    assert a["article"][2] == "ABC to buy DEF & units"
    assert all(e == "{}" for e in p["expect"])  # no G-SIB expectations


def test_task_ontology_is_flat_top10_and_config_is_separate():
    cfg = load_config(CFG)
    onto = load_ontology(ROOT / cfg.ontology.path, cfg.ontology.sha256)
    leaves = [onto.node(c) for c in onto.node("topics").children]
    assert [n.id.removeprefix("topic.") for n in leaves] == list(reuters21578_top10.TOP10)
    assert all(n.is_leaf and n.yes_no_question for n in leaves)
    t = cfg.templates["reuters_top10_v1"]
    tmpl = load_template(ROOT / t.path, sha256_file(ROOT / t.path), "boolean")
    tmpl.check_ontology(onto)
    # Nothing shared with the G-SIB study's ontology, corpus or results.
    assert "gsib" not in cfg.ontology.path and cfg.ontology.path != load_config(ROOT / "configs/classification.yaml").ontology.path
    assert cfg.corpus.directory.startswith("corpus_native/") and cfg.output.directory == "classification_results_native"
    assert {m.template for m in cfg.models.values()} == {"reuters_top10_v1"}


def test_metrics_follow_sklearn_conventions():
    s = _scorer()
    gold = [{"earn"}, {"acq"}, {"grain", "wheat"}, {"earn"}]
    pred = [{"earn"}, {"earn"}, {"grain"}, set()]
    m = s.metrics(gold, pred, labels=("earn", "acq", "grain", "wheat"))
    assert m["per_topic"]["earn"] == pytest.approx(
        {"support": 2, "predicted": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5})
    assert m["per_topic"]["acq"]["f1"] == 0.0 and m["per_topic"]["grain"]["f1"] == 1.0
    assert m["macro"]["f1"] == pytest.approx((0.5 + 0 + 1 + 0) / 4)
    # micro: tp=2 (earn, grain), fp=1, fn=3
    assert m["micro"]["precision"] == pytest.approx(2 / 3) and m["micro"]["recall"] == pytest.approx(2 / 5)
    assert m["subset_accuracy"] == pytest.approx(1 / 4)
    assert m["hamming_accuracy"] == pytest.approx(1 - 4 / 16)


def test_training_split_is_a_separate_corpus(tmp_path):
    from classifier.sources import reuters21578_top10_train

    _archive(tmp_path, '<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" NEWID="5"><TOPICS><D>earn</D></TOPICS>'
                       '<TEXT><BODY>t</BODY></TEXT></REUTERS>'
                       '<REUTERS TOPICS="YES" LEWISSPLIT="TEST" NEWID="6"><TOPICS><D>earn</D></TOPICS>'
                       '<TEXT><BODY>u</BODY></TEXT></REUTERS>')
    a, p = reuters21578_top10_train.ingest(tmp_path, None)
    assert p["source_id"].to_list() == ["5"] and p["split"][0] == "train" and a["article_id"][0].startswith("rtr10tr_")
    dev, test = load_config(ROOT / "configs/native_reuters_top10_train.yaml"), load_config(CFG)
    assert dev.corpus.directory != test.corpus.directory


def test_v3_ontology_has_criteria_for_every_topic():
    cfg = load_config(ROOT / "configs/native_reuters_top10_v3.yaml")
    onto = load_ontology(ROOT / cfg.ontology.path, cfg.ontology.sha256)
    t = cfg.templates["reuters_top10_v3"]
    tmpl = load_template(ROOT / t.path, sha256_file(ROOT / t.path), "boolean")
    tmpl.check_ontology(onto)
    for c in onto.node("topics").children:
        q = tmpl.render(onto.node("topics"), onto.node(c))
        assert q["instructions"] and q["criteria"]["true"] and q["criteria"]["false"]
