import csv

import pytest

from adjudication import score


def write(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


COLS = ["label_id", "gsib_activity", "market_context_only", "neither", "markets", "lending", "treasury_alm", "leaves"]


def test_weighted_scoring(tmp_path):
    write(tmp_path / "l.csv", [
        {"label_id": "L1", "gsib_activity": "y", "market_context_only": "n", "neither": "n", "markets": "y",
         "lending": "n", "treasury_alm": "n", "leaves": "markets.macro.rates"},
        {"label_id": "L2", "gsib_activity": "n", "market_context_only": "y", "neither": "n", "leaves": ""},
        {"label_id": "L3", "gsib_activity": "", "market_context_only": "", "neither": ""},  # unlabelled: skipped
    ], COLS)
    write(tmp_path / "k.csv", [{"label_id": "L1", "article_id": "a1", "stratum": "split", "weight": 10},
                               {"label_id": "L2", "article_id": "a2", "stratum": "all_no", "weight": 100},
                               {"label_id": "L3", "article_id": "a3", "stratum": "all_no", "weight": 100}],
          ["label_id", "article_id", "stratum", "weight"])
    labels = score.read_labels(tmp_path / "l.csv")
    assert set(labels) == {"L1", "L2"}
    pred = {"a1": {"relevance.gsib_activity", "business.markets", "markets.macro.rates"},
            "a2": {"relevance.gsib_activity"}}  # false positive on a heavily weighted article
    s = score.score(labels, score.read_key(tmp_path / "k.csv"), pred, {"markets.macro.rates"})
    g = s["relevance.gsib_activity"]
    assert g["precision"] == pytest.approx(10 / 110) and g["recall"] == 1.0
    assert s["leaves(micro)"]["precision"] == 1.0


def test_annotator_agreement(tmp_path):
    a = {"L1": {n: 1 for n in list(score.TOP.values()) + list(score.ARMS.values())} | {"_leaves": set()},
         "L2": {n: 0 for n in list(score.TOP.values()) + list(score.ARMS.values())} | {"_leaves": set()}}
    b = {"L1": dict(a["L1"]), "L2": dict(a["L2"], **{"relevance.gsib_activity": 1})}
    ag = score.annotator_agreement(a, b)
    assert ag["relevance.gsib_activity"]["agreement"] == 0.5 and ag["disagreements"] == ["L2"]


def test_queue_blinds_labellers(tmp_path):
    from adjudication import queue

    picked = [{"label_id": "L0001", "article_id": "blbg_x", "stratum": "split", "weight": 2.0,
               "headline": "h", "date": "d", "url": "u", "article": "t"}]
    paths = queue.write(picked, {"m": 1}, tmp_path, "q")
    header = open(paths["queue"]).readline()
    assert "stratum" not in header and "article_id" not in header and "weight" not in header
    assert "stratum" in open(paths["key"]).readline()
