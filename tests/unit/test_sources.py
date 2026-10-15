import io
import json
import tarfile
import zipfile

import polars as pl
import pytest

from classifier.sources import cfpb, common, edgar_gsib, fomc, reuters21578, source_for, twitter_topic


def test_registry_covers_all_providers():
    from classifier.sources import PROVIDERS

    for p in PROVIDERS:
        m = source_for(p)
        assert all(hasattr(m, f) for f in ("fetch", "files_sha256", "ingest"))


def test_twitter_expectations(tmp_path):
    pl.DataFrame({"text": ["Fed holds rates", "XYZ prices IPO"], "label": [1, 11]}).write_csv(tmp_path / "topic_train.csv")
    pl.DataFrame({"text": ["Dollar slips"], "label": [8]}).write_csv(tmp_path / "topic_valid.csv")
    a, p = twitter_topic.ingest(tmp_path, None)
    exp = [json.loads(x) for x in p["expect"]]
    assert p["group"].to_list() == ["Fed | Central Banks", "IPO", "Currencies"]
    assert exp[0] == {common.GSIB: 0, common.CTX: 1} and exp[1] == {} and a["article_id"][0].startswith("tw_")


def test_fomc_is_negative_control(tmp_path):
    for f in fomc.FILES:
        pl.DataFrame({"index": [1], "sentence": ["Participants expected inflation to ease."], "year": [2019],
                      "label": [0], "orig_index": [1]}).write_csv(tmp_path / f)
    _, p = fomc.ingest(tmp_path, None)
    assert all(json.loads(x) == {common.GSIB: 0, common.CTX: 1} for x in p["expect"])


def test_cfpb_grouping_sampling_and_no_label_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(cfpb, "PER_GROUP", 2)
    monkeypatch.setattr(cfpb, "FILES", ["data/part.parquet"])
    rows = [("2024-01-01", "Mortgage", "Conventional", "x", f"My mortgage servicer {i}", "WELLS FARGO & COMPANY", i)
            for i in range(5)]
    rows += [("2024-01-01", "Credit card", "General-purpose credit card", "x", "Card interest", "CITIBANK, N.A.", 99),
             ("2024-01-01", "Credit card", "General-purpose prepaid card", "x", "Prepaid", "X", 98),
             ("2024-01-01", "Mortgage", "x", "x", None, "X", 97)]  # no narrative: skipped
    cols = ["Date received", "Product", "Sub-product", "Issue", "Consumer complaint narrative", "Company", "Complaint ID"]
    (tmp_path / "data").mkdir()
    pl.DataFrame(rows, schema=cols, orient="row").write_parquet(tmp_path / "data/part.parquet")
    a, p = cfpb.ingest(tmp_path, None)
    assert dict(p.group_by("group").len().iter_rows()) == {"mortgage": 2, "credit_card": 1}
    assert a["headline"].unique().to_list() == [""]  # product is never shown to the model
    m = json.loads(p.filter(pl.col("group") == "mortgage")["expect"][0])
    assert m["lending.residential_mortgage"] == 1 and m["lending.consumer"] == 0
    assert json.loads(p.filter(pl.col("group") == "credit_card")["extra"][0])["us_gsib"] is True
    a2, p2 = cfpb.ingest(tmp_path, None)
    assert p2["source_id"].to_list() == p["source_id"].to_list()  # deterministic


def test_edgar_passages_and_groups(tmp_path):
    md = "\n".join(["Net interest income rose as loan balances grew." * 20] * 12)
    for y in edgar_gsib.YEARS:
        (tmp_path / f"filtered_{y}.jsonl").write_text(
            json.dumps({"filename": f"jpm{y}.txt", "cik": "0000019617", "year": str(y), "section_7": md}) + "\n"
            + json.dumps({"filename": f"aapl{y}.txt", "cik": "320193", "year": str(y), "section_7": md[:6000]}) + "\n"
            + json.dumps({"filename": f"wfc{y}.txt", "cik": "72971", "year": str(y), "section_7": "See annual report."}) + "\n")
    a, p = edgar_gsib.ingest(tmp_path, None)
    assert a["article"].str.len_chars().max() <= edgar_gsib.PASSAGE_CHARS
    groups = set(p["group"].to_list())
    assert groups == {"bank:JPMorgan Chase", "control:Apple"}  # Wells Fargo stub (by reference) dropped
    assert all(json.loads(x)["g_sib"] for x, g in zip(p["extra"], p["group"]) if g.startswith("bank:"))


def test_reuters_modapte(tmp_path):
    sgm = ('<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" NEWID="1"><DATE>26-FEB-1987</DATE><TOPICS><D>money-fx</D></TOPICS>'
           '<TEXT><TITLE>DOLLAR FALLS</TITLE><BODY>The dollar fell &amp; yen rose.</BODY></TEXT></REUTERS>'
           '<REUTERS TOPICS="NO" LEWISSPLIT="TRAIN" NEWID="2"><TEXT><BODY>x</BODY></TEXT></REUTERS>'
           '<REUTERS TOPICS="YES" LEWISSPLIT="NOT-USED" NEWID="3"><TOPICS><D>earn</D></TOPICS><TEXT><BODY>y</BODY></TEXT></REUTERS>')
    tb = io.BytesIO()
    with tarfile.open(fileobj=tb, mode="w:gz") as t:
        data = sgm.encode("latin-1")
        info = tarfile.TarInfo("reut2-000.sgm")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(tmp_path / reuters21578.ZIP, "w") as z:
        z.writestr("reuters21578.tar.gz", tb.getvalue())
    a, p = reuters21578.ingest(tmp_path, None)
    assert a.height == 1 and a["headline"][0] == "DOLLAR FALLS" and "&" in a["article"][0]
    assert p["group"][0] == "money-fx" and p["split"][0] == "train"


def test_split_passages():
    out = common.split_passages("a" * 25 + "\n" + "b" * 10, 20)
    assert all(len(x) <= 20 for x in out) and "".join(out).replace("\n", "") == "a" * 25 + "b" * 10


def test_benchmark_scoring(tmp_path):
    from adjudication import benchmark
    from classifier import results

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pl.DataFrame({"row_index": [0, 1, 2], "article_id": ["x_1", "x_2", "x_3"], "source_id": ["a", "b", "c"],
                  "split": [""] * 3, "label": [""] * 3, "group": ["fed", "fed", "ipo"],
                  "expect": [json.dumps({common.GSIB: 0}), json.dumps({common.GSIB: 0}), "{}"],
                  "extra": ["{}"] * 3}).write_parquet(corpus / "provenance.parquet")
    pl.DataFrame({"article_id": ["x_1", "x_2", "x_3"]}).write_parquet(corpus / "canonical.parquet")
    pl.DataFrame({"article_id": ["x_1", "x_2", "x_3"], "canonical_article_id": ["x_1", "x_2", "x_3"]}).write_parquet(
        corpus / "clusters.parquet")
    run = tmp_path / "run" / "results" / "model=jev"
    run.mkdir(parents=True)
    rows = []
    for aid, g in [("x_1", 0.9), ("x_2", 0.1), ("x_3", 0.8)]:
        r = {c: None for c in results.SCHEMA}
        r.update(article_id=aid, node_id="relevance", child_id=common.GSIB, score=g, selected_positive=g >= .5,
                 status="SUCCESS", attempt=1)
        rows.append(r)
    pl.DataFrame(rows, schema=results.SCHEMA).write_parquet(run / "part-000000.parquet")
    s = benchmark.score(tmp_path / "run", "jev", corpus)
    assert s["per_group"]["fed"][common.GSIB] == 0.5 and s["per_group"]["ipo"][common.GSIB] == 1.0
    e = s["expectations"][common.GSIB]
    assert e["items"] == 2 and e["false_positive_rate"] == 0.5 and e["recall"] is None


def test_finmteb_wiki_and_news(tmp_path):
    from types import SimpleNamespace as NS

    from classifier.sources import finmteb

    (tmp_path / "data").mkdir()
    pl.DataFrame({"sentences": [["A bank lends.", "Cars are built."]], "labels": [["Banking", "Automotive"]]}).write_parquet(
        tmp_path / "data/test-00000-of-00001.parquet")
    a, p = finmteb.ingest(tmp_path, NS(repository="FinanceMTEB/WikiCompany2Industry-en"))
    assert p["group"].to_list() == ["Banking", "Automotive"] and set(p["expect"]) == {"{}"}
    (tmp_path / "corpus").mkdir()
    pl.DataFrame({"_id": ["7"], "text": ["Company X buys back shares."], "title": [""]}).write_parquet(
        tmp_path / "corpus/train-00000-of-00001.parquet")
    a, p = finmteb.ingest(tmp_path, NS(repository="FinanceMTEB/TradeTheEventNews"))
    assert a["article_id"][0].startswith("edt_") and p["source_id"][0] == "7"
