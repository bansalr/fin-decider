from classifier.sources import fincausal


def test_ingest_and_pin(tmp_path):
    d = tmp_path / "FNP_2020_FinCausal" / "data" / "trial"
    d.mkdir(parents=True)
    (d / "fnp2020-fincausal-task1.csv").write_text(
        "Index; Text; Gold\n1.1; Banks lent more because rates fell.; 1\n1.2; Shares rose.; 0\n1.3; Banks lent more because rates fell.; 1\n")
    t = tmp_path / "FNP_2022_FinCausal" / "data"
    t.mkdir(parents=True)
    (t / "blind_fnp2022-fincausal-task2.csv").write_text("Index; Text\n9.1; Shares rose.\n")
    arts, prov = fincausal.ingest(tmp_path)
    assert arts.height == 4 and prov.height == 4
    assert arts["article_id"].n_unique() == 2  # identical text -> identical id across files
    assert arts["headline"].unique().to_list() == [""] and arts["article_id"][0].startswith("fcl_")
    assert prov["gold_causal"].to_list()[:2] == ["1", "0"] and prov["year"].to_list()[-1] == "2022"
    h = fincausal.files_sha256(tmp_path)
    (d / "fnp2020-fincausal-task1.csv").write_text("Index; Text; Gold\n1.1; changed; 0\n")
    assert fincausal.files_sha256(tmp_path) != h
