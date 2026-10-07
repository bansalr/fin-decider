import json

import polars as pl
import pytest

from classifier import results
from classifier.ontology import load_ontology
from report import summary
from report.summary import cohen_kappa


def rows(article, model, items, *, cost=0.001, tokens=100, cached=False):
    """items: list of (node_id, child_id, score, prediction_key)."""
    out = []
    for node, child, score, key in items:
        r = {c: None for c in results.SCHEMA}
        r.update(run_id="r", article_id=article, model_id=model, node_id=node, node_depth=0, child_id=child,
                 score=score, selected_positive=score >= 0.5, selected_for_traversal=score >= 0.3,
                 status="SUCCESS", prediction_key=key, attempt=1, attempts=1, reused_from_cache=cached,
                 provider_cost_usd=cost, input_tokens=tokens, output_tokens=0, latency_ms=100.0)
        out.append(r)
    return out


def write_run(tmp, run_id, model, shared, data, scope=None, dry=False, elapsed=10.0):
    d = tmp / run_id
    (d / "results" / f"model={model}").mkdir(parents=True)
    pl.DataFrame(data, schema=results.SCHEMA).write_parquet(d / "results" / f"model={model}" / "part-000000.parquet")
    (d / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "shared_inputs_sha256": shared, "dry_run": dry,
        "scope": scope or {"kind": "engineering_first_n_by_article_id", "limit": 2},
        "corpus": {"stats": {"canonical_articles": 1000}}, "ontology": {"version": "1.0.0"},
        "resolved_config": {"model": {"provider": "vercel_ai_gateway"}},
        "models": {model: {"requested": model, "elapsed_s_total": elapsed}},
    }))
    return d


ROOT3 = ["relevance.gsib_activity", "relevance.market_context_only", "relevance.neither"]


def article(a, model, scores):
    return rows(a, model, [("relevance", c, s, f"{a}-root") for c, s in zip(ROOT3, scores)])


@pytest.fixture
def onto(cfg):
    return load_ontology(cfg.ontology.path, cfg.ontology.sha256)


def test_ops_counts_calls_by_key_and_projects(tmp_path, onto):
    data = article("a1", "jev", [0.9, 0.1, 0.1]) + article("a2", "jev", [0.2, 0.8, 0.1])
    d = write_run(tmp_path, "jev-x", "jev", "S", data)
    rep = summary.build(d, "jev", onto)
    o = rep["ops"]
    assert o["api_calls"] == 2 and o["calls_per_article"] == 1.0  # 6 rows, 2 requests
    assert o["cost_usd"] == pytest.approx(0.002) and o["input_tokens"] == 200
    p = o["projection_full_corpus"]
    assert p["cost_usd"] == pytest.approx(0.002 / 2 * 1000)
    assert p["hours_at_same_concurrency"] == pytest.approx(10.0 / 2 * 1000 / 3600)


def test_cached_decisions_not_counted(tmp_path, onto):
    data = article("a1", "jev", [0.9, 0.1, 0.1]) + rows("a2", "jev", [("relevance", c, 0.5, "k") for c in ROOT3], cached=True)
    rep = summary.build(write_run(tmp_path, "jev-x", "jev", "S", data), "jev", onto)
    assert rep["ops"]["api_calls"] == 1


def test_label_shares(tmp_path, onto):
    data = article("a1", "jev", [0.9, 0.1, 0.1]) + article("a2", "jev", [0.2, 0.8, 0.1])
    lab = summary.build(write_run(tmp_path, "jev-x", "jev", "S", data), "jev", onto)["labels"]
    assert lab["relevance"] == {"relevance.gsib_activity": 0.5, "relevance.market_context_only": 0.5,
                                "relevance.neither": 0.0}
    assert lab["mean_positive_leaves"] == 0 and "relevance.neither" not in lab["top_leaves"]


def test_kappa():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa([1, 1], [1, 1]) is None


def test_agreement_only_with_matching_runs(tmp_path, onto):
    j = write_run(tmp_path, "jev-x", "jev", "S", article("a1", "jev", [0.9, 0.1, 0.1]) + article("a2", "jev", [0.9, 0.1, 0.1]))
    write_run(tmp_path, "laya-x", "laya", "S", article("a1", "laya", [0.9, 0.1, 0.1]) + article("a2", "laya", [0.1, 0.9, 0.1]))
    write_run(tmp_path, "d1-other", "d1", "OTHER", article("a1", "d1", [0.9, 0.1, 0.1]))
    write_run(tmp_path, "d1-dry", "d1", "S", article("a1", "d1", [0.9, 0.1, 0.1]), dry=True)
    ag = summary.build(j, "jev", onto)["agreement"]
    assert set(ag) == {"laya"}
    assert ag["laya"]["articles_compared"] == 2
    assert ag["laya"]["relevance"]["relevance.gsib_activity"]["agreement"] == 0.5


def test_no_comparable_runs(tmp_path, onto):
    j = write_run(tmp_path, "jev-x", "jev", "S", article("a1", "jev", [0.9, 0.1, 0.1]))
    assert summary.build(j, "jev", onto)["agreement"] == {}


def test_classifier_does_not_import_report(repo_root):
    offenders = [p.name for p in (repo_root / "src/classifier").rglob("*.py")
                 if p.name != "cli.py" and ("import report" in p.read_text() or "from report" in p.read_text())]
    assert offenders == []


def test_dry_run_makes_no_network_calls(repo_root, tmp_path, monkeypatch):
    import httpx
    from typer.testing import CliRunner

    from classifier.cli import app

    def boom(*a, **k):
        raise AssertionError("network call during dry run")

    monkeypatch.setattr(httpx.AsyncClient, "send", boom)
    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.chdir(repo_root)
    res = CliRunner().invoke(app, ["run", "--dry-run", "--limit", "2", "-m", "jev", "-m", "clef", "--show", "1",
                                   "--set", "run.allow_dirty_git=true", "--output-dir", str(tmp_path / "out"),
                                   "--reports-dir", str(tmp_path / "rep"), "--progress-every", "1000"])
    assert res.exit_code == 0, res.output
    assert "DRY RUN" in res.output and "Dev-set accuracy" in res.output and "vs clef" in res.output
