import json

import polars as pl
import pytest
import yaml

from classifier import results
from classifier.adapters.base import ProviderFailure
from classifier.adapters.fake import FakeAdapter
from classifier.config import Config
from classifier.runner import prepare_run, run_model


@pytest.fixture
def small_cfg(repo_root, tmp_path, monkeypatch):
    monkeypatch.chdir(repo_root)
    d = yaml.safe_load((repo_root / "configs/classification.yaml").read_text())
    d["run"]["allow_dirty_git"] = True
    d["corpus"]["directory"] = str(tmp_path / "corpus")
    d["output"]["directory"] = str(tmp_path / "out")
    d["output"]["shard_rows"] = 100
    cfg = Config.model_validate(d)
    corpus = pl.DataFrame({
        "article_id": [f"blbg_{i:03d}" for i in range(12)],
        "headline": [f"Headline {i}" for i in range(12)],
        "article": [f"Body of article {i} about bank lending." for i in range(12)],
    })
    (tmp_path / "corpus").mkdir()
    corpus.write_parquet(tmp_path / "corpus/canonical.parquet")
    (tmp_path / "corpus/manifest.json").write_text(json.dumps({
        "corpus_manifest_hash": "c" * 64, "dataset_sha256": cfg.dataset.file_sha256,
        "dedupe": cfg.corpus.dedupe.model_dump(), "normalization_version": cfg.corpus.normalization_version,
        "stats": {},
    }))
    return d, cfg


def variant(d, path, value):
    import copy
    d = copy.deepcopy(d)
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value
    return Config.model_validate(d)


def test_run_identity(small_cfg):
    d, cfg = small_cfg
    base = prepare_run(cfg, limit=None).run_hash
    assert prepare_run(cfg, limit=None).run_hash == base
    assert prepare_run(variant(d, ["models", "jev", "concurrency"], 4), limit=None).run_hash == base
    assert prepare_run(variant(d, ["models", "jev", "retries"], 1), limit=None).run_hash == base
    assert prepare_run(variant(d, ["output", "directory"], "/elsewhere"), limit=None).run_hash == base
    assert prepare_run(variant(d, ["models", "d1", "enabled"], False), limit=None).run_hash != base
    changed = [
        variant(d, ["hierarchy", "traversal", "threshold"], 0.25),
        variant(d, ["hierarchy", "classification", "positive_threshold"], 0.6),
        variant(d, ["models", "jev", "model"], "typesafe-ai/jev-2"),
        variant(d, ["input", "max_chars"], 8000),
    ]
    for c in changed:
        assert prepare_run(c, limit=None).run_hash != base
    assert prepare_run(cfg, limit=5).run_hash != base


def test_run_identity_corpus_and_template(small_cfg, tmp_path):
    d, cfg = small_cfg
    base = prepare_run(cfg, limit=None).run_hash
    m = json.loads((tmp_path / "corpus/manifest.json").read_text())
    m["corpus_manifest_hash"] = "d" * 64
    (tmp_path / "corpus/manifest.json").write_text(json.dumps(m))
    assert prepare_run(cfg, limit=None).run_hash != base
    with pytest.raises(ValueError):
        prepare_run(variant(d, ["templates", "boolean_v1", "sha256"], "e" * 64), limit=None)


async def test_run_and_resume(small_cfg):
    d, cfg = small_cfg
    rc = prepare_run(cfg, limit=None)
    # First session: node "relevance.gsib_activity" fails for everyone.
    a1 = FakeAdapter("jev", fail_nodes={"relevance.gsib_activity": ProviderFailure("TIMEOUT", "timeout")},
                     overrides={"relevance.gsib_activity": 0.9})
    s1 = await run_model(rc, "jev", resume=False, max_cost_usd=None, adapter=a1)
    assert s1["articles"] == 12 and s1["incomplete"] == 12
    man = json.loads((rc.run_dir / "manifest.json").read_text())
    assert man["models"]["jev"]["status"] == "INCOMPLETE"

    with pytest.raises(RuntimeError, match="--resume"):
        await run_model(rc, "jev", resume=False, max_cost_usd=None, adapter=FakeAdapter("jev"))

    # Second session: provider healthy; root decisions are reused from cache.
    a2 = FakeAdapter("jev", overrides={"relevance.gsib_activity": 0.9})
    s2 = await run_model(rc, "jev", resume=True, max_cost_usd=None, adapter=a2)
    assert all(node != "relevance" for _, node in a2.calls)
    assert s2["complete"] == 12

    # Third session: nothing left to do.
    a3 = FakeAdapter("jev")
    s3 = await run_model(rc, "jev", resume=True, max_cost_usd=None, adapter=a3)
    assert s3["articles"] == 0 and a3.calls == []

    man = json.loads((rc.run_dir / "manifest.json").read_text())
    assert man["models"]["jev"]["status"] == "COMPLETE"
    final = pl.read_parquet(rc.run_dir / "results/model=jev/part-000000.parquet")
    assert final["status"].unique().to_list() == ["SUCCESS"]
    assert final.select(pl.col("attempt").max()).item() == 2
    # Each (article, node, child) appears once after compaction.
    assert final.height == final.unique(subset=["article_id", "node_id", "child_id"]).height
    labels = results.leaf_labels(final)
    assert labels.height > 0
