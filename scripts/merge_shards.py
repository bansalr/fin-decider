"""Recombine the shard runs of one model (`classify run --shard I/N`) into a single run directory.

The merged directory has the usual layout (manifest.json, results/model=<m>/part-000000.parquet), so the
report, scoring and adjudication tools read it like any other run. Before merging it checks that the shards
belong together: same model settings, ontology, corpus, dataset and git commit, a complete set of
indices 0..N-1, and every shard COMPLETE with no article in two shards.

    .venv/bin/python scripts/merge_shards.py classification_results/clm-aaa classification_results/clm-bbb ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl


def merge(runs: list[Path], out_root: Path | None = None) -> Path:
    mans = [json.loads((r / "manifest.json").read_text()) for r in runs]
    models = {m for man in mans for m in man["models"]}
    if len(models) != 1:
        raise ValueError(f"shards are from different models: {sorted(models)}")
    model = models.pop()
    scopes = [man["scope"] for man in mans]
    if any(s.get("kind") != "shard" for s in scopes):
        raise ValueError("every run must be a --shard run")
    counts = {s["count"] for s in scopes}
    if len(counts) != 1:
        raise ValueError(f"shards disagree on the shard count: {sorted(counts)}")
    n = counts.pop()
    if sorted(s["index"] for s in scopes) != list(range(n)):
        raise ValueError(f"need exactly one run per shard 0..{n - 1}, got {sorted(s['index'] for s in scopes)}")
    checks = {"model settings": lambda m: m["models"][model].get("model_config_hash"),
              "git commit": lambda m: m["git"]["commit"], "ontology": lambda m: m["ontology"],
              "corpus": lambda m: m["corpus"], "dataset": lambda m: m["dataset"]}
    for what, get in checks.items():  # shared_inputs_sha256 includes the scope, so it differs per shard by design
        if len({json.dumps(get(man), sort_keys=True) for man in mans}) > 1:
            raise ValueError(f"shards differ in {what}")
    statuses = [man["models"][model].get("status") for man in mans]
    if any(s != "COMPLETE" for s in statuses):
        raise ValueError(f"every shard must be COMPLETE, got {statuses}")

    frames = [pl.read_parquet(sorted((r / "results" / f"model={model}").glob("*.parquet"))) for r in runs]
    ids = [set(f["article_id"].unique().to_list()) for f in frames]
    if sum(len(x) for x in ids) != len(set().union(*ids)):
        raise ValueError("an article appears in more than one shard")
    merged = pl.concat(frames, how="vertical_relaxed").sort(["article_id", "node_id", "child_id"])

    order = sorted(range(n), key=lambda i: scopes[i]["index"])
    run_ids = [mans[i]["run_id"] for i in order]
    digest = hashlib.sha256("\n".join(run_ids).encode()).hexdigest()[:16]
    out = (out_root or runs[0].parent) / f"{model}-{digest}_merged"
    (out / "results" / f"model={model}").mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out / "results" / f"model={model}" / "part-000000.parquet")
    man = dict(mans[order[0]])
    man["models"] = {model: dict(man["models"][model], status="COMPLETE",
                                 complete_articles=sum(m["models"][model].get("complete_articles") or 0 for m in mans),
                                 elapsed_s_total=max(m["models"][model].get("elapsed_s_total") or 0 for m in mans))}
    man.update(run_id=out.name,
               scope={"kind": "merged_shards", "count": n, "articles": sum(s["articles"] for s in scopes),
                      "of": scopes[0].get("of"), "shards": run_ids})
    (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out-root", type=Path, default=None, help="default: the shards' parent folder")
    a = ap.parse_args()
    out = merge(a.runs, a.out_root)
    print(f"merged {len(a.runs)} shards -> {out}")


if __name__ == "__main__":
    main()
