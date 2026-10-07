"""Result storage (spec §24, §25, §28).

Rows are article × model × node × child. Per-call fields (tokens, cost, latency,
generation_id) repeat on every child row of the same call; aggregate them by
prediction_key, not by row.

Work shards are append-only; an article's rows are written only once the article
has finished traversal. compact() keeps the latest attempt per article.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .adapters.base import NodeDecision

SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.String,
    "article_id": pl.String,
    "model_id": pl.String,
    "model_version": pl.String,
    "task_root": pl.String,
    "node_id": pl.String,
    "node_depth": pl.Int32,
    "child_id": pl.String,
    "score": pl.Float64,
    "selected_positive": pl.Boolean,
    "selected_for_traversal": pl.Boolean,
    "local_threshold": pl.Float64,
    "traversal_threshold": pl.Float64,
    "path_score": pl.Float64,
    "original_chars": pl.Int64,
    "submitted_chars": pl.Int64,
    "truncated": pl.Boolean,
    "truncation_strategy": pl.String,
    "latency_ms": pl.Float64,
    "input_tokens": pl.Int64,
    "output_tokens": pl.Int64,
    "provider_cost_usd": pl.Float64,
    "config_hash": pl.String,
    "ontology_hash": pl.String,
    "dataset_hash": pl.String,
    "corpus_manifest_hash": pl.String,
    "template_hash": pl.String,
    "code_commit": pl.String,
    "status": pl.String,
    "error_code": pl.String,
    "created_at": pl.String,
    "prediction_key": pl.String,
    "attempt": pl.Int32,
    "attempts": pl.Int32,
    "reused_from_cache": pl.Boolean,
    "model_returned": pl.String,
    "routing_provider": pl.String,
    "generation_id": pl.String,
    "state_chars": pl.Int64,
}


def work_dir(run_dir: Path, model: str) -> Path:
    return run_dir / "_work" / f"model={model}"


class ShardWriter:
    def __init__(self, run_dir: Path, model: str, shard_rows: int):
        self.dir = work_dir(run_dir, model)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_rows = shard_rows
        self.buffer: list[dict[str, Any]] = []
        existing = sorted(self.dir.glob("part-*.parquet"))
        self.next_index = (int(existing[-1].stem.split("-")[1]) + 1) if existing else 0

    def add_article(self, rows: list[dict[str, Any]]) -> None:
        self.buffer.extend(rows)
        if len(self.buffer) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        df = pl.DataFrame(self.buffer, schema=SCHEMA)
        tmp = self.dir / f".part-{self.next_index:06d}.parquet.tmp"
        df.write_parquet(tmp)
        tmp.rename(self.dir / f"part-{self.next_index:06d}.parquet")  # atomic publish
        self.next_index += 1
        self.buffer = []


def read_work(run_dir: Path, model: str) -> pl.DataFrame:
    files = sorted(work_dir(run_dir, model).glob("part-*.parquet"))
    if not files:
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(files)


def latest_attempts(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only each article's latest attempt."""
    if df.is_empty():
        return df
    last = df.group_by("article_id").agg(pl.col("attempt").max().alias("_max"))
    return df.join(last, on="article_id").filter(pl.col("attempt") == pl.col("_max")).drop("_max")


def load_cache(run_dir: Path, model: str) -> tuple[dict[str, NodeDecision], set[str], dict[str, int]]:
    """(successful decisions by prediction_key, complete article ids, latest attempt per article)."""
    df = read_work(run_dir, model)
    if df.is_empty():
        return {}, set(), {}
    attempts = dict(df.group_by("article_id").agg(pl.col("attempt").max()).iter_rows())
    latest = latest_attempts(df)
    status_ok = latest.group_by("article_id").agg((pl.col("status") == "SUCCESS").all().alias("ok"))
    complete = set(status_ok.filter(pl.col("ok"))["article_id"].to_list())

    # Only incomplete articles are re-traversed, so only their decisions are worth caching.
    cache: dict[str, NodeDecision] = {}
    ok = df.filter((pl.col("status") == "SUCCESS") & ~pl.col("article_id").is_in(list(complete)))
    for (key,), g in ok.group_by(["prediction_key"]):
        first = g.row(0, named=True)
        cache[key] = NodeDecision(
            node_id=first["node_id"],
            scores=dict(zip(g["child_id"].to_list(), g["score"].to_list())),
            model_returned=first["model_returned"],
            routing_provider=first["routing_provider"],
            latency_ms=first["latency_ms"],
            input_tokens=first["input_tokens"],
            output_tokens=first["output_tokens"],
            cost_usd=first["provider_cost_usd"],
            attempts=first["attempts"] or 1,
            generation_id=first["generation_id"],
            state_chars=first["state_chars"],
        )
    return cache, complete, attempts


def complete_articles(run_dir: Path, model: str) -> set[str]:
    latest = latest_attempts(read_work(run_dir, model).select("article_id", "attempt", "status"))
    if latest.is_empty():
        return set()
    ok = latest.group_by("article_id").agg((pl.col("status") == "SUCCESS").all().alias("ok"))
    return set(ok.filter(pl.col("ok"))["article_id"].to_list())


def compact(run_dir: Path, model: str) -> Path:
    df = latest_attempts(read_work(run_dir, model)).sort(["article_id", "node_depth", "node_id", "child_id"])
    out = run_dir / "results" / f"model={model}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part-*.parquet"):
        old.unlink()
    path = out / "part-000000.parquet"
    df.write_parquet(path)
    return path


def leaf_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Reconstruct positive final labels from node rows (spec §25): every positive child row."""
    return df.filter((pl.col("status") == "SUCCESS") & pl.col("selected_positive")).select(
        "article_id", "model_id", "child_id", "score", "path_score"
    )
