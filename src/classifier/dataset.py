"""Pinned dataset acquisition and ingestion (spec §5, §6)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from .config import Dataset
from .provenance import sha256_file, sha256_text
from .textnorm import normalize


class DatasetHashMismatch(RuntimeError):
    pass


def fetch(ds: Dataset) -> Path:
    """Download the pinned file (cached) and verify its SHA-256 before use."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            ds.repository, ds.file, repo_type="dataset", revision=ds.revision, local_dir=ds.local_dir
        )
    )
    verify(path, ds.file_sha256)
    return path


def verify(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise DatasetHashMismatch(f"{path}: expected sha256 {expected}, got {actual}")


def _iso(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    return normalize(str(value))


def article_id(headline: str, date: str, body: str) -> str:
    """Deterministic, row-order-independent identity (spec §5)."""
    return "blbg_" + sha256_text(f"{normalize(headline)}\n{date}\n{normalize(body)}")


def ingest(path: Path) -> pl.DataFrame:
    """Load the raw parquet into the canonical article schema, one row per source row."""
    raw = pl.read_parquet(path)
    expected = {"Headline", "Journalists", "Date", "Link", "Article"}
    if set(raw.columns) != expected:
        raise ValueError(f"unexpected dataset columns {raw.columns}; expected {sorted(expected)}")

    rows = []
    for i, r in enumerate(raw.iter_rows(named=True)):
        headline = r["Headline"] or ""
        body = r["Article"] or ""
        date = _iso(r["Date"])
        norm_h, norm_b = normalize(headline), normalize(body)
        rows.append(
            {
                "row_index": i,
                "article_id": "blbg_" + sha256_text(f"{norm_h}\n{date}\n{norm_b}"),
                "source": "Bloomberg",
                "headline": headline,
                "article": body,
                "date": date,
                "url": r["Link"] or "",
                "journalists": list(r["Journalists"] or []),
                "source_hash": "sha256:" + sha256_text(f"{headline}\n{date}\n{body}"),
                "content_hash": sha256_text(f"{norm_h}\n{norm_b}"),
            }
        )
    return pl.DataFrame(rows)
