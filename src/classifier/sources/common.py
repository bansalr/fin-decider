"""Shared helpers for corpus sources.

Every source module provides:
    fetch(ds) -> Path                 download/refresh local copy (idempotent)
    files_sha256(local: Path) -> str  pin over the exact files ingested
    ingest(local: Path, ds) -> (articles, provenance)

`articles` follows the canonical schema used by dedupe and the classifier.
`provenance` keeps one row per source row: its original id/split/label, a `group`
for per-group reporting, and `expect` = JSON {node_id: 0|1} fixed in code BEFORE any
run (spec §0.14), used by scripts/score_benchmark.py. Empty {} = no expectation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from ..provenance import sha256_file, sha256_text
from ..textnorm import normalize

ART_SCHEMA = {"row_index": pl.Int64, "article_id": pl.String, "source": pl.String, "headline": pl.String,
              "article": pl.String, "date": pl.String, "url": pl.String, "journalists": pl.List(pl.String),
              "source_hash": pl.String, "content_hash": pl.String}
PROV_SCHEMA = {"row_index": pl.Int64, "article_id": pl.String, "source_id": pl.String, "split": pl.String,
               "label": pl.String, "group": pl.String, "expect": pl.String, "extra": pl.String}

GSIB, CTX, NEITHER = "relevance.gsib_activity", "relevance.market_context_only", "relevance.neither"


def hash_files(local: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files, key=lambda p: str(p.relative_to(local))):
        h.update(f"{p.relative_to(local)}\t{sha256_file(p)}\n".encode())
    return h.hexdigest()


class Builder:
    """Accumulates (article, provenance) rows with deterministic ids."""

    def __init__(self, prefix: str, source: str):
        self.prefix, self.source = prefix, source
        self.arts: list[dict] = []
        self.prov: list[dict] = []

    def add(self, text: str, *, headline: str = "", date: str = "", url: str = "", source_id: str = "",
            split: str = "", label: str = "", group: str = "", expect: dict | None = None,
            extra: dict | None = None) -> None:
        text = (text or "").strip()
        headline = (headline or "").strip()
        if not text and not headline:
            return
        nh, nb = normalize(headline), normalize(text)
        i = len(self.arts)
        aid = f"{self.prefix}_" + sha256_text(f"{nh}\n{date}\n{nb}")
        self.arts.append({"row_index": i, "article_id": aid, "source": self.source, "headline": headline,
                          "article": text, "date": date, "url": url, "journalists": [],
                          "source_hash": "sha256:" + sha256_text(f"{headline}\n{date}\n{text}"),
                          "content_hash": sha256_text(f"{nh}\n{nb}")})
        self.prov.append({"row_index": i, "article_id": aid, "source_id": str(source_id), "split": split,
                          "label": label, "group": group, "expect": json.dumps(expect or {}, sort_keys=True),
                          "extra": json.dumps(extra or {}, sort_keys=True, default=str)})

    def frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        return pl.DataFrame(self.arts, schema=ART_SCHEMA), pl.DataFrame(self.prov, schema=PROV_SCHEMA)


def download(url: str, dest: Path) -> Path:
    """Stream a URL to dest (atomic rename). httpx ships its own CA bundle (certifi)."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def stream_lines(url: str, token: str | None = None):
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.stream("GET", url, follow_redirects=True, headers=headers,
                      timeout=httpx.Timeout(60, read=900)) as r:
        r.raise_for_status()
        yield from r.iter_lines()


def hf_fetch(repo: str, revision: str, files: list[str], local_dir: str) -> Path:
    from huggingface_hub import hf_hub_download

    for f in files:
        hf_hub_download(repo, f, repo_type="dataset", revision=revision, local_dir=local_dir)
    return Path(local_dir)


def split_passages(text: str, max_chars: int) -> list[str]:
    """Split long text on paragraph boundaries into passages of at most max_chars."""
    paras = [p.strip() for p in text.replace("\r", "").split("\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > max_chars:  # an over-long paragraph: hard cut
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:max_chars])
            p = p[max_chars:]
        if len(cur) + len(p) + 1 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out
