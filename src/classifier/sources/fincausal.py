"""FinCausal (FNP 2020-2022 shared tasks, github.com/yseop/YseopLab).

Short text segments (1-3 sentences) from financial news (2020) and SEC 10-K MD&A
sections (2022), with causality annotations. Ingested as its own corpus: one row per
distinct segment in the canonical article schema (empty headline), plus a provenance
table keeping every source row and its FinCausal labels.

Pinning: the repo is fetched at a 40-hex commit, and `files_sha256` hashes the sorted
(path, sha256) list of every ingested CSV, so any change to the data files fails fast.
"""

from __future__ import annotations

import csv
import hashlib
import io
import subprocess
from pathlib import Path

import polars as pl

from ..provenance import sha256_file, sha256_text
from ..textnorm import normalize

GLOBS = ("FNP_2020_FinCausal/data/**/*.csv", "FNP_2021_FinCausal/data/*.csv",
         "FNP_2022_FinCausal/data/*.csv", "FNP_2022_FinCausal/annotation/tmp/*/*.csv")


def data_files(repo: Path) -> list[Path]:
    files = {p for g in GLOBS for p in repo.glob(g)}
    return sorted(files, key=lambda p: str(p.relative_to(repo)))


def files_sha256(repo: Path) -> str:
    h = hashlib.sha256()
    for p in data_files(repo):
        h.update(f"{p.relative_to(repo)}\t{sha256_file(p)}\n".encode())
    return h.hexdigest()


def fetch(ds) -> Path:
    repo = Path(ds.local_dir)
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", f"https://github.com/{ds.repository}.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "fetch", "-q", "origin", ds.revision], check=False)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", ds.revision], check=True)
    return repo


def _rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**9)
    raw = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw), delimiter=";")
    return [{(k or "").strip(): (v or "") for k, v in r.items()} for r in reader]


def ingest(repo: Path, ds=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """-> (articles in the canonical schema, one row per source row; provenance)."""
    arts, prov = [], []
    i = 0
    for f in data_files(repo):
        rel = str(f.relative_to(repo))
        year = rel.split("_")[1]
        task = "task1" if "task1" in f.name else "task2"
        for r in _rows(f):
            text = r.get("Text", "").strip()
            if not text:
                continue
            norm = normalize(text)
            aid = "fcl_" + sha256_text(f"\n\n{norm}")
            arts.append({
                "row_index": i, "article_id": aid, "source": "FinCausal", "headline": "", "article": text,
                "date": "", "url": "", "journalists": [], "source_hash": "sha256:" + sha256_text(text),
                "content_hash": sha256_text(f"\n{norm}"),
            })
            prov.append({"row_index": i, "article_id": aid, "file": rel, "year": year, "task": task,
                         "orig_index": r.get("Index", "").strip(),
                         "gold_causal": r.get("Gold", "").strip() or None,
                         "cause": r.get("Cause", "").strip() or None, "effect": r.get("Effect", "").strip() or None})
            i += 1
    art_schema = {"row_index": pl.Int64, "article_id": pl.String, "source": pl.String, "headline": pl.String,
                  "article": pl.String, "date": pl.String, "url": pl.String, "journalists": pl.List(pl.String),
                  "source_hash": pl.String, "content_hash": pl.String}
    prov_schema = {"row_index": pl.Int64, "article_id": pl.String, "file": pl.String, "year": pl.String,
                   "task": pl.String, "orig_index": pl.String, "gold_causal": pl.String, "cause": pl.String,
                   "effect": pl.String}
    return pl.DataFrame(arts, schema=art_schema), pl.DataFrame(prov, schema=prov_schema)
