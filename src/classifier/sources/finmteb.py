"""FinMTEB-packaged English corpora (huggingface.co/FinanceMTEB; Tang & Yang, EMNLP 2025).

Repackaged for embedding evaluation, so labels are whatever the packaging kept:
- WikiCompany2Industry-en: Wikipedia-style company passages grouped by industry (6 industries).
  No per-item expectation (spec §0.14): a bank passage may only cover history or headquarters, and a
  non-bank passage may describe a genuine lending arm (e.g. a carmaker's finance company). Reported
  per industry: Banking vs Financial services vs non-financial industries.
- TradeTheEventNews: 10,000 corporate-event press releases (EDT corpus), no labels kept -> groups only.
- USnews: 9,999 US financial news articles, no labels -> cross-source check, groups only.
The FinMTEB copies state no license; treat as internal use until the originals' terms are traced.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .common import Builder, hash_files, hf_fetch

CONFIGS = {
    "FinanceMTEB/WikiCompany2Industry-en": ["data/test-00000-of-00001.parquet"],
    "FinanceMTEB/TradeTheEventNews": ["corpus/train-00000-of-00001.parquet"],
    "FinanceMTEB/USnews": ["corpus/train-00000-of-00001.parquet"],
}


def fetch(ds) -> Path:
    return hf_fetch(ds.repository, ds.revision, CONFIGS[ds.repository], ds.local_dir)


def files_sha256(local: Path, repository: str | None = None) -> str:
    files = [p for fs in CONFIGS.values() for f in fs if (p := local / f).exists()]
    return hash_files(local, files)


def ingest(local: Path, ds):
    repo = ds.repository
    if repo.endswith("WikiCompany2Industry-en"):
        b = Builder("wiki", "WikiCompany2Industry")
        df = pl.read_parquet(local / CONFIGS[repo][0])
        for ci, r in enumerate(df.iter_rows(named=True)):
            for si, (text, label) in enumerate(zip(r["sentences"], r["labels"])):
                b.add(text, source_id=f"{ci}:{si}", label=label, group=label, expect={})
        return b.frames()
    prefix, name = ("edt", "TradeTheEventNews") if repo.endswith("TradeTheEventNews") else ("usn", "USnews")
    b = Builder(prefix, name)
    for r in pl.read_parquet(local / CONFIGS[repo][0]).iter_rows(named=True):
        b.add(r["text"], headline=r.get("title") or "", source_id=str(r["_id"]), group=name, expect={})
    return b.frames()
