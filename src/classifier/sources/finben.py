"""FinBen datasets (Xie et al., NeurIPS 2024; huggingface.co/TheFinAI, CC BY-SA 3.0, gated: requires
an HF token with access granted).

- en-headlines: gold-commodity news headlines (Sinha & Khandait), packaged as one instruction per
  (headline, question type). Unique headlines are extracted from the "Text:" part of the query; the
  "Price or Not" answer becomes the reporting group. Expectation (spec §0.14): commodity market news
  -> not G-SIB activity, market context only.
- en-ner: sentences from SEC-filed credit agreements (Alvarado et al., 2015). Reported as one group;
  no per-item expectation (many sentences are only party names or boilerplate).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import polars as pl

from .common import CTX, GSIB, Builder, hash_files

FILES = {
    "TheFinAI/en-headlines": ["data/test-00000-of-00001-2a1cddfd1b6bd5cb.parquet",
                              "data/train-00000-of-00001-b4185af3fde5de2a.parquet",
                              "data/valid-00000-of-00001-eaa51bb586dd9ed5.parquet"],
    "TheFinAI/en-ner": ["data/test-00000-of-00001-440604057ec20623.parquet",
                        "data/train-00000-of-00001-e198e05854da56b5.parquet",
                        "data/valid-00000-of-00001-67538f97a04e37fa.parquet"],
}
_TEXT = re.compile(r"\nText:\s*(.*?)\s*(?:\nAnswer:.*)?$", re.S)


def fetch(ds) -> Path:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for gated FinBen datasets")
    for f in FILES[ds.repository]:
        hf_hub_download(ds.repository, f, repo_type="dataset", revision=ds.revision, local_dir=ds.local_dir,
                        token=token)
    return Path(ds.local_dir)


def files_sha256(local: Path) -> str:
    files = [local / f for fs in FILES.values() for f in fs if (local / f).exists()]
    return hash_files(local, files)


def ingest(local: Path, ds):
    if ds.repository.endswith("en-headlines"):
        b = Builder("gold", "GoldHeadlines")
        price: dict[str, str] = {}
        split_of: dict[str, str] = {}
        for f in FILES[ds.repository]:
            df = pl.read_parquet(local / f)
            for r in df.iter_rows(named=True):
                m = _TEXT.search(r["query"] or "")
                if not m:
                    continue
                h = m.group(1).strip()
                split_of.setdefault(h, f.split("/")[-1].split("-")[0])
                # The train split leaves label_type blank; the question type is still in the query text.
                if r["label_type"] == "Price or Not" or "Price or Not" in (r["query"] or ""):
                    price[h] = "price" if str(r["answer"]).strip().lower() == "yes" else "no_price"
        for h in sorted(split_of):
            b.add(h, split=split_of[h], label=price.get(h, ""), group=price.get(h, "unknown"),
                  expect={GSIB: 0, CTX: 1})
        return b.frames()
    b = Builder("cagr", "CreditAgreementNER")
    for f in FILES[ds.repository]:
        for i, r in enumerate(pl.read_parquet(local / f).iter_rows(named=True)):
            b.add(r["text"], source_id=f"{f.split('/')[-1].split('-')[0]}:{i}", split=f.split("/")[-1].split("-")[0],
                  group="credit_agreement", expect={})
    return b.frames()
