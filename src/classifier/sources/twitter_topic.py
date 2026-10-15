"""Twitter Financial News Topic (huggingface.co/datasets/zeroshot/twitter-financial-news-topic, MIT).

21,107 finance-news tweets with 20 topic labels. Expectations (fixed before any run, spec §0.14):
clearly market-context topics -> not G-SIB activity, market context only. Other topics carry no
expectation; they are reported per group.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .common import CTX, GSIB, Builder, hash_files, hf_fetch

FILES = ["topic_train.csv", "topic_valid.csv"]
LABELS = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
          "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
          "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
          "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]
MARKET_CONTEXT = {"Fed | Central Banks", "Macro", "Currencies", "Stock Movement", "Stock Commentary",
                  "Energy | Oil", "Gold | Metals | Materials"}


def fetch(ds) -> Path:
    return hf_fetch(ds.repository, ds.revision, FILES, ds.local_dir)


def files_sha256(local: Path) -> str:
    return hash_files(local, [local / f for f in FILES])


def ingest(local: Path, ds):
    b = Builder("tw", "TwitterFinancialNewsTopic")
    for f in FILES:
        for r in pl.read_csv(local / f).iter_rows(named=True):
            topic = LABELS[int(r["label"])]
            exp = {GSIB: 0, CTX: 1} if topic in MARKET_CONTEXT else {}
            b.add(r["text"], split=f.split("_")[1].split(".")[0], label=topic, group=topic, expect=exp)
    return b.frames()
