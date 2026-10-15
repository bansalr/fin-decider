"""FOMC Communication / Trillion Dollar Words (huggingface.co/datasets/gtfintechlab/fomc_communication,
CC BY-NC 4.0: internal benchmarking only).

2,480 sentences from Federal Reserve statements, minutes, speeches and press conferences. Negative
control (spec §0.14): no sentence describes a commercial bank doing business -> every item expects
not G-SIB activity and market context only.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .common import CTX, GSIB, Builder, hash_files, hf_fetch

FILES = ["train.csv", "test.csv"]
STANCE = {0: "dovish", 1: "hawkish", 2: "neutral"}


def fetch(ds) -> Path:
    return hf_fetch(ds.repository, ds.revision, FILES, ds.local_dir)


def files_sha256(local: Path) -> str:
    return hash_files(local, [local / f for f in FILES])


def ingest(local: Path, ds):
    b = Builder("fomc", "FOMC")
    for f in FILES:
        for r in pl.read_csv(local / f).iter_rows(named=True):
            b.add(r["sentence"], date=str(r.get("year") or ""), source_id=str(r.get("index")),
                  split=f.split(".")[0], label=STANCE.get(r["label"], str(r["label"])), group="fomc",
                  expect={GSIB: 0, CTX: 1})
    return b.frames()
