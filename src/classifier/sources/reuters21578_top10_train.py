"""Reuters-21578 R(10), ModApte TRAINING split (6,490 stories): the development set for the native-task
benchmark's question wording. Questions may be revised against this split, never against the test split.
"""

from __future__ import annotations

from pathlib import Path

from .reuters21578_top10 import TOP10, fetch, files_sha256, ingest_split  # noqa: F401  (same archive)


def ingest(local: Path, ds):
    return ingest_split(local, "TRAIN")
