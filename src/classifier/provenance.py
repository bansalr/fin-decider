"""Hashing, git/environment capture, and run identity (spec §23, §30)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable serialization: sorted keys, no whitespace, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_info(repo: str | Path = ".") -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": True}
    return {"commit": commit, "dirty": dirty}


def environment_info() -> dict[str, Any]:
    libs = {}
    for name in ("pydantic", "httpx", "pyarrow", "polars", "datasketch", "huggingface-hub"):
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libs[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": libs,
    }


def make_run_id(
    *,
    resolved_config: dict[str, Any],
    ontology_hash: str,
    dataset_hash: str,
    corpus_manifest_hash: str,
    template_hashes: dict[str, str],
    git_commit: str | None,
    now: datetime | None = None,
) -> tuple[str, str]:
    """Return (run_id, full_hash). The hash is authoritative; the timestamp is cosmetic.

    The timestamp is NOT part of the hash, so the same inputs always map to the
    same hash and a resume can locate its run directory.
    """
    full = sha256_obj(
        {
            "config": resolved_config,
            "ontology": ontology_hash,
            "dataset": dataset_hash,
            "corpus_manifest": corpus_manifest_hash,
            "templates": template_hashes,
            "git_commit": git_commit,
        }
    )
    return full[:16], full


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
