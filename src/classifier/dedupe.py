"""Deterministic corpus deduplication (dedupe_v1).

1. Drop rows whose normalized headline and body are both empty.
2. Exact pass: group by content_hash (normalized headline + body).
3. Near pass: MinHash LSH over lowercased body word shingles among exact-group
   representatives; candidate pairs are verified against the configured Jaccard
   estimate, then merged with union-find.
4. Canonical member per cluster: earliest date, then smallest article_id.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .config import Dedupe
from .provenance import sha256_obj
from .textnorm import normalize

MIN_SHINGLES = 10
MINHASH_SCHEME = "affine32"  # explicit: datasketch 2.x requires it and defaults may change


def shingles(body: str, k: int) -> list[bytes]:
    words = normalize(body).lower().split()
    if len(words) < k:
        return []
    return list({" ".join(words[i : i + k]).encode("utf-8") for i in range(len(words) - k + 1)})


def _signatures(args: tuple[list[str], int, int, int]) -> list[np.ndarray | None]:
    from datasketch import MinHash

    bodies, k, num_perm, seed = args
    out: list[np.ndarray | None] = []
    for body in bodies:
        sh = shingles(body, k)
        if len(sh) < MIN_SHINGLES:
            out.append(None)
            continue
        m = MinHash(num_perm=num_perm, seed=seed, scheme=MINHASH_SCHEME)
        m.update_batch(sh)
        out.append(m.hashvalues.copy())
    return out


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Smaller index wins so the structure is order-independent in outcome.
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(
    bodies: list[str], params: Dedupe, seed: int, workers: int = 8, chunk: int = 2000
) -> list[tuple[int, int]]:
    from datasketch import LeanMinHash, MinHashLSH

    jobs = [(bodies[i : i + chunk], params.shingle_words, params.num_perm, seed) for i in range(0, len(bodies), chunk)]
    sigs: list[np.ndarray | None] = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(_signatures, jobs):
                sigs.extend(part)
    else:
        for job in jobs:
            sigs.extend(_signatures(job))

    lsh = MinHashLSH(threshold=params.jaccard_threshold, num_perm=params.num_perm)
    lean: dict[int, LeanMinHash] = {}
    for i, hv in enumerate(sigs):
        if hv is None:
            continue
        lean[i] = LeanMinHash(seed=seed, hashvalues=hv, scheme=MINHASH_SCHEME)
        lsh.insert(i, lean[i], check_duplication=False)

    pairs = set()
    for i, m in lean.items():
        for j in lsh.query(m):
            if j <= i:
                continue
            # Verify: the LSH threshold is approximate, the signature estimate is the rule.
            if float(np.mean(sigs[i] == sigs[j])) >= params.jaccard_threshold:
                pairs.add((i, j))
    return sorted(pairs)


def dedupe(df: pl.DataFrame, params: Dedupe, seed: int, workers: int = 8) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Return (canonical, clusters, stats). `df` is the output of dataset.ingest()."""
    total = df.height
    nonempty = df.filter(
        (pl.col("headline").map_elements(normalize, return_dtype=pl.String) != "")
        | (pl.col("article").map_elements(normalize, return_dtype=pl.String) != "")
    )
    dropped_empty = total - nonempty.height

    # Deterministic order for everything downstream.
    nonempty = nonempty.sort(["date", "article_id", "row_index"])

    # Exact groups: representative = first in (date, article_id) order.
    reps = nonempty.unique(subset=["content_hash"], keep="first", maintain_order=True)
    rep_hashes = reps["content_hash"].to_list()
    rep_index = {h: i for i, h in enumerate(rep_hashes)}

    uf = UnionFind(len(rep_hashes))
    near_pairs: list[tuple[int, int]] = []
    if params.near == "minhash_lsh":
        near_pairs = near_duplicate_pairs(reps["article"].to_list(), params, seed, workers=workers)
        for a, b in near_pairs:
            uf.union(a, b)

    rep_cluster = [uf.find(i) for i in range(len(rep_hashes))]

    members = nonempty.with_columns(
        pl.col("content_hash")
        .replace_strict({h: rep_cluster[i] for h, i in rep_index.items()}, return_dtype=pl.Int64)
        .alias("_cluster")
    )

    # Canonical member per cluster: earliest date, then smallest article_id (members is sorted that way).
    canon = members.group_by("_cluster", maintain_order=True).agg(
        pl.col("article_id").first().alias("canonical_article_id"),
        pl.col("row_index").first().alias("_canon_row"),
        pl.col("content_hash").first().alias("_canon_hash"),
        pl.len().alias("cluster_size"),
    )
    members = members.join(canon, on="_cluster", how="left").with_columns(
        pl.col("canonical_article_id").str.replace("^blbg_", "clu_").alias("cluster_id"),
        pl.when(pl.col("row_index") == pl.col("_canon_row"))
        .then(pl.lit("self"))
        .when(pl.col("content_hash") == pl.col("_canon_hash"))
        .then(pl.lit("exact"))
        .otherwise(pl.lit("near"))
        .alias("dup_reason"),
    )

    clusters = members.select(
        "row_index", "article_id", "cluster_id", "canonical_article_id", "dup_reason", "content_hash"
    ).sort("row_index")

    canonical = (
        members.filter(pl.col("dup_reason") == "self")
        .select(
            "article_id", "source", "headline", "article", "date", "url", "journalists",
            "source_hash", "content_hash", "row_index", "cluster_id", "cluster_size",
        )
        .sort("article_id")
    )

    sizes = canonical["cluster_size"]
    stats = {
        "source_rows": total,
        "dropped_empty": dropped_empty,
        "exact_groups": len(rep_hashes),
        "exact_duplicates_removed": nonempty.height - len(rep_hashes),
        "near_pairs": len(near_pairs),
        "near_duplicates_removed": len(rep_hashes) - canonical.height,
        "canonical_articles": canonical.height,
        "max_cluster_size": int(sizes.max()) if canonical.height else 0,
        "clusters_size_gt1": int((sizes > 1).sum()),
    }
    return canonical, clusters, stats


def corpus_manifest_hash(canonical: pl.DataFrame, params: Dedupe, normalization_version: str, dataset_sha256: str) -> str:
    return sha256_obj(
        {
            "canonical_ids": canonical["article_id"].sort().to_list(),
            "dedupe": params.model_dump(),
            "normalization": normalization_version,
            "minhash_scheme": MINHASH_SCHEME,
            "dataset_sha256": dataset_sha256,
        }
    )


def write_corpus(
    out_dir: Path, canonical: pl.DataFrame, clusters: pl.DataFrame, stats: dict, params: Dedupe,
    normalization_version: str, dataset_sha256: str, seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical.write_parquet(out_dir / "canonical.parquet")
    clusters.write_parquet(out_dir / "clusters.parquet")
    manifest = {
        "corpus_manifest_hash": corpus_manifest_hash(canonical, params, normalization_version, dataset_sha256),
        "dataset_sha256": dataset_sha256,
        "normalization_version": normalization_version,
        "dedupe": params.model_dump(),
        "minhash_seed": seed,
        "minhash_scheme": MINHASH_SCHEME,
        "min_shingles_for_near": MIN_SHINGLES,
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_corpus(out_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return pl.read_parquet(out_dir / "canonical.parquet"), manifest
