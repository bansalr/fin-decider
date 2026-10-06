import polars as pl

from classifier.dataset import article_id
from classifier.dedupe import corpus_manifest_hash, dedupe
from classifier.provenance import sha256_text
from classifier.textnorm import normalize

BASE = ("The bank arranged a five billion dollar syndicated loan for the company to refinance existing debt "
        "and fund general corporate purposes according to people familiar with the matter who asked not to be "
        "identified because the talks are private and the terms may still change before signing next week")


def rows(items):
    out = []
    for i, (h, d, b) in enumerate(items):
        out.append({
            "row_index": i, "article_id": article_id(h, d, b), "source": "Bloomberg", "headline": h, "article": b,
            "date": d, "url": f"u{i}", "journalists": [], "source_hash": "x",
            "content_hash": sha256_text(f"{normalize(h)}\n{normalize(b)}"),
        })
    return pl.DataFrame(rows_ := out) if (rows_ := out) else None


def items():
    return [
        ("Loan deal", "2012-01-02T00:00:00", BASE),                       # 0 canonical of cluster A
        ("Loan deal", "2012-01-03T00:00:00", BASE),                       # 1 exact dup of 0 (later date)
        ("Loan deal", "2012-01-01T00:00:00", BASE + " today"),            # 2 near dup, earliest date -> canonical
        ("Other news", "2012-01-02T00:00:00", "Completely different story about weather in Paris " * 3),
        ("", "2012-01-02T00:00:00", ""),                                   # empty, dropped
    ]


def near_params(cfg):
    return cfg.corpus.dedupe.model_copy(update={"near": "minhash_lsh"})


def test_near_dedupe_clusters_and_canonical(cfg):
    df = rows(items())
    canonical, clusters, stats = dedupe(df, near_params(cfg), cfg.run.seed, workers=1)
    assert stats["dropped_empty"] == 1
    assert stats["canonical_articles"] == 2
    loan = canonical.filter(pl.col("headline") == "Loan deal").row(0, named=True)
    assert loan["date"] == "2012-01-01T00:00:00" and loan["cluster_size"] == 3
    reasons = dict(zip(clusters["row_index"].to_list(), clusters["dup_reason"].to_list()))
    assert reasons[2] == "self" and reasons[3] == "self"
    assert reasons[0] == "near" and reasons[1] == "near"   # relative to canonical row 2
    assert 4 not in reasons


def test_dedupe_deterministic_under_row_order(cfg):
    p = near_params(cfg)
    a = dedupe(rows(items()), p, cfg.run.seed, workers=1)[0]
    b = dedupe(rows(list(reversed(items()))), p, cfg.run.seed, workers=1)[0]
    assert corpus_manifest_hash(a, p, "norm_v1", "d") == corpus_manifest_hash(b, p, "norm_v1", "d")
    assert a["article_id"].to_list() == b["article_id"].to_list()


def test_exact_only_keeps_near_duplicates(cfg):
    params = cfg.corpus.dedupe.model_copy(update={"near": "none"})
    canonical, clusters, stats = dedupe(rows(items()), params, cfg.run.seed, workers=1)
    assert stats["near_pairs"] == 0
    assert stats["canonical_articles"] == 3          # row 1 (exact copy of 0) removed; near dup 2 kept
    reasons = dict(zip(clusters["row_index"].to_list(), clusters["dup_reason"].to_list()))
    assert reasons[0] == "self" and reasons[1] == "exact" and reasons[2] == "self"
