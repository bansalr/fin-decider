"""Corpus sources other than the primary Bloomberg dataset (huggingface provider).

Each module: fetch(ds) -> Path, files_sha256(local) -> str, ingest(local, ds) -> (articles, provenance).
The pipeline verifies files_sha256 against the config's `file_sha256` before ingesting.
"""

from importlib import import_module

PROVIDERS = {
    "github_fincausal": "fincausal",
    "hf_twitter_topic": "twitter_topic",
    "hf_fomc": "fomc",
    "hf_cfpb_archive": "cfpb",
    "hf_edgar_gsib": "edgar_gsib",
    "uci_reuters21578": "reuters21578",
    "uci_reuters21578_top10": "reuters21578_top10",
    "uci_reuters21578_top10_train": "reuters21578_top10_train",
    "hf_finmteb": "finmteb",
    "hf_finben_gated": "finben",
}


def source_for(provider: str):
    if provider not in PROVIDERS:
        raise ValueError(f"no source module for provider {provider!r}")
    return import_module(f"{__name__}.{PROVIDERS[provider]}")
