"""Strict configuration (spec §20, §22).

Every scientifically material parameter lives in the YAML; nothing material is
defaulted in code. Unknown keys fail.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
UNPINNED_ALIASES = {"latest", "jev-latest", "main", "head"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _sha(v: str) -> str:
    if not SHA256_RE.match(v):
        raise ValueError(f"expected a 64-hex SHA-256, got {v!r}")
    return v


Sha256 = Annotated[str, AfterValidator(_sha)]


class Benchmark(Strict):
    name: str
    spec_version: str


class Run(Strict):
    label: str
    seed: int
    allow_dirty_git: bool


class Dataset(Strict):
    provider: Literal["huggingface"]
    repository: str
    revision: str
    file: str
    file_sha256: Sha256
    local_dir: str

    @field_validator("revision")
    @classmethod
    def _pinned(cls, v: str) -> str:
        if not GIT_SHA_RE.match(v):
            raise ValueError(f"dataset revision must be a 40-hex commit SHA, got {v!r}")
        return v


class Dedupe(Strict):
    version: str
    exact: Literal["sha256_normalized_headline_body"]
    near: Literal["minhash_lsh", "none"]
    num_perm: int = Field(ge=16, le=1024)
    shingle_words: int = Field(ge=1, le=20)
    jaccard_threshold: float = Field(gt=0, le=1)
    canonical_rule: Literal["earliest_date_then_article_id"]


class Corpus(Strict):
    directory: str
    normalization_version: str
    dedupe: Dedupe


class OntologyCfg(Strict):
    path: str
    version: str
    sha256: Sha256


class Normalization(Strict):
    unicode: Literal["NFC", "NFKC"]
    collapse_whitespace: bool
    strip_html: bool


class Input(Strict):
    mode: Literal["headline_plus_article"]
    max_chars: int = Field(gt=0)
    separator: str
    truncation_strategy: Literal["head"]
    normalization: Normalization


class Traversal(Strict):
    mode: Literal["breadth_first"]
    threshold: float = Field(ge=0, le=1)
    max_depth: int = Field(ge=1)
    max_nodes_per_article: int = Field(ge=1)


class Classification(Strict):
    positive_threshold: float = Field(ge=0, le=1)


class Hierarchy(Strict):
    traversal: Traversal
    classification: Classification
    materialize_skipped_by_routing: bool


class Template(Strict):
    kind: Literal["boolean", "label"]
    path: str
    sha256: Sha256


class ModelCfg(Strict):
    enabled: bool
    adapter: Literal["jev", "laya", "d1", "clef", "systemone", "fake"]
    provider: Literal["vercel_ai_gateway", "cloudflare_workers_ai", "local_server", "local"]
    endpoint: str | None
    model: str
    provider_lock: list[str]
    api_key_env: str | None
    template: str
    timeout_seconds: float = Field(gt=0)
    retries: int = Field(ge=0, le=20)
    concurrency: int = Field(ge=1, le=512)
    allow_returned_model_mismatch: bool
    # Model-specific technical input limit (spec §7): cap the article before sending,
    # and on HTTP 413/422 retry with 20% less text (up to 5 times). Both are material.
    max_input_chars: int | None = Field(default=None, gt=0)
    shrink_on_422: bool = False
    # Cloudflare Workers AI only
    account_id_env: str | None = None
    price_per_m_input_tokens: float | None = Field(default=None, ge=0)  # cost estimate; provider reports none
    expected_model_returned: str | None = None  # pinned from the first live probe / server config
    # Self-hosted open-weight servers (local_server) only
    revisions: dict[str, str] | None = None  # HF repo -> 40-hex commit for every weight file served
    quantization: Literal["none", "bnb8"] | None = None
    request_model: str | None = None  # value sent as `model` in the request body

    @field_validator("model")
    @classmethod
    def _not_latest(cls, v: str) -> str:
        tail = v.split("/")[-1].lower()
        if tail in UNPINNED_ALIASES or tail.endswith("-latest"):
            raise ValueError(f"model {v!r} is an unpinned 'latest' alias")
        return v

    @model_validator(mode="after")
    def _provider_fields(self) -> "ModelCfg":
        if self.provider in ("vercel_ai_gateway", "cloudflare_workers_ai") and (not self.endpoint or not self.api_key_env):
            raise ValueError(f"{self.provider} models need endpoint and api_key_env")
        if self.provider == "cloudflare_workers_ai" and not self.account_id_env:
            raise ValueError("cloudflare_workers_ai models need account_id_env")
        if self.account_id_env and self.provider != "cloudflare_workers_ai":
            raise ValueError("account_id_env is only valid for cloudflare_workers_ai")
        if self.expected_model_returned and self.provider not in ("cloudflare_workers_ai", "local_server"):
            raise ValueError("expected_model_returned is only valid for cloudflare_workers_ai and local_server")
        local = (self.revisions, self.quantization, self.request_model)
        if self.provider == "local_server":
            if not self.endpoint or not self.revisions or not self.quantization or not self.expected_model_returned:
                raise ValueError("local_server models need endpoint, revisions, quantization, expected_model_returned")
            bad = {k: v for k, v in self.revisions.items() if not GIT_SHA_RE.match(v)}
            if bad:
                raise ValueError(f"revisions must be 40-hex commit SHAs: {bad}")
        elif any(x is not None for x in local):
            raise ValueError("revisions/quantization/request_model are only valid for local_server")
        return self


class Output(Strict):
    directory: str
    format: Literal["parquet"]
    shard_rows: int = Field(ge=100)
    include_raw_provider_response: bool


class Config(Strict):
    benchmark: Benchmark
    run: Run
    dataset: Dataset
    corpus: Corpus
    ontology: OntologyCfg
    input: Input
    hierarchy: Hierarchy
    templates: dict[str, Template]
    models: dict[str, ModelCfg]
    output: Output

    @model_validator(mode="after")
    def _templates_exist(self) -> "Config":
        for name, m in self.models.items():
            if m.template not in self.templates:
                raise ValueError(f"model {name!r} references unknown template {m.template!r}")
        return self


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply `dotted.key=value` overrides; values are parsed as YAML scalars.
    Overrides go through the same strict validation as the file, and material
    ones change the run ID exactly as editing the file would."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like a.b.c=value")
        parts = key.split(".")
        cur = data
        for p in parts[:-1]:
            if not isinstance(cur.get(p), dict):
                raise ValueError(f"override {item!r}: no section {p!r}")
            cur = cur[p]
        if parts[-1] not in cur:
            raise ValueError(f"override {item!r}: unknown key {parts[-1]!r}")
        cur[parts[-1]] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(apply_overrides(data, overrides or []))


def check_credentials(cfg: Config, models: list[str]) -> None:
    missing = [
        env
        for m in models
        for env in (cfg.models[m].api_key_env, cfg.models[m].account_id_env)
        if env and not os.environ.get(env)
    ]
    if missing:
        raise RuntimeError(f"missing credentials in environment: {sorted(set(missing))}")
