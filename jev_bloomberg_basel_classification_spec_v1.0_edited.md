# Jev × Bloomberg × Basel Hierarchical Classification Benchmark

## Implementation Specification v1.0 (with v1.1 amendments)

## 0. v1.1 amendments

These amendments take precedence over the v1.0 text below where they
conflict. They record decisions made after v1.0 and the deviations the
implementation (`src/classifier/`, `configs/classification.yaml`) makes.

### 0.1 Model panel

Anchor models: Jev (`typesafe-ai/jev`) and Laya
(`convaiinnovations/laya`).

Comparators in v1:

-   Liquid d1 (`liquid/d1`): a third hosted decision model on the same
    gateway API, receiving byte-identical requests.
-   GLiClass (`knowledgator/gliclass-large-v3.0`, revision
    `e065d1844f913a9aa611cf33623a9538b8aa8841`): an open-weight
    zero-shot encoder run locally (see §0.9).

Deferred to v2: an NLI cross-encoder (DeBERTa-v3-large zeroshot v2.0),
which GLiClass outperforms on published zero-shot benchmarks at about
4x the throughput; hosted rerankers, whose per-label re-submission of
the article makes full-corpus cost disproportionate; and embedding
similarity, which has no a-priori decision threshold.

### 0.2 Provider: Vercel AI Gateway

Both anchors are invoked through Vercel AI Gateway
(`POST https://ai-gateway.vercel.sh/v1/evaluate`), authenticated by
`VERCEL_API_KEY`. This replaces `provider: typesafe` and
`TYPESAFE_API_KEY` in §20/§21.

Fairness (§18) is enforced structurally: both models receive
byte-identical requests (same state text, same question keys, same
rendered template). Only `model` and the provider lock differ. A single
template (`templates/decision/boolean_v1.txt`) is shared. Each
`multi_label` node is one request with one independent `boolean`
question per child.

Context limits reported by the gateway catalog: Jev 32,000 tokens (state +
longest question), Laya 8,192 tokens. The 12,000-character input cap fits
both, so no model-specific truncation is applied.

### 0.3 Version pinning deviation (§17)

The gateway exposes no dated or pinned version identifiers for either
model. In place of a pinned version, the run enforces and records:

-   an exact model ID in configuration (`latest` aliases are rejected);
-   a provider lock (`providerOptions.gateway.only`);
-   a per-response check that the returned `model` equals the requested
    model and that the gateway's `finalProvider` is in the lock (fatal
    otherwise);
-   a hash of the gateway catalog entry (id, type, context window,
    release date, owner) at run start; a resume fails if it has changed;
-   the gateway `generationId` of every call.

Gateway decision fallbacks are never configured; a response indicating
a fallback is fatal.

### 0.4 Full deduplicated corpus replaces sampling (§5, §8, §23, §29)

The primary run classifies every canonical article of the full
deduplicated Bloomberg corpus. There is no benchmark sample.

Deduplication (`dedupe_v2_exact`): drop rows with empty normalized
headline and body; remove exact duplicates only, defined as identical
SHA-256 of normalized headline + `\n` + normalized body (publish date is
ignored, so a word-for-word republication on another day is a
duplicate). The canonical member of each duplicate group is the
earliest-dated article, ties broken by smallest `article_id`.
`corpus/clusters.parquet` retains every source row's group and
duplicate reason.

Near-duplicate merging is deliberately not applied. A MinHash LSH pass
(`near: minhash_lsh`, Jaccard ≥ 0.90) is implemented but disabled:
on this corpus it merged recurring templated reports (daily price
tables, money-market tables, earnings lists) that differ in date and
figures and are distinct articles.

`sample_manifest_hash` is replaced throughout by `corpus_manifest_hash`:
SHA-256 over the sorted canonical article IDs, dedupe parameters,
normalization version, MinHash scheme, and dataset file hash. The run
manifest records `corpus` instead of `sample`.

Runs restricted with `--limit` are engineering runs: the scope is part
of the run identity and recorded in the manifest.

### 0.5 Input mode (§7)

The single canonical mode is `headline_plus_article`: normalized
headline, separator, normalized body, head-truncated to `max_chars`.

### 0.6 Run identity materiality

Each model has its own run identity. A run's hash covers the shared
sections (benchmark, dataset, corpus, ontology, input, hierarchy), that
model's material settings, its template, the corpus manifest hash, the
ontology file hash, the git commit, and the scope (full corpus or
`--limit N`). Other models' settings never affect a run's identity.
Run directories are named `<model>-<hash>`.

Operational parameters that cannot change a classification are
excluded: `concurrency`, `retries`, `timeout_seconds`, `device`,
`batch_size`, `batch_wait_ms`, `run.label`, `run.allow_dirty_git`,
`output.directory`, `output.shard_rows`. Model `revision`, `dtype` and
`max_length` are material.

Every manifest records `shared_inputs_sha256` (shared sections,
ontology, corpus, scope). Runs of different models with equal
`shared_inputs_sha256` classified the same articles under the same
ontology and thresholds and may be compared directly.

Engineering runs may set `allow_dirty_git: true`; official runs require
a clean tree. Configuration may be overridden on the command line
(`--set a.b=value`, `--output-dir`); overrides pass the same strict
validation and affect identity exactly as editing the file would.

### 0.7 Result rows

Rows carry, in addition to §25: `prediction_key`, `attempt`,
`attempts`, `reused_from_cache`, `model_returned`, `routing_provider`,
`generation_id`. Per-call fields (latency, tokens, cost, generation ID)
repeat on every child row of the same call; aggregate them by
`prediction_key`. Unvisited nodes are omitted (`SKIPPED_BY_ROUTING` is
not materialized). Work shards live under `_work/`; the final
`results/model=<m>/` keeps the latest attempt per article.

### 0.8 Ontology v1.1 and the shared decision template

`gsib_basel_ontology_v1.1.json` keeps v1.0's node IDs, hierarchy,
names and types unchanged, and adds to every non-root node a plain
`yes_no_question` and `criteria` (`true`/`false`), and further
`excludes` for known hard boundaries. It is generated from
`ontology/v1.1_additions.yaml` by `ontology/build_v1_1.py`; v1.0 is not
modified.

Motivation: live probes showed decision models differ sharply in
sensitivity to question form. With v1.0's fields rendered as a long
definition-plus-criteria block, Laya's scores were near-flat (about
0.35 on every relevance child), while a concrete yes/no question
separated the same cases (0.91 vs 0.001).

The shared decision template is chosen on a synthetic development set
(`dev/template_cases_v1.jsonl`; short constructed articles, none drawn
from the Bloomberg corpus) by a pre-declared rule: the candidate with
the highest minimum balanced accuracy at 0.5 across Jev, Laya and d1,
ties broken by fewer tokens. Candidates: `boolean_v1`, `boolean_v2a`
(question + criteria), `boolean_v2b` (+ exclusions line),
`boolean_v2c` (question only). The selected template is frozen before
any corpus run. Tooling: `scripts/template_check.py`.

Observed on the draft ontology: Jev and d1 scored 0.95–0.99 on every
candidate; Laya 0.65–0.73, mostly through false positives on sibling
categories and on negatively phrased questions. Jev returned slightly
different probabilities for identical requests across repeated runs;
Jev results are therefore reproducible in distribution, not bit for
bit.

### 0.9 GLiClass representation

GLiClass is a uni-encoder (DeBERTa-v3-large) whose labels and article
share one 512-token window, labels first. Ontology v1.1 adds a short
plain-language `label` (at most about 8 words) to every node for this
purpose. The label template is chosen on the same synthetic dev set by
the highest balanced accuracy at 0.5 (`scripts/label_check.py`):

| template | label text | balanced accuracy |
|---|---|---|
| `label_v1` | `{name}: {definition}` | 0.52 |
| `label_name` | `{name}` | 0.67 |
| `label_v2` | `{label}` | 0.73 |

`label_v1`, the format originally planned, was near chance: the model
scored the "Neither / Out of Scope" child at about 0.99 for every
article. `label_v2` is selected. GLiClass receives less ontology
information than the decision models (no definitions, criteria or
exclusions), never more (§18). The article is truncated from the end to
fit after the labels; each call records `model_submitted_tokens` and
`model_truncated`.

Scores are the model's independent sigmoid outputs for every label (no
pipeline threshold); traversal applies the shared thresholds. The full
corpus runs on a Colab T4 in `float16`; engineering runs on Apple MPS
use `float32`. dtype is material, so these are separate run identities.

### 0.10 Intermediate run and freeze rule

Before full-corpus execution, every model runs on the first 1,000
canonical articles by `article_id` (content hashes, so effectively a
deterministic random sample). The ontology, templates and thresholds
are frozen before that run. Afterwards only engineering changes are
made (bugs, rate limits, concurrency). Any semantic change prompted by
the intermediate run is recorded as informed by that sample and yields
new run identities.

## 1. Purpose

This specification defines a reproducible, model-neutral pipeline for
classifying Bloomberg Financial News articles into a Basel-derived G-SIB
Markets and Lending ontology.

The principal system under test is TypeSafe Jev. Comparator systems are
zero-shot, non-generative classifiers that require no fitting,
fine-tuning, calibration, or training on the Bloomberg benchmark.

The classification pipeline is deliberately separate from human
adjudication and evaluation.

The classification pipeline performs only:

> Article + ontology + model configuration → hierarchical classification
> results

It MUST NOT read or write human judgments, disagreement queues, gold
labels, evaluation metrics, reports, or benchmark winner/loser
determinations.

A separate adjudication/evaluation system consumes completed
classification results.

------------------------------------------------------------------------

## 2. Architectural boundaries

### 2.1 Classification pipeline

Inputs:

-   immutable Bloomberg dataset snapshot;
-   deterministic article sample;
-   versioned ontology;
-   versioned classification templates;
-   pinned model configurations;
-   classification configuration;
-   secrets supplied through environment variables.

Outputs:

-   classification result records;
-   a minimal run manifest sufficient to identify the inputs and
    configuration that generated those records.

The pipeline MUST NOT write:

-   disagreement files;
-   adjudication queues;
-   human labels;
-   gold labels;
-   metrics;
-   comparison reports;
-   charts;
-   model rankings.

### 2.2 Adjudication/evaluation pipeline

This is a separate downstream pipeline and is not implemented as part of
the classifier runtime.

It may consume classification outputs from Jev and benchmark models,
identify disagreements, sample agreements for audit, collect blinded
human judgments, and compute metrics.

There MUST be no dependency from classification code to adjudication
code.

The dependency direction is:

    dataset → classifier → classification results → evaluator/adjudicator

and never the reverse.

------------------------------------------------------------------------

## 3. Classification problem

The benchmark is hierarchical and multi-label.

An article can traverse multiple branches of the ontology. For example,
an acquisition financing story may simultaneously map to:

    Lending
      → Leveraged / Sponsor Finance
        → Acquisition Finance
        → Bridge Finance

and:

    Markets
      → Underwriting / Syndication
        → Debt Capital Markets
        → Loan Syndication

The classifier MUST therefore support multiple selected children at
every multi-label node.

The ontology, not Python source code, defines the inference graph.

------------------------------------------------------------------------

## 4. Classification semantics

The target is G-SIB business activity, not generic financial-news topic
classification.

The question is:

> Which G-SIB Markets, Lending, or related business activities are
> materially implicated or described by this article?

A mention of an instrument does not automatically imply the
corresponding banking activity.

Examples:

-   "Apple shares rose 5% after earnings" is equity-market context, but
    does not by itself establish a G-SIB equities business activity.
-   "A dealer structured an equity derivative for institutional
    investors tied to Apple shares" does establish a Markets activity.
-   "The Federal Reserve changed its repo rate" must not automatically
    be classified as dealer repo financing.
-   "Boeing received new aircraft orders" must not automatically be
    classified as Object Finance.
-   "Banks arranged a non-recourse facility for aircraft purchases" may
    be Object Finance.

The ontology definitions, inclusions, exclusions, and questions are
authoritative.

------------------------------------------------------------------------

## 5. Dataset

Primary dataset:

    danidanou/Bloomberg_Financial_News

The implementation MUST pin an immutable dataset revision and store its
content hash.

Canonical internal article schema:

``` json
{
  "article_id": "blbg_<sha256>",
  "source": "Bloomberg",
  "headline": "...",
  "article": "...",
  "date": "YYYY-MM-DD",
  "url": "...",
  "journalists": ["..."],
  "source_hash": "sha256:..."
}
```

`article_id` MUST be deterministic and MUST NOT depend on dataframe row
number.

Recommended hash material:

    normalize(headline) + "\n" + normalize(date) + "\n" + normalize(article)

Normalization MUST be versioned.

------------------------------------------------------------------------

## 6. Dataset provenance

Required configuration:

``` yaml
dataset:
  provider: huggingface
  repository: danidanou/Bloomberg_Financial_News
  revision: "<PINNED_COMMIT_SHA>"
  file: "<DATA_FILE>"
  file_sha256: "<SHA256>"
```

Execution MUST fail if the downloaded/local file hash does not match
`file_sha256`.

------------------------------------------------------------------------


## 7. Input construction

Supported input modes:

-   `headline` + `full_article`
-   `headline` + `summary`

Default:

``` yaml
input:
  mode: headline_plus_lead
  max_chars: 12000
  separator: "\n\n"
  normalization:
    unicode: NFC
    collapse_whitespace: true
    strip_html: true
```

Model adapters MAY tokenize differently, but they MUST receive
semantically equivalent article content subject to their technical
context limits.

------------------------------------------------------------------------

## 9. Ontology

The ontology is a separate versioned YAML artifact.

Recommended path:

    ontology/gsib_basel_v1.0.yaml

The ontology MUST define:

-   ontology ID and semantic version;
-   root nodes;
-   node IDs;
-   canonical names;
-   node type;
-   question;
-   definition;
-   inclusions;
-   exclusions;
-   synonyms;
-   parent/child relationships;
-   Basel metadata where applicable;
-   routing behavior where node-specific overrides are required.

The ontology MUST be validated before any inference call.

No hierarchy routing may be hard-coded into model adapters.

------------------------------------------------------------------------

## 10. Node types

Supported node types:

### `multi_label`

Zero, one, or several children may be selected.

### `single_choice`

Exactly one child is selected, unless an explicit `none` child exists.

### `leaf`

No classifier call is made below this node.

The initial ontology is predominantly `multi_label`.

------------------------------------------------------------------------

## 11. Hierarchical inference

Classification proceeds through multiple calls.

Conceptual flow:

    article
       ↓
    root/relevance
       ↓
    business arm(s)
       ↓
    family node(s)
       ↓
    sub-family node(s)
       ↓
    leaves

Traversal MUST allow multiple branches.

Generic algorithm:

``` python
queue = [root_nodes]

while queue:
    node = queue.pop()

    if node.type == "leaf":
        record_leaf(node)
        continue

    decision = classifier.classify(article, node)
    record_decision(decision)

    for child in decision.children:
        if should_traverse(child):
            queue.push(child)
```

Breadth-first traversal is the default because it makes per-level
behavior easier to inspect, but traversal order MUST NOT change
classification semantics.

------------------------------------------------------------------------

## 12. Model adapter contract

Every classifier implements the same logical interface:

``` python
class ClassificationAdapter:
    def classify(
        self,
        article: ArticleInput,
        node: OntologyNode,
        context: ClassificationContext
    ) -> NodeDecision:
        ...
```

`NodeDecision` contains:

``` json
{
  "node_id": "markets.financing",
  "scores": {
    "markets.financing.repo": 0.93,
    "markets.financing.securities_lending": 0.08,
    "markets.financing.prime": 0.62
  },

}
```

Model-specific raw response formats MUST be normalized inside the
adapter.

------------------------------------------------------------------------

## 13. Hierarchical routing thresholds

Routing and final classification thresholds are separate parameters.

Default:

``` yaml
hierarchy:
  traversal:
    mode: breadth_first
    threshold: 0.30

  classification:
    positive_threshold: 0.50
```

Rationale:

A parent may be uncertain while a child, once queried, is highly
discriminative. A lower traversal threshold prevents premature pruning.

For a child with:

    parent score = 0.42
    child score = 0.91

the child can still be explored if traversal threshold is 0.30.

A leaf is considered positive only according to the configured final
threshold, unless the ontology node specifies an explicit override.

Thresholds MUST NOT be changed using human judgments from the same
evaluation run.

------------------------------------------------------------------------

## 14. Path probabilities

For every traversed path, preserve node-local scores.

The implementation MAY additionally calculate a derived path product:

    P_path = ∏ node_score

For example:

    Markets = 0.88
    Financing | Markets = 0.72
    Repo | Financing = 0.91

gives:

    P_path = 0.576576

This is a derived diagnostic only. It MUST NOT be treated as a
calibrated probability unless separately validated.

Both local scores and derived path scores MUST remain distinguishable.

------------------------------------------------------------------------

## 15. Relevance routing

The ontology begins with a relevance decision to distinguish:

-   G-SIB activity;
-   market context only;
-   neither.

`market_context_only` is used for financial-market stories that describe
asset prices, macro developments, issuers, or market conditions without
materially describing a G-SIB Markets/Lending activity.

If `gsib_activity` is selected, the article proceeds to the business
hierarchy.

The ontology MAY permit both `gsib_activity` and `market_context_only`
when a story contains both bank activity and broader market context.

------------------------------------------------------------------------

## 16. Model suite

The initial suite is expected to include:

-   Jev;
-   GLiClass;
-   an NLI-based zero-shot encoder such as a pinned DeBERTa/BART
    checkpoint;
-   optional sentence/document embedding similarity baseline.

Every model MUST:

-   be publicly executable or accessible under the benchmark's provider
    terms;
-   require no training or fine-tuning on Bloomberg;
-   require no human-label calibration;
-   be non-generative for classification;
-   have a pinned model/version/revision.

Model availability is configuration-driven.

------------------------------------------------------------------------

## 17. Jev adapter

Jev MUST be invoked using a pinned concrete model version, never an
unpinned `latest` alias for a benchmark run.

The adapter SHOULD use Jev's native typed decision mechanisms:

-   `Choice` for mutually exclusive nodes;
-   independent binary decisions (`Noul`) for multi-label nodes.

Multiple questions MAY be batched where the provider interface permits,
but batching MUST NOT alter the logical node boundaries recorded in
output.

The adapter MUST record:

-   requested model version;
-   returned model version when available;
-   node scores;
-   selected children;
-   request latency;
-   provider usage fields needed to interpret the classification.

If the returned model differs from the requested pinned model, the run
MUST fail unless explicitly configured otherwise.

------------------------------------------------------------------------

## 18. Comparator fairness

All systems MUST receive equivalent semantic information.

The ontology provides a common intermediate representation:

``` json
{
  "id": "lending.specialised.project_finance",
  "name": "Project Finance",
  "definition": "...",
  "includes": ["..."],
  "excludes": ["..."],
  "synonyms": ["..."]
}
```

Each adapter may transform this into its native representation, such as:

-   Jev criteria/questions;
-   GLiClass labels/descriptions;
-   NLI hypotheses;
-   embedding descriptions.

No comparator may receive materially richer label definitions than
another in the primary experiment.

------------------------------------------------------------------------

## 19. Template versioning

Model-specific formatting templates are versioned benchmark inputs.

Recommended structure:

    templates/
      jev/
        node_v1.txt
      gliclass/
        node_v1.txt
      nli/
        hypothesis_v1.txt
      embeddings/
        description_v1.txt

Every template MUST have a SHA256 included in run provenance.

Changing wording creates a new run identity.

------------------------------------------------------------------------

## 20. Configuration

Primary file:

    configs/classification.yaml

Reference configuration:

``` yaml
benchmark:
  name: jev_bloomberg_basel
  spec_version: "1.0.0"

run:
  label: baseline
  seed: 917341
  allow_dirty_git: false

dataset:
  provider: huggingface
  repository: danidanou/Bloomberg_Financial_News
  revision: "<PINNED_COMMIT_SHA>"
  file: "<DATA_FILE>"
  file_sha256: "<SHA256>"

ontology:
  path: ontology/gsib_basel_v1.0.yaml
  version: "1.0.0"
  sha256: "<SHA256>"


input:
  mode: headline + article
  max_chars: 12000
  separator: "\n\n"
  normalization:
    unicode: NFC
    collapse_whitespace: true
    strip_html: true

hierarchy:
  traversal:
    mode: breadth_first
    threshold: 0.30
    max_depth: 10
    max_nodes_per_article: 100
  classification:
    positive_threshold: 0.50

models:
  jev:
    enabled: true
    provider: typesafe
    model: "<PINNED_JEV_VERSION>"
    api_key_env: TYPESAFE_API_KEY
    timeout_seconds: 60
    retries: 5
    concurrency: 20
    allow_returned_model_mismatch: false

  gliclass:
    enabled: true
    provider: huggingface
    model: "<PINNED_MODEL_ID>"
    revision: "<PINNED_COMMIT_SHA>"
    device: cuda
    batch_size: 32
    threshold: 0.50

  nli:
    enabled: true
    provider: huggingface
    model: "<PINNED_MODEL_ID>"
    revision: "<PINNED_COMMIT_SHA>"
    device: cuda
    batch_size: 32
    threshold: 0.50

  embeddings:
    enabled: false
    provider: huggingface
    model: "<PINNED_MODEL_ID>"
    revision: "<PINNED_COMMIT_SHA>"
    device: cuda
    batch_size: 64
    threshold: 0.50

output:
  directory: classification_results/
  format: parquet
  include_node_scores: true
  include_latency: true
  include_usage: true
  include_raw_provider_response: false
```

Scientifically material parameters MUST NOT be hidden in source-code
defaults.

------------------------------------------------------------------------

## 21. Secrets

Secrets MUST NOT appear in the classification YAML.

Use environment variables, typically loaded from `.env` during local
development.

`.env.example`:

``` bash
TYPESAFE_API_KEY=
HF_TOKEN=
```

`.gitignore` MUST include:

``` text
.env
.env.*
!.env.example
secrets/
```

Resolved configuration artifacts MUST contain only the environment
variable name, never the secret value.

------------------------------------------------------------------------

## 22. Configuration validation

Use a strict Pydantic model and/or JSON Schema.

Execution MUST fail before inference if:

-   unknown configuration keys exist;
-   required values are missing;
-   a model revision/version is unpinned;
-   ontology hash differs;
-   dataset hash differs;
-   sample seed is absent;
-   thresholds are outside valid ranges;
-   required credentials for an enabled provider are missing;
-   a node references a nonexistent child;
-   ontology cycles exist;
-   a leaf has children;
-   duplicate node IDs exist.

No permissive parsing for benchmark-critical configuration.

------------------------------------------------------------------------

## 23. Run identity and provenance

Every classification run has an immutable `run_id`.

Recommended derivation:

    SHA256(
      canonical_resolved_config
      + ontology_hash
      + dataset_hash
      + sample_manifest_hash
      + template_hashes
      + git_commit
    )

Human-readable representation:

    20261001T221503Z_7f2a4c19

The hash component is authoritative.

Changing any scientifically material input MUST produce a different run
ID.

Examples:

-   dataset revision;
-   sample;
-   ontology wording;
-   ontology topology;
-   model version;
-   threshold;
-   article context mode;
-   template wording;
-   code commit;
-   normalization rules.

------------------------------------------------------------------------

## 24. Classification output boundary

The classification pipeline writes only:

``` text
classification_results/
  <run_id>/
    manifest.json
    results.parquet
```

Optional partitioning of `results.parquet` by model is permitted:

``` text
classification_results/
  <run_id>/
    manifest.json
    results/
      model=jev/*.parquet
      model=gliclass/*.parquet
      model=nli/*.parquet
```

It MUST NOT create adjudication or evaluation artifacts.

------------------------------------------------------------------------

## 25. Result schema

The canonical normalized representation is one row per:

    article × model × ontology node × child

Recommended fields:

``` text
run_id
article_id
model_id
model_version
task_root
node_id
node_depth
child_id
score
selected_positive
selected_for_traversal
local_threshold
traversal_threshold
path_score
original_chars
submitted_chars
truncated
truncation_strategy
latency_ms
input_tokens
output_tokens
provider_cost_usd
config_hash
ontology_hash
dataset_hash
sample_manifest_hash
template_hash
code_commit
status
error_code
created_at
```

`provider_cost_usd` MAY be null if not supplied/calculable at
classification time.

Final leaf classifications MUST be reconstructable from these node-level
rows.

------------------------------------------------------------------------

## 26. Result status

Allowed status values:

-   `SUCCESS`
-   `TIMEOUT`
-   `RATE_LIMITED`
-   `PROVIDER_ERROR`
-   `INVALID_REQUEST`
-   `MODEL_ERROR`
-   `SKIPPED_BY_ROUTING`

A provider failure MUST NOT be converted into a negative classification.

`SKIPPED_BY_ROUTING` MAY be materialized for all unvisited nodes or
omitted; this behavior MUST be fixed by configuration and recorded in
the manifest.

------------------------------------------------------------------------

## 27. Retry behavior

Retries apply only to retryable provider failures.

Configuration:

``` yaml
models:
  jev:
    retries: 5
    timeout_seconds: 60
```

Retry count and final status MUST be represented in the classification
result or manifest-level execution metadata.

Authentication and validation failures MUST fail fast.

------------------------------------------------------------------------

## 28. Idempotency and resume

Classification MUST be resumable.

A logical prediction key is:

    SHA256(
      model_version
      + article_input_hash
      + node_hash
      + ontology_hash
      + model_config_hash
      + template_hash
    )

On `--resume`, successful existing classifications for the same
prediction key MUST be reused.

The pipeline MUST never mix records generated from different prediction
keys under one run identity.

------------------------------------------------------------------------

## 29. Run manifest

`manifest.json` is metadata, not an evaluation artifact.

It MUST contain enough information to reproduce and interpret the result
file.

Example:

``` json
{
  "run_id": "20261001T221503Z_7f2a4c19",
  "benchmark_spec": "1.0.0",
  "status": "COMPLETE",
  "git": {
    "commit": "81be...",
    "dirty": false
  },
  "dataset": {
    "repository": "danidanou/Bloomberg_Financial_News",
    "revision": "...",
    "sha256": "..."
  },
  "sample": {
    "size": 10000,
    "seed": 917341,
    "manifest_sha256": "..."
  },
  "ontology": {
    "version": "1.0.0",
    "sha256": "..."
  },
  "models": {
    "jev": {
      "requested": "...",
      "returned_versions": ["..."]
    }
  },
  "config_sha256": "...",
  "template_hashes": {
    "jev": "...",
    "gliclass": "...",
    "nli": "..."
  },
  "started_at": "...",
  "completed_at": "..."
}
```

No human judgments or metrics belong in this file.

------------------------------------------------------------------------

## 30. Git and environment traceability

Every run MUST record:

-   Git commit;
-   dirty status;
-   classification specification version;
-   Python version;
-   operating system;
-   CPU/GPU identifiers where relevant;
-   CUDA version where relevant;
-   model-library versions.

Primary benchmark runs SHOULD reject dirty Git state.

If exploratory runs permit dirty state, the run identity MUST include a
hash of the diff.

A lockfile such as `uv.lock` SHOULD pin dependencies.

------------------------------------------------------------------------

## 31. Label-order robustness

Primary runs use one canonical ontology child order.

A separate robustness run MAY permute child order deterministically.

Such a run is a distinct classification run and therefore receives a
distinct run ID.

Do not mix robustness permutations into the primary result file.

------------------------------------------------------------------------

## 32. Taxonomy-scale experiments

Separate classification runs MAY test:

-   broad taxonomy;
-   intermediate taxonomy;
-   full leaf taxonomy.

Each is represented by a separate ontology version or explicit ontology
view configuration and therefore produces a separate run ID.

The classifier itself remains unchanged.

------------------------------------------------------------------------

## 33. Definition-ablation experiments

Separate runs MAY compare:

1.  label name only;
2.  label name + definition;
3.  label name + definition + inclusions/exclusions.

This MUST be controlled through versioned templates/configuration.

Primary benchmark outputs MUST not be retroactively changed based on
ablation results.

------------------------------------------------------------------------

## 34. CLI

Required commands:

``` bash
classify validate --config configs/classification.yaml
```

Validates configuration, dataset metadata, ontology, templates,
credentials, and model pinning without running inference.

``` bash
classify prepare --config configs/classification.yaml
```

Builds the deterministic sample manifest and computes the final run
identity.

``` bash
classify run --config configs/classification.yaml
```

Runs all enabled models.

``` bash
classify run --config configs/classification.yaml --model jev
```

Runs one enabled model.

``` bash
classify run --config configs/classification.yaml --model jev --resume
```

Resumes missing/failed work without changing the run identity.

No adjudication/evaluation commands belong in this CLI.

------------------------------------------------------------------------

## 35. Repository structure

Recommended classification repository:

``` text
jev-basel-classifier/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
│
├── configs/
│   └── classification.yaml
│
├── ontology/
│   └── gsib_basel_v1.0.yaml
│
├── templates/
│   ├── jev/
│   ├── gliclass/
│   ├── nli/
│   └── embeddings/
│
├── schemas/
│   ├── config.schema.json
│   ├── ontology.schema.json
│   └── result.schema.json
│
├── src/
│   └── classifier/
│       ├── cli.py
│       ├── config.py
│       ├── provenance.py
│       ├── dataset.py
│       ├── sampling.py
│       ├── inputs.py
│       ├── ontology.py
│       ├── traversal.py
│       ├── results.py
│       └── adapters/
│           ├── base.py
│           ├── jev.py
│           ├── gliclass.py
│           ├── nli.py
│           └── embeddings.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── classification_results/
```

The adjudication/evaluation implementation SHOULD live in a separate
package or repository.

------------------------------------------------------------------------

## 36. Tests required before benchmark execution

### Ontology tests

-   no duplicate IDs;
-   no cycles;
-   every child exists;
-   every child has exactly one declared structural parent unless
    explicitly allowed;
-   every non-leaf has at least one child;
-   every leaf has no children;
-   required semantic fields are populated;
-   root nodes exist.

### Traversal tests

-   multi-label branching works;
-   low-confidence parents can still traverse at the traversal
    threshold;
-   final-positive threshold is independent of traversal threshold;
-   maximum depth is enforced;
-   maximum nodes per article is enforced;
-   a provider error does not become a negative classification.

### Reproducibility tests

-   identical config produces identical run ID;
-   changing ontology text changes run ID;
-   changing threshold changes run ID;
-   changing sample seed changes run ID;
-   changing template changes run ID;
-   changing model revision changes run ID.

### Adapter contract tests

Every adapter must pass common fixtures and emit the same normalized
`NodeDecision` shape.

------------------------------------------------------------------------

## 37. Classification definition of done

The v1 classification system is complete when it can:

1.  validate the standalone ontology;
2.  pin and verify a Bloomberg dataset snapshot;
3.  deterministically select articles;
4.  construct versioned model inputs;
5.  execute hierarchical multi-label traversal through multiple model
    calls;
6.  run Jev and at least one comparator through the common adapter
    contract;
7.  write only normalized classification results plus minimal provenance
    manifest;
8.  resume interrupted runs without duplicating successful calls;
9.  produce a different run identity for every scientifically material
    input/configuration change;
10. operate without importing or referencing any human adjudication
    data.

------------------------------------------------------------------------

## 38. Downstream contract for adjudication

The only contract between classification and downstream evaluation is
the classification result schema.

The future adjudication pipeline may:

-   load two or more classification runs;
-   align them by `article_id` and ontology version;
-   detect disagreements;
-   sample consensus cases;
-   create blinded review queues;
-   store human judgments;
-   compute gold labels and metrics.

It MUST NOT modify classification outputs.

If human review leads to an ontology revision, that revision creates a
new ontology version and a new classification run rather than rewriting
historical results.
