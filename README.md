# fin-decider

Hierarchical, multi-label classification of the full deduplicated Bloomberg
Financial News corpus into the Basel-derived G-SIB ontology
(`gsib_basel_ontology_v1.0.json`), with Jev and Laya as anchor models.

The specification is `jev_bloomberg_basel_classification_spec_v1.0_edited.md`;
its §0 lists the v1.1 amendments this implementation follows. This package is the
classifier only: it writes classification results and a run manifest, never
labels, metrics, or adjudication artifacts.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install uv
UV_PROJECT_ENVIRONMENT=.venv .venv/bin/uv sync
cp .env.example .env   # then set VERCEL_API_KEY (Vercel AI Gateway)
```

The Gateway account needs a credit card on file before it serves any request,
including Laya's free tier.

## Pipeline

```bash
.venv/bin/classify validate                 # config, ontology, template, dataset hash, credentials
.venv/bin/classify prepare                  # download pinned dataset, dedupe, freeze corpus/ (~minutes)
.venv/bin/classify probe -m jev             # one live decision; prints returned model, routing, usage
.venv/bin/classify probe -m laya
.venv/bin/classify run --limit 50           # engineering smoke run (needs run.allow_dirty_git or a clean tree)
.venv/bin/classify run -m laya --resume     # full corpus
.venv/bin/classify run -m jev --resume --max-cost-usd 300
```

`--dry-run` swaps in an offline deterministic adapter to exercise the pipeline
without network calls. `--resume` continues an interrupted run under the same run
ID, reusing every successful decision.

## Outputs

```text
corpus/                    canonical.parquet, clusters.parquet, manifest.json
classification_results/<run_id>/
  manifest.json            identity, provenance, model catalog snapshots, status
  run.log                  progress log
  _work/model=<m>/         append-only shards (all attempts)
  results/model=<m>/       final rows, latest attempt per article
```

One row per article × model × node × child. Per-call usage fields repeat across
the child rows of a call; aggregate them by `prediction_key`. Final labels are the
`SUCCESS` rows with `selected_positive`.

## Tests

```bash
.venv/bin/python -m pytest -q
```
