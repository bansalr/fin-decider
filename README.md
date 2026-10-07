# fin-decider

Hierarchical, multi-label classification of the full deduplicated Bloomberg
Financial News corpus into the Basel-derived G-SIB ontology
(`gsib_basel_ontology_v1.0.json`), with Jev and Laya as anchor models and Liquid d1 and Cloudflare Clef as comparators.

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

### Cloudflare (Clef)

1. Log in at https://dash.cloudflare.com (or sign up).
2. Workers AI gives 10,000 free neurons per day: enough for a probe and the
   dev-set check, not for a 1,000-article run. Upgrade under
   **Workers & Pages → Plans → Workers Paid** ($5/month plus usage).
3. Open **AI → Workers AI → Use REST API**. Copy the **Account ID**, then
   **Create a Workers AI API Token** → **Create API Token**, and copy the token
   (shown once).
4. Add both to `.env`:
   ```
   CLOUDFLARE_ACCOUNT_ID=...
   CLOUDFLARE_API_TOKEN=...
   ```
5. Run `.venv/bin/classify probe -m clef` and set `expected_model_returned` in
   `configs/classification.yaml` to the `model_returned` it prints.

## Pipeline

```bash
.venv/bin/classify validate                 # config, ontology, template, dataset hash, credentials
.venv/bin/classify prepare                  # download pinned dataset, dedupe, freeze corpus/ (~minutes)
.venv/bin/classify probe -m jev             # one live decision; prints returned model, routing, usage
.venv/bin/classify probe -m laya
.venv/bin/classify run --dry-run --limit 3 --show 2 --set run.allow_dirty_git=true   # offline demo, fake scores
.venv/bin/classify run --limit 1000         # intermediate run (all enabled models), then a report per model
.venv/bin/classify run -m laya --resume     # full corpus
.venv/bin/classify run -m jev --resume --max-cost-usd 300
```

Every run ends with a report per model (cost/time/throughput, label distribution,
cross-model agreement, synthetic dev-set accuracy), printed and saved to
`reports/<run_id>.{json,md}`. `--show N` prints the traversal tree for the first N
articles; `--no-dev-eval` skips the dev-set check; `classify report -m <model>`
reprints a report. `--dry-run` swaps in an offline deterministic adapter to exercise the pipeline
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
