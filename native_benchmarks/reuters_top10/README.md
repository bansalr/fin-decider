# Native-task benchmark: Reuters-21578 R(10) topics

A separate track from the G-SIB study. Jev is scored on **the dataset's own labels** with the **metrics published for the same test set**, so the results can sit beside other models' published numbers. Nothing in the G-SIB ontology, templates, configs or scorers is shared or changed.

| Piece | File |
|---|---|
| Data source | `src/classifier/sources/reuters21578_top10.py` (provider `uci_reuters21578_top10`, same pinned archive as `bench_reuters`) |
| Task ontology | `native_benchmarks/reuters_top10/ontology_v1.json` (root `topics`, 10 leaves) |
| Template | `native_benchmarks/reuters_top10/template_v1.txt` (the leaf's yes/no question only) |
| Config | `configs/native_reuters_top10.yaml` (Jev, d1 and Clef; corpus `corpus_native/reuters_top10`; results in `classification_results_native/`) |
| Scorer | `scripts/score_native_reuters.py` |
| Tests | `tests/unit/test_native_reuters.py` |

## Test set
The standard **ModApte R(10)** test set: `LEWISSPLIT="TEST"`, `TOPICS="YES"`, at least one of the ten most frequent topics.
- **Stories:** 2,545, the same count as LabelFusion. 2,533 are unique; the 12 exact duplicates share their canonical story's answer and are each scored.
- **Topic counts:**

  | earn | acq | crude | money-fx | grain | interest | trade | ship | wheat | corn |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 1,087 | 719 | 189 | 179 | 149 | 131 | 117 | 89 | 71 | 56 |

Title-only briefs (217) and the unprocessed stories are kept, as in the standard set. This is the one difference in selection from `bench_reuters`.

## Task framing (fixed before any run)
- **Flat multi-label:** one request per story, with 10 independent yes/no questions (noul), one per topic. grain, wheat and corn are siblings rather than a hierarchy, so a miss on grain can't hide wheat.
- **Question wording:** each question is a plain reading of the topic's **official Reuters name** (`cat-descriptions_120396.txt`, e.g. *Earnings and Earnings Forecasts (EARN)*).
  - The wording was written without looking at test stories or Reuters' indexing conventions.
  - It **must not be revised after seeing results.** A revision means a new ontology version reported separately, preferably tuned on the training split.
- **Positive threshold:** 0.50, as in the G-SIB study.

## Metrics and comparison
- **Headline:** macro-averaged F1, precision and recall over the 10 topics (sklearn conventions).
- **Also reported:** micro-F1, subset accuracy and Hamming accuracy. LabelFusion doesn't define its multi-label "accuracy", so neither accuracy figure is used for the comparison.

**Published reference:** LabelFusion, FNP 2026 ([ACL Anthology 2026.fnp-1.9](https://aclanthology.org/2026.fnp-1.9/), [arXiv 2512.10793](https://arxiv.org/abs/2512.10793)), Table 1, macro %:

| Model | Training data | F1 | Prec | Rec |
|---|---|---:|---:|---:|
| GPT-5-nano, zero-shot | none | 75.9 | 89.2 | 70.5 |
| GPT-5-nano, one-shot | 1 example | 76.1 | 92.0 | 68.0 |
| TF-IDF + logistic regression | 5,842 stories | 68.2 | 95.4 | 56.9 |
| RoBERTa, fine-tuned | 5,842 stories | 94.6 | 93.2 | 96.6 |
| LabelFusion | 5,842 stories | 96.0 | 96.7 | 96.1 |

**Like-for-like is zero-shot vs zero-shot.** The fine-tuned rows learned Reuters' indexing conventions from its training split; for example, dividend and stock-split notices are indexed `earn`. LabelFusion doesn't publish its prompt or input handling, so small differences in setup remain.

## Commands
```bash
.venv/bin/classify prepare -c configs/native_reuters_top10.yaml          # local, no API calls
.venv/bin/classify run -c configs/native_reuters_top10.yaml -m jev --no-dev-eval --max-cost-usd 1
.venv/bin/python scripts/score_native_reuters.py --run classification_results_native/<run_id>
```
Estimated Jev cost: about $0.10 for all 2,533 unique stories (one call per story).

## Revised questions (v2.1), developed on the training split only
After the pre-registered v1 run, the questions were revised **using only the ModApte training split** (`configs/native_reuters_top10_train*.yaml`, a fixed sample of the first 1,000 stories by article id).

**v1 errors on that sample were mostly Reuters' filing conventions:**
- **earn:** dividend declarations and stock splits are filed under earn, and v1 missed them.
- **trade:** means trade policy and trade balances, not single commodity sales or shipments.

**How v2.1 was chosen:**
- **v2** reworded six topics.
- **v2.1** keeps the v2 wording only where it raised F1 on the training sample: earn and trade. crude, interest, ship and wheat reverted to v1, because v2 made them too strict.
- The test split was then run **once**, with v2.1 frozen (`ontology_v2_1.json`, commit before the run).

| Jev, zero-shot (macro %) | Training-split sample F1 | Test F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|
| v1 (pre-registered) | 83.3 | 84.8 | 79.4 | 94.1 |
| v2 (intermediate, not run on test) | 84.0 | n/a | n/a | n/a |
| **v2.1** | **86.1** | **87.1** | **81.4** | **95.0** |

On the test set, v2.1 raised earn from 92.6 to 96.4 F1 and trade from 54.0 to 72.7.

**How to report it:** report v1 as the zero-shot result. v2.1 is "zero-shot with category guidance written from training-split errors". The model never trains on labels, but the wording draws on 1,000 labelled training stories, so it sits between zero-shot and few-shot.
