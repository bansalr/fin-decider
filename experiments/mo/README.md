# Mo (Modaic) evaluation, 2026-11-23/24

`modaic/mo` via Vercel AI Gateway: same `/v1/evaluate` request, questions, templates, cutoffs and inputs as Jev
(adapter `mo`, config entries `mo` / `mo_v2a` / `mo_graded`, disabled by default). Plan: everything Jev ran except
the full 445,391-article corpus (Bloomberg ran on the 14,699-article slice shared with d1 and Clef, which contains
the 300 labelled articles). Total Vercel usage: $11.30. Run logs: `logs/mo_plan/`; analysis outputs: `reports/mo/`.

## Headline
Mo ranks as well as Jev but is far more conservative at the shared 0.5 cutoff. With a cutoff of about 0.25-0.30 it
matches Jev on our G-SIB labels, beats it on the known-answer benchmarks, and is the best decision model at FiQA
reranking; it is about a third cheaper per article and slower per call.

## Bloomberg G-SIB activity (300 labelled articles weighted to the corpus)
| Model / cutoff | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Jev, 0.50 | 0.69 [0.58-0.78] | 0.95 [0.89-0.99] | 0.80 [0.72-0.86] | 0.967 |
| Mo, 0.50 | 0.98 [0.94-1.00] | 0.51 [0.38-0.64] | 0.67 [0.55-0.77] | 0.969 |
| Jev, cutoff tuned by 2-fold CV (0.67-0.76) | 0.87 | 0.86 | 0.86 | |
| Mo, cutoff tuned by 2-fold CV (0.28-0.30) | 0.83 | 0.88 | 0.85 | |

Share flagged on the 14,699 slice: Mo 5.4% at 0.50, 11.7% at 0.25; Jev 13.8% at 0.50 (true share about 9.7%).
Agreement with Jev 91.2% (kappa 0.50), with Clef 91.1% (0.46), with d1 82.6% (0.32).

Per node at 0.50 (weighted F1, Jev / Mo): market context 0.90 / 0.94; neither 0.73 / 0.87; markets 0.71 / 0.73;
lending 0.65 / 0.34; treasury 0.59 / 0.59; 31 categories pooled 0.51 / 0.53 (Mo precision 0.65, recall 0.44).

Prompt variants on the 300 (unweighted, F1 at 0.50 / 2-fold tuned): Mo question-only 0.60 / 0.83; + criteria
0.62 / 0.78; graded 0.69 / 0.73. Unlike Jev, criteria and graded scoring do not help Mo.

## Known-answer benchmarks (share flagged as G-SIB activity)
| Set | Should be | Jev 0.50 | Mo 0.50 | Mo 0.25 |
|---|---|---|---|---|
| FOMC | no | 0.6% | 0.7% | 1.1% |
| Gold headlines | no | 0.9% | 0.3% | 1.1% |
| Twitter, non-bank topics | no | 2.4% | 0.7% | 2.1% |
| CFPB complaints | yes | 78.4% | 16.6% | 55.9% |
| EDGAR bank filings | yes | 99.5% | 74.8% | |
| Wikipedia banking | yes | 95.3% | 48.7% | |

Pooled over FOMC, EDGAR, credit agreements and Wikipedia (8,190 items): F1 at 0.50 Jev 0.893, Mo 0.726; with each
model's cutoff tuned on the 300 labelled articles, Mo 0.912 (cutoff 0.25), Jev 0.830 (0.74). AUC 0.988 / 0.983.

## Reuters-21578 (macro F1, test set)
| Questions | Jev | Mo |
|---|---|---|
| Original | 84.8 | 83.4 |
| v2.1 | 87.1 | 86.4 |
| v3.1 | 89.1 | 87.2 (88.4 with cutoff 0.65 tuned on training data) |
| Graded v2, tuned cutoff | 90.9 | 89.8 (cutoff 0.39) |
Macro AUC (v3.1): Jev 99.8, Mo 99.2.

## FiQA-2018 reranking (nDCG@10, 648 questions, BM25 top 100)
Voyage rerank-3 0.463; Mo graded 0.419; Mo yes/no 0.415; Jev graded 0.410; Jev yes/no 0.405; Cohere v5 pro 0.403.
Mo - Jev (yes/no) +0.010 [-0.002, +0.023]; Mo yes/no - Cohere +0.012 [+0.000, +0.026]; Voyage - Mo +0.048
[+0.035, +0.061]. Cost: Mo $1.21 (yes/no), $1.61 (graded); Jev $1.77 / $2.41; Cohere $1.15; Voyage $1.20.

## Cost and speed
Bloomberg: $0.048 per 1,000 articles (Jev $0.072); about 22 articles/s at 32 parallel requests (Jev about 29
unthrottled); median call latency 668 ms (Jev 218 ms). Full corpus estimate: about $21 and 5.6 hours.
Mo returns full-precision probabilities (Jev rounds to two decimals). Mo bills no output tokens and uses about 14%
fewer input tokens than Jev on the same text. Three benchmark runs show multi-hour gaps; these were the Mac sleeping
(the run timer, which pauses during sleep, shows normal throughput), not Mo.

## Runs
Bloomberg slice `mo-d017575bddd8b8ca`; pilot `mo-e29db90d761354ab`; 300-article variants `mo-9538f2ec2561363b`,
`mo_v2a-0ae023cd9d1525bf`, `mo_graded-8df8743676374866`; Reuters test `mo-dbcc92337eedd45c` (v1),
`mo-05eaf8b2d42e4922` (v2.1), `mo-9c85e19a6f7a09f2` (v3.1), `mo-08d262cb74d82a08` (graded); training split
`mo-0cba36f95927430b`, `mo-5148f10396b5f772`; benchmarks listed in `logs/mo_plan/*.log`.
