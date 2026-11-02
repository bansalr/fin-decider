# Graded relevance scoring study (2026-11-02)

Idea (from turbopuffer's reranking write-up): instead of one yes/no probability per question, ask Jev to rate
the match on a 1-10 scale (one `choice` question per child, one option per level) and score the child by the
expected level, `sum(level x probability)`, rescaled to 0..1 (so 0.5 = level 5.5). Implemented as template
kind `graded` (`src/classifier/templates.py`, `adapters/http_decision.py`); nothing existing was changed.

Templates: `template_graded_v1.txt` (generic relevance levels), `template_graded_v2.txt` (levels anchored to the
question's true/false criteria: 1-5 = does not count, 6-10 = counts), `templates/decision/graded_v2.txt`
(same idea for the G-SIB ontology).

## Part 1: Reuters-21578 R(10), Jev, v3.1 questions + criteria in every arm

Development on the first 1,000 training stories only (v1, then v2); confirmation on the other 5,410 training
stories; the test set run once with cutoffs fixed beforehand (`reports/graded/test_preregistration.txt`).

| Stage | Yes/no v3.1 | Graded v2 | Difference (95% CI, paired bootstrap) |
|---|---|---|---|
| Dev 1,000, macro F1 at 0.5 | 91.9 | 90.1 (v1: 89.7) | |
| Hold-out 5,410, macro F1 at 0.5 | 89.8 | 89.6 | -0.25 [-0.94, +0.51] |
| Hold-out, cutoff tuned on dev (0.51 / 0.35) | 89.6 | 90.6 | +1.04 [+0.38, +1.68] |
| **Test 2,545, pre-registered cutoffs (0.51 / 0.35)** | **89.1** | **90.9** | **+1.82 [+0.54, +3.13]** |
| Test, macro F1 at 0.5 | 89.1 | 88.2 | -0.94 [-2.27, +0.40] |
| Test, macro AUC / average precision | 99.8 / 96.3 | 99.8 / 95.8 | |

Reading: ranking quality is unchanged (AUC 99.8 both). The gain comes from the graded score's best cutoff
being stable (0.34-0.39 on dev, hold-out and test), where the yes/no score's is not (0.51 dev, 0.36 hold-out,
0.24 test). Graded cuts acquisition false positives sharply but adds some in interest, trade and money-fx.

## Part 2: G-SIB ontology 1.1.1, 300 Claude-labelled Bloomberg articles

Arms on the same articles and commit: `jev` (production `boolean_v2c`, question only), `jev_v2a`
(`boolean_v2a`, question + criteria), `jev_graded` (`graded_v2`, question + criteria). Config
`configs/gsib_graded_study.yaml`, run with `--ids-file labeling/queue_v1_ids.txt`.
Scored by `scripts/compare_graded_gsib.py` (output: `reports/graded/gsib_labelled300.md`).

| Arm | AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | F1, 2-fold tuned | FP@0.5 | Leaf AUC | Leaf F1@0.5 |
|---|---|---|---|---|---|---|---|---|
| Yes/no, question only (production) | 0.962 | 0.60 | 0.92 | 0.72 | 0.80 | 38 | 0.802 | 0.56 |
| Yes/no + criteria | **0.981** | 0.70 | 0.93 | 0.80 | 0.86 | 24 | 0.839 | 0.62 |
| Graded + criteria | 0.966 | **0.79** | 0.90 | **0.84** | **0.87** | **15** | **0.861** | **0.65** |

Caveats: one labeller (Claude); 61 positives; the queue over-samples model disagreements.

## Reproduce
```
classify run -c configs/native_reuters_top10_train_graded_v2.yaml -m jev --no-dev-eval        # train split
classify run -c configs/native_reuters_top10_graded_v2.yaml -m jev --no-dev-eval              # test, once
python scripts/compare_graded_reuters.py --corpus corpus_native/reuters_top10 \
  --tune-corpus corpus_native/reuters_top10_train --tune-first 1000 --tune yesno=<train v3.1> graded=<train graded> \
  yesno=<test v3.1> graded=<test graded>
classify run -c configs/gsib_graded_study.yaml -m jev -m jev_v2a -m jev_graded --ids-file labeling/queue_v1_ids.txt
python scripts/compare_graded_gsib.py jev=<run> jev_v2a=<run> jev_graded=<run>
```
Runs: Reuters train v3.1 `jev-16bc31eb63e4f789`, train graded v2 `jev-7d08c09027d27f83`, test v3.1
`jev-1122dced82d61b05`, test graded v2 `jev-e9fe5eed997073bc`; G-SIB `jev-b0a13f2509f7ef84`,
`jev_v2a-8083d5cf4c331d41`, `jev_graded-fdefadbec9d45b42`. Total Jev cost of the study: $2.94.
