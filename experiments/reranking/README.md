# Reranking benchmark: FiQA-2018 (2026-11-02)

Jev as a search reranker, yes/no vs graded (turbopuffer's 1-10 method), against Cohere Rerank v5 pro and
Voyage rerank-3. BEIR FiQA-2018 test split (BeIR/fiqa @979c07a, BeIR/fiqa-qrels @252958f): 648 questions,
57,638 passages. BM25 (k1 0.9, b 0.4) retrieves the top 100; every reranker re-orders the same 100. Ties broken
by BM25 rank. All calls through Vercel AI Gateway. Script: `fiqa_rerank.py` (candidates / run / score).

| Reranker | nDCG@10 | MRR@10 | Recall@10 | Cost (648 questions) |
|---|---|---|---|---|
| Voyage rerank-3 | **0.463** | **0.588** | **0.469** | $1.20 |
| Jev, graded (1-10) | 0.410 | 0.507 | 0.451 | $2.41 |
| Jev, yes/no | 0.405 | 0.508 | 0.443 | $1.77 |
| Cohere Rerank v5 pro | 0.403 | 0.494 | 0.445 | $1.15 |
| BM25 only | 0.231 | 0.286 | 0.290 | - |

Paired bootstrap 95% CIs (nDCG@10): graded - yes/no +0.006 [-0.002, +0.014]; Jev yes/no - Cohere +0.002
[-0.012, +0.015]; Voyage - Jev graded +0.053 [+0.040, +0.065]. BM25 top-100 holds a relevant passage for 69.6%
of questions (recall@100 0.492). Graded gives a finer ordering (median 86 distinct scores per 100 candidates vs
27 for yes/no, whose probabilities are 2-decimal) but no meaningful nDCG gain.

## Jev versions (checked 2026-11-02)
The unversioned `typesafe-ai/jev` alias serves 1.14.0: on 2,000 FiQA pairs its yes/no scores matched pinned
`jev-1.14.0` as often as a repeat of itself (63% identical, mean |diff| 0.006) vs 33% / 0.022 for `jev-1.13.0`.
The bare `jev_yesno` / `jev_graded` arms above are therefore Jev 1.14. Pinned 1.13.0 on all 648 questions:
yes/no 0.401, graded 0.403 (1.14 - 1.13: yes/no +0.004 [-0.002, +0.009], graded +0.007 [-0.000, +0.015]).

Reuters-21578 test under pinned versions (classification, macro F1): original questions 84.8 (1.13,
`jev-694513e76cbaa24b`) / 84.5 (1.14, `jev-f37c9cc0b3ab8e2c`); v3.1 questions 89.1 (1.13, `jev-06dcc38a85cd4e66`)
/ 89.2 (1.14, `jev-a7d9d6299b9b694f`). The version does not explain the prompt-wording gains.

## Comparison with kapa.ai's test
kapa.ai (r/LLMDevs, 1,000 company-knowledge cases, top-50 candidates) found Voyage rerank-3 first (0.535), Jev 1.14
yes/no 0.003 behind, Jev ahead of Cohere rerank-v4.0-pro, and graded slightly below yes/no. FiQA agrees on the order
of Voyage first and graded no better than yes/no; Jev ties Cohere v5 pro here and trails Voyage by 0.058.
