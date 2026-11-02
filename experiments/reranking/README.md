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
