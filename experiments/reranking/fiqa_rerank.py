"""Reranking benchmark on FiQA-2018 (BEIR test split): Jev (yes/no and graded) vs Cohere Rerank vs Voyage rerank.

Protocol (the usual BEIR rerank setup):
  1. BM25 retrieves the top 100 passages per test question (Okapi BM25, k1=0.9, b=0.4, title + text,
     lower-cased word tokens, no stemming). Our BM25 is a plain re-implementation, so its own score
     differs a little from the Elasticsearch BM25 BEIR reports.
  2. Each reranker re-orders those 100 candidates. Ties (Jev returns 2-decimal probabilities) are broken by
     BM25 rank, the same for every model.
  3. Scored with nDCG@10 (BEIR's headline metric), MRR@10 and Recall@10 against the official test qrels;
     paired bootstrap 95% CIs over questions.

Rerankers (all through Vercel AI Gateway with VERCEL_API_KEY):
  jev_yesno   typesafe-ai/jev, one boolean question per (question, passage): "Does this passage answer ...?"
  jev_graded  typesafe-ai/jev, one 1-10 choice question per pair; score = probability-weighted mean level
              (turbopuffer's method)
  cohere      cohere/rerank-v5-pro          voyage  voyage/rerank-3
Every API response is cached under data/bench/fiqa/cache/<arm>.jsonl, so reruns resume and cost nothing.

    .venv/bin/python experiments/reranking/fiqa_rerank.py candidates
    .venv/bin/python experiments/reranking/fiqa_rerank.py run --arms jev_yesno jev_graded cohere voyage [--queries 20]
    .venv/bin/python experiments/reranking/fiqa_rerank.py score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import polars as pl
from dotenv import load_dotenv

D = Path("data/bench/fiqa")
CACHE = D / "cache"
GATEWAY = "https://ai-gateway.vercel.sh"
TOPK = 100
MAX_CHARS = 4000
TOK = re.compile(r"\w+")

LEVELS = {
    "1": "Unrelated: the passage has nothing to do with the question.",
    "2": "Barely related: the passage only shares a broad field with the question.",
    "3": "Mentions the subject but does not address the question.",
    "4": "Touches the question's topic but gives no useful answer.",
    "5": "Partly relevant: some useful information, but mostly about something else.",
    "6": "Relevant background that helps answer the question.",
    "7": "Partially answers the question.",
    "8": "Mostly answers the question.",
    "9": "Answers the question well.",
    "10": "Perfect match: directly and completely answers the question.",
}
RERANK_MODELS = {"cohere": "cohere/rerank-v5-pro", "voyage": "voyage/rerank-3"}


def passage(row) -> str:
    t = (row["title"] + "\n\n" if row["title"] else "") + row["text"]
    return t[:MAX_CHARS]


def load():
    corpus = pl.read_parquet(D / "corpus-00000-of-00001.parquet")
    queries = pl.read_parquet(D / "queries-00000-of-00001.parquet")
    qrels = pl.read_csv(D / "test.tsv", separator="\t", schema_overrides={"query-id": pl.Utf8, "corpus-id": pl.Utf8})
    return corpus, queries, qrels


# ---- 1. BM25 candidates ------------------------------------------------------------------------------------
def candidates() -> None:
    corpus, queries, qrels = load()
    k1, b = 0.9, 0.4
    ids = corpus["_id"].to_list()
    docs = [TOK.findall((r["title"] + " " + r["text"]).lower()) for r in corpus.iter_rows(named=True)]
    N = len(docs)
    avgdl = sum(map(len, docs)) / N
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, d in enumerate(docs):
        for w, tf in Counter(d).items():
            postings[w].append((i, tf))
    dl = [len(d) for d in docs]
    idf = {w: math.log(1 + (N - len(p) + 0.5) / (len(p) + 0.5)) for w, p in postings.items()}
    test_q = set(qrels["query-id"].to_list())
    rows = []
    for qid, text in queries.select("_id", "text").iter_rows():
        if qid not in test_q:
            continue
        sc: dict[int, float] = defaultdict(float)
        for w in set(TOK.findall(text.lower())):
            for i, tf in postings.get(w, ()):
                sc[i] += idf[w] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[i] / avgdl))
        top = sorted(sc.items(), key=lambda t: -t[1])[:TOPK]
        rows += [{"query_id": qid, "rank": r, "corpus_id": ids[i], "bm25": s} for r, (i, s) in enumerate(top)]
    pl.DataFrame(rows).write_parquet(D / "candidates_bm25_top100.parquet")
    print(f"{len(test_q)} test questions, {len(rows):,} candidate pairs -> {D / 'candidates_bm25_top100.parquet'}")


# ---- 2. rerankers ----------------------------------------------------------------------------------------
def cached(arm: str) -> dict[str, dict]:
    p = CACHE / f"{arm}.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if "error" in r and r["key"] in out:
                continue  # keep an earlier success
            out[r["key"]] = r
    return out


async def post(client, path, body, retries=6):
    for attempt in range(retries + 1):
        try:
            r = await client.post(path, json=body)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 402, 403):
                raise SystemExit(f"fatal {r.status_code}: {r.text[:300]}")
            if r.status_code in (400, 413, 422):
                return {"error": f"{r.status_code}: {r.text[:300]}"}
        except httpx.HTTPError:
            pass
        await asyncio.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
    return {"error": "retries exhausted"}


def gw_cost(d) -> float:
    gw = ((d.get("providerMetadata") or d.get("provider_metadata") or {}).get("gateway") or {})
    try:
        return float(gw.get("cost") or 0)
    except (TypeError, ValueError):
        return 0.0


async def run_arm(arm: str, qsel: list[str], cands: pl.DataFrame, text: dict, qtext: dict, conc: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    have = {k: v for k, v in cached(arm).items() if "error" not in v}  # errors are retried
    H = {"Authorization": f"Bearer {os.environ['VERCEL_API_KEY']}", "Content-Type": "application/json"}
    sem = asyncio.Semaphore(conc)
    out = open(CACHE / f"{arm}.jsonl", "a")
    by_q = {k[0]: g.sort("rank")["corpus_id"].to_list() for k, g in cands.group_by("query_id") if k[0] in set(qsel)}
    done = 0

    async with httpx.AsyncClient(base_url=GATEWAY, headers=H, timeout=120,
                                 limits=httpx.Limits(max_connections=conc * 2)) as client:
        async def one_jev(qid, cid):
            nonlocal done
            key = f"{qid}|{cid}"
            if key in have:
                return
            q = qtext[qid]
            if arm == "jev_yesno":
                question = {"type": "boolean", "instructions": f'Does this passage answer the question: "{q}"?'}
            else:
                question = {"type": "choice", "criteria": LEVELS,
                            "instructions": f'How relevant is the passage to the question: "{q}"? Choose the level that fits best.'}
            body = {"model": "typesafe-ai/jev", "state": text[cid], "questions": {"q1": question},
                    "providerOptions": {"gateway": {"only": ["typesafe-ai"]}}}
            async with sem:
                d = await post(client, "/v1/evaluate", body)
            rec = {"key": key, "cost": gw_cost(d)}
            if "error" in d:
                rec["error"] = d["error"]
            else:
                a = (d.get("answers") or {}).get("q1") or {}
                if arm == "jev_yesno":
                    rec["score"] = a.get("probability")
                else:
                    probs = {int(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
                    m = sum(probs.values())
                    rec["score"] = (sum(k * p for k, p in probs.items()) / m - 1) / 9 if m else None
                    rec["probs"] = a.get("probabilities")
            out.write(json.dumps(rec) + "\n")
            done += 1
            if done % 2000 == 0:
                out.flush()
                print(f"  {arm}: {done:,} new pairs", flush=True)

        async def one_rerank(qid):
            nonlocal done
            if qid in have:
                return
            docs = by_q[qid]
            body = {"model": RERANK_MODELS[arm], "query": qtext[qid], "documents": [text[c] for c in docs]}
            async with sem:
                d = await post(client, "/v1/rerank", body)
            rec = {"key": qid, "cost": gw_cost(d), "tokens": ((d.get("meta") or {}).get("billed_units") or {}).get("input_tokens")}
            if "error" in d:
                rec["error"] = d["error"]
            else:
                rec["scores"] = {docs[r["index"]]: r["relevance_score"] for r in d.get("results", [])}
            out.write(json.dumps(rec) + "\n")
            done += 1

        if arm.startswith("jev"):
            await asyncio.gather(*(one_jev(q, c) for q in qsel for c in by_q[q]))
        else:
            await asyncio.gather(*(one_rerank(q) for q in qsel))
    out.close()
    print(f"{arm}: {done:,} new requests")


def run(arms: list[str], nq: int | None, conc: int) -> None:
    corpus, queries, qrels = load()
    cands = pl.read_parquet(D / "candidates_bm25_top100.parquet")
    qids = sorted(cands["query_id"].unique().to_list())
    if nq:
        qids = random.Random(20261102).sample(qids, nq)
    need = set(cands.filter(pl.col("query_id").is_in(qids))["corpus_id"].to_list())
    text = {r["_id"]: passage(r) for r in corpus.filter(pl.col("_id").is_in(list(need))).iter_rows(named=True)}
    qtext = dict(queries.select("_id", "text").iter_rows())
    for arm in arms:
        asyncio.run(run_arm(arm, qids, cands, text, qtext, conc))


# ---- 3. scoring ------------------------------------------------------------------------------------------
def score(nq: int | None) -> None:
    _, _, qrels = load()
    rel = defaultdict(dict)
    for q, c, s in qrels.select("query-id", "corpus-id", "score").iter_rows():
        rel[q][c] = s
    cands = pl.read_parquet(D / "candidates_bm25_top100.parquet")
    qids = sorted(cands["query_id"].unique().to_list())
    if nq:
        qids = random.Random(20261102).sample(qids, nq)
    by_q = {k[0]: g.sort("rank")["corpus_id"].to_list() for k, g in cands.group_by("query_id")}
    arms = {"bm25": None}
    costs = {}
    for arm in ("jev_yesno", "jev_graded", "cohere", "voyage"):
        c = cached(arm)
        if c:
            arms[arm] = c
            costs[arm] = sum(r.get("cost", 0) for r in c.values())

    def ranking(arm, q):
        docs = by_q[q]
        if arm == "bm25":
            return docs
        c = arms[arm]
        if arm.startswith("jev"):
            s = {d: (c.get(f"{q}|{d}") or {}).get("score") for d in docs}
        else:
            s = (c.get(q) or {}).get("scores") or {}
        if any(s.get(d) is None for d in docs):
            return None
        return sorted(docs, key=lambda d: (-s[d], docs.index(d)))

    def metrics(order, q):
        g = rel[q]
        dcg = sum((2 ** g.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(order[:10]))
        ideal = sorted(g.values(), reverse=True)[:10]
        idcg = sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(ideal))
        mrr = next((1 / (i + 1) for i, d in enumerate(order[:10]) if g.get(d, 0) > 0), 0.0)
        rec = sum(1 for d in order[:10] if g.get(d, 0) > 0) / len(g)
        return dcg / idcg if idcg else 0.0, mrr, rec

    per = {}
    complete = [q for q in qids if all(ranking(a, q) is not None for a in arms)]
    for a in arms:
        per[a] = [metrics(ranking(a, q), q) for q in complete]
    ceiling = sum(1 for q in complete if any(d in rel[q] for d in by_q[q])) / len(complete)
    recall100 = sum(len(set(by_q[q]) & set(rel[q])) / len(rel[q]) for q in complete) / len(complete)
    print(f"{len(complete)} of {len(qids)} questions scored by every arm; BM25 top-100 holds >=1 relevant passage for "
          f"{ceiling:.1%} of questions (recall@100 {recall100:.3f})\n")
    print("| reranker | nDCG@10 | MRR@10 | Recall@10 | cost (these questions) |\n|---|---|---|---|---|")
    for a, m in per.items():
        n = len(m)
        print(f"| {a} | {sum(x[0] for x in m) / n:.3f} | {sum(x[1] for x in m) / n:.3f} | {sum(x[2] for x in m) / n:.3f} | "
              f"{'$%.2f' % costs[a] if a in costs else '-'} |")
    rng = random.Random(0)
    print("\npaired bootstrap of nDCG@10 differences (2,000 resamples of questions), 95% CI:")
    pairs = [("jev_graded", "jev_yesno"), ("jev_graded", "cohere"), ("jev_graded", "voyage"), ("jev_yesno", "cohere"),
             ("jev_yesno", "voyage"), ("cohere", "voyage"), ("jev_yesno", "bm25")]
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        diff = [x[0] - y[0] for x, y in zip(per[a], per[b])]
        bs = sorted(sum(diff[rng.randrange(len(diff))] for _ in diff) / len(diff) for _ in range(2000))
        print(f"  {a} - {b}: {sum(diff) / len(diff):+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}]")


def main() -> None:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "run", "score"])
    ap.add_argument("--arms", nargs="*", default=["jev_yesno", "jev_graded", "cohere", "voyage"])
    ap.add_argument("--queries", type=int, default=None, help="random sample of N test questions (fixed seed)")
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "candidates":
        candidates()
    elif a.cmd == "run":
        run(a.arms, a.queries, a.concurrency)
    else:
        score(a.queries)


if __name__ == "__main__":
    main()
