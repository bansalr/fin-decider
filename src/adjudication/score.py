"""Score model runs against human labels from a labelling queue.

Weights undo the stratified sampling, so precision/recall estimate performance on the
population the queue was drawn from (articles every source model scored).
"""

from __future__ import annotations

import csv
from pathlib import Path

from report.summary import cohen_kappa

TOP = {"gsib_activity": "relevance.gsib_activity", "market_context_only": "relevance.market_context_only",
       "neither": "relevance.neither"}
ARMS = {"markets": "business.markets", "lending": "business.lending", "treasury_alm": "business.treasury_alm"}


def _yn(v: str) -> int | None:
    v = (v or "").strip().lower()
    return 1 if v in ("y", "yes", "1", "true") else 0 if v in ("n", "no", "0", "false") else None


def read_labels(path: Path) -> dict[str, dict]:
    """label_id -> {node_id: 0/1, "_leaves": set}; unlabelled rows are skipped."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            top = {TOP[c]: _yn(row.get(c, "")) for c in TOP}
            if any(v is None for v in top.values()):
                continue
            lab = dict(top)
            g = top["relevance.gsib_activity"]
            for c, nid in ARMS.items():
                lab[nid] = (_yn(row.get(c, "")) or 0) if g else 0
            lab["_leaves"] = {x.strip() for x in (row.get("leaves") or "").split(";") if x.strip()}
            out[row["label_id"]] = lab
    return out


def read_key(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["label_id"]: {"article_id": r["article_id"], "stratum": r["stratum"], "weight": float(r["weight"])}
                for r in csv.DictReader(fh)}


def annotator_agreement(a: dict[str, dict], b: dict[str, dict]) -> dict:
    ids = sorted(set(a) & set(b))
    out = {"items": len(ids)}
    for nid in list(TOP.values()) + list(ARMS.values()):
        x, y = [a[i][nid] for i in ids], [b[i][nid] for i in ids]
        out[nid] = {"agreement": sum(p == q for p, q in zip(x, y)) / len(ids) if ids else None, "kappa": cohen_kappa(x, y)}
    out["disagreements"] = [i for i in ids if any(a[i][n] != b[i][n] for n in TOP.values())]
    return out


def _prf(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else None
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


def score(labels: dict[str, dict], key: dict[str, dict], predicted: dict[str, set[str]], leaf_ids: set[str]) -> dict:
    """predicted: article_id -> set of positive child ids from one model run."""
    out, missing = {}, 0
    nodes = list(TOP.values()) + list(ARMS.values())
    acc = {n: [0.0, 0.0, 0.0] for n in nodes + ["leaves(micro)"]}
    for lid, lab in labels.items():
        k = key[lid]
        if k["article_id"] not in predicted:
            missing += 1
            continue
        w, pred = k["weight"], predicted[k["article_id"]]
        for n in nodes:
            y, p = lab[n], int(n in pred)
            acc[n][0] += w * (y and p)
            acc[n][1] += w * (p and not y)
            acc[n][2] += w * (y and not p)
        gold, got = lab["_leaves"] & leaf_ids, pred & leaf_ids
        acc["leaves(micro)"][0] += w * len(gold & got)
        acc["leaves(micro)"][1] += w * len(got - gold)
        acc["leaves(micro)"][2] += w * len(gold - got)
    for n, (tp, fp, fn) in acc.items():
        out[n] = _prf(tp, fp, fn)
    out["_scored"] = len(labels) - missing
    out["_missing_predictions"] = missing
    return out
