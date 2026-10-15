"""Score a run on the Reuters-21578 R(10) native-task benchmark (configs/native_reuters_top10.yaml).

Separate from the G-SIB scorers (adjudication.*): gold labels are the stories' own Reuters topics, and the
metrics are the ones published for this test set, so a zero-shot Jev run can be set beside them.

Each of the 2,545 ModApte test stories is scored (exact duplicates share their canonical article's answer).
A topic is predicted when its child decision is selected positive (score >= the config's positive_threshold,
fixed at 0.50 before any run). Stories whose decision failed or is missing are reported, not scored.

Metrics (sklearn conventions, zero division -> 0):
    per topic   precision, recall, F1, support
    macro       mean over the 10 topics of precision, recall and F1   <- comparable to LabelFusion Table 1
    micro       pooled over all story-topic pairs
    subset accuracy (exact match of the whole topic set), Hamming accuracy (per story-topic pair)
LabelFusion does not define its multi-label "accuracy"; both readings are printed, neither is the headline.

    .venv/bin/python scripts/score_native_reuters.py --run classification_results_native/jev-...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from classifier import results

TOP10 = ("earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn")
PREFIX = "topic."

# LabelFusion (FNP 2026, ACL Anthology 2026.fnp-1.9; arXiv 2512.10793), Table 1:
# same 2,545-story ModApte R(10) test set, macro-averaged, percent. (model, training data, F1, P, R)
PUBLISHED = [
    ("GPT-5-nano, zero-shot", "none", 75.9, 89.2, 70.5),
    ("GPT-5-nano, one-shot", "1 example", 76.1, 92.0, 68.0),
    ("TF-IDF + logistic regression", "5,842 stories", 68.2, 95.4, 56.9),
    ("RoBERTa, fine-tuned", "5,842 stories", 94.6, 93.2, 96.6),
    ("LabelFusion (RoBERTa + GPT-5-nano)", "5,842 stories", 96.0, 96.7, 96.1),
]


def load_predictions(run_dir: Path, model: str) -> tuple[dict[str, set[str]], set[str]]:
    """-> ({article_id: predicted topics} for fully successful articles, {article_ids with a failed decision})."""
    final = sorted((run_dir / "results" / f"model={model}").glob("*.parquet"))
    df = pl.read_parquet(final) if final else results.latest_attempts(results.read_work(run_dir, model))
    failed = set(df.filter(pl.col("status") != "SUCCESS")["article_id"].to_list())
    pred: dict[str, set[str]] = {}
    for aid, child, pos in df.filter(pl.col("status") == "SUCCESS").select(
            "article_id", "child_id", "selected_positive").iter_rows():
        s = pred.setdefault(aid, set())
        if pos and child.startswith(PREFIX):
            s.add(child[len(PREFIX):])
    return {a: p for a, p in pred.items() if a not in failed}, failed


def metrics(gold: list[set[str]], pred: list[set[str]], labels=TOP10) -> dict:
    div = lambda a, b: a / b if b else 0.0  # noqa: E731
    per, T = {}, [0, 0, 0]
    for t in labels:
        tp = sum(t in g and t in p for g, p in zip(gold, pred))
        fp = sum(t not in g and t in p for g, p in zip(gold, pred))
        fn = sum(t in g and t not in p for g, p in zip(gold, pred))
        pr, rc = div(tp, tp + fp), div(tp, tp + fn)
        per[t] = {"support": tp + fn, "predicted": tp + fp, "precision": pr, "recall": rc,
                  "f1": div(2 * pr * rc, pr + rc)}
        T = [T[0] + tp, T[1] + fp, T[2] + fn]
    n = len(gold)
    mp, mr = div(T[0], T[0] + T[1]), div(T[0], T[0] + T[2])
    lab = set(labels)
    return {
        "stories": n, "per_topic": per,
        "macro": {k: sum(v[k] for v in per.values()) / len(labels) for k in ("precision", "recall", "f1")},
        "micro": {"precision": mp, "recall": mr, "f1": div(2 * mp * mr, mp + mr)},
        "subset_accuracy": div(sum((g & lab) == (p & lab) for g, p in zip(gold, pred)), n),
        "hamming_accuracy": div(n * len(labels) - T[1] - T[2], n * len(labels)),
    }


def score(run_dir: Path, model: str, corpus_dir: Path) -> dict:
    prov = pl.read_parquet(corpus_dir / "provenance.parquet")
    to_canon = dict(pl.read_parquet(corpus_dir / "clusters.parquet")
                    .select("article_id", "canonical_article_id").iter_rows())
    pred, failed = load_predictions(run_dir, model)
    gold, got, missing, failed_rows = [], [], 0, 0
    for aid, label in prov.select("article_id", "label").iter_rows():
        c = to_canon.get(aid, aid)
        if c in failed:
            failed_rows += 1
        elif c not in pred:
            missing += 1
        else:
            gold.append(set(label.split(";")))
            got.append(pred[c])
    return {"source_rows": prov.height, "failed_rows": failed_rows, "missing_rows": missing,
            **metrics(gold, got)}


def render(s: dict, name: str, published: bool = True) -> str:
    pc = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    out = [f"scored {s['stories']:,} of {s["source_rows"]:,} stories"
           f" ({s['missing_rows']:,} without results, {s['failed_rows']:,} with a failed decision)"]
    if published and s["stories"] < s["source_rows"]:
        out.append("  ! incomplete: not comparable with the published numbers until all 2,545 are scored")
    out += ["", f"  {'topic':10s} {'support':>7s} {'predicted':>9s}   prec  recall     F1"]
    for t, v in s["per_topic"].items():
        out.append(f"  {t:10s} {v['support']:7d} {v['predicted']:9d}  {pc(v['precision'])}  {pc(v['recall'])}  {pc(v['f1'])}")
    m, u = s["macro"], s["micro"]
    out += ["", f"  macro      F1 {pc(m['f1'])}   precision {pc(m['precision'])}   recall {pc(m['recall'])}",
            f"  micro      F1 {pc(u['f1'])}   precision {pc(u['precision'])}   recall {pc(u['recall'])}",
            f"  subset accuracy {pc(s['subset_accuracy'])}   Hamming accuracy {pc(s['hamming_accuracy'])}"]
    if not published:  # development (training-split) runs: no published numbers to compare against
        return "\n".join(out)
    out += ["", "comparison, macro-averaged % on the same test set (LabelFusion, FNP 2026, Table 1):",
            f"  {'model':38s} {'training data':15s}    F1   prec  recall",
            f"  {name + ' (this run, zero-shot)':38s} {'none':15s} {pc(m['f1'])}  {pc(m['precision'])}  {pc(m['recall'])}"]
    for model, data, f1, p, r in PUBLISHED:
        out.append(f"  {model:38s} {data:15s} {f1:5.1f}  {p:5.1f}  {r:5.1f}")
    out.append("  (like-for-like: the zero-shot rows; fine-tuned rows were trained on the 5,842-story training split)")
    return "\n".join(out)


def main() -> None:
    from classifier.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/native_reuters_top10.yaml")
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("-m", "--model", default=None, help="default: the run's only model")
    ap.add_argument("--json", default=None, help="also write the scores as JSON")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run = Path(a.run)
    model = a.model or next(iter(json.loads((run / "manifest.json").read_text())["models"]))
    s = score(run, model, Path(cfg.corpus.directory))
    print(f"{cfg.benchmark.name} · {model} · {run.name}\n")
    print(render(s, model, published=cfg.benchmark.name == "native_reuters21578_top10"))
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
