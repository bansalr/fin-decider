"""Build a blinded, stratified labelling queue from the articles several models scored.

Strata (by G-SIB activity votes of the source runs):
  split      models disagree        -> oversampled; most informative
  all_yes    every model said yes
  all_no     every model said no    -> catches shared errors
Each row's sampling weight = stratum size / sampled from stratum, so scores can be
reweighted to the population of commonly-scored articles.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import polars as pl

from classifier import results
from classifier.ontology import Ontology

GSIB = "relevance.gsib_activity"
LABEL_COLUMNS = [
    "gsib_activity", "market_context_only", "neither",           # y / n
    "markets", "lending", "treasury_alm",                        # y / n, only if gsib_activity = y
    "leaves",                                                    # leaf codes, ';'-separated
    "confidence", "notes", "labeller",
]


def positives(run_dir: Path, model: str) -> dict[str, set[str]]:
    final = sorted((run_dir / "results" / f"model={model}").glob("*.parquet"))
    df = pl.read_parquet(final) if final else results.latest_attempts(results.read_work(run_dir, model))
    out = {a: set() for a in df["article_id"].unique().to_list()}
    p = df.filter((pl.col("status") == "SUCCESS") & pl.col("selected_positive"))
    for a, c in p.select("article_id", "child_id").iter_rows():
        out[a].add(c)
    return out


def build(runs: dict[str, Path], corpus: pl.DataFrame, sizes: dict[str, int], seed: int) -> tuple[list[dict], dict]:
    pos = {m: positives(d, m) for m, d in runs.items()}
    common = sorted(set.intersection(*[set(p) for p in pos.values()]))
    strata: dict[str, list[str]] = {"split": [], "all_yes": [], "all_no": []}
    for a in common:
        votes = sum(GSIB in pos[m][a] for m in runs)
        strata["all_yes" if votes == len(runs) else "all_no" if votes == 0 else "split"].append(a)
    rng = random.Random(seed)
    picked = []
    for s, ids in strata.items():
        k = min(sizes[s], len(ids))
        for a in rng.sample(ids, k):
            picked.append({"article_id": a, "stratum": s, "weight": len(ids) / k})
    rng.shuffle(picked)
    text = {r["article_id"]: r for r in corpus.filter(pl.col("article_id").is_in([p["article_id"] for p in picked]))
            .select("article_id", "headline", "date", "url", "article").iter_rows(named=True)}
    for i, p in enumerate(picked, 1):
        p["label_id"] = f"L{i:04d}"
        p.update({k: text[p["article_id"]][k] for k in ("headline", "date", "url", "article")})
    meta = {"source_runs": {m: str(d) for m, d in runs.items()}, "seed": seed, "common_articles": len(common),
            "strata_population": {s: len(v) for s, v in strata.items()}, "strata_sampled": sizes}
    return picked, meta


def write(picked: list[dict], meta: dict, out_dir: Path, name: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    queue = out_dir / f"{name}.csv"           # what labellers see: blinded, no strata or model output
    key = out_dir / f"{name}_key.csv"         # kept back from labellers
    with open(queue, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label_id", "headline", "date", "url", "article"] + LABEL_COLUMNS)
        for p in picked:
            w.writerow([p["label_id"], p["headline"], p["date"], p["url"], p["article"]] + [""] * len(LABEL_COLUMNS))
    with open(key, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label_id", "article_id", "stratum", "weight"])
        for p in picked:
            w.writerow([p["label_id"], p["article_id"], p["stratum"], f"{p['weight']:.6f}"])
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"queue": queue, "key": key}


def guide(onto: Ontology) -> str:
    """Labelling guide from the ontology's own questions and criteria."""
    lines = [
        f"# Labelling guide (ontology {onto.version})",
        "",
        "Label each article from its text alone. Model outputs are deliberately hidden.",
        "",
        "## Columns",
        "",
        "- `gsib_activity`, `market_context_only`, `neither`: `y` or `n` each. More than one can be `y`"
        " (an article can describe bank activity *and* market context).",
        "- `markets`, `lending`, `treasury_alm`: `y`/`n`, only when `gsib_activity` is `y`; otherwise leave blank.",
        "- `leaves`: the codes of every activity category below that the article materially describes,"
        " separated by `;`. Leave blank if none.",
        "- `confidence`: 1 (unsure) to 3 (certain). `notes`: anything ambiguous. `labeller`: your initials.",
        "",
        "Rule of thumb: the activity must actually happen in the article (a deal, trade, loan, facility,"
        " or a bank business line's results), not just be mentioned. Central banks, bank analysts'"
        " forecasts, and price moves alone are market context, not G-SIB activity.",
        "",
        "## Top level",
        "",
    ]
    for c in onto.children(onto.roots[0]):
        lines += [f"**{c.id.split('.', 1)[1]}**: {c.yes_no_question}", f"- yes: {c.criteria_dict['true']}",
                  f"- no: {c.criteria_dict['false']}", ""]
    lines += ["## Activity categories (codes for `leaves`)", ""]
    for nid, n in onto.nodes.items():
        if n.is_leaf and n.parent != onto.roots[0]:
            lines += [f"- `{nid}`: **{n.name}**. {n.yes_no_question}"]
    return "\n".join(lines) + "\n"
