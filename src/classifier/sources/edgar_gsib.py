"""EDGAR-CORPUS (huggingface.co/datasets/eloukas/edgar-corpus, Apache 2.0): 10-K Item 7 (MD&A)
of the 8 US G-SIBs plus 8 large non-financial companies as controls.

Each year's JSONL (about 1.8 GB) is streamed once and only the target companies are kept
(`filtered_<year>.jsonl`, hashed for the pin). MD&A text is split into passages on paragraph
boundaries. No per-passage expectation (a bank MD&A passage may discuss accounting policy): results are
reported per group, bank vs control (spec §0.14).
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import Builder, hash_files, split_passages, stream_lines

YEARS = (2019, 2020)
SPLITS = ("train", "validate", "test")
PASSAGE_CHARS = 6000
BANKS = {"19617": "JPMorgan Chase", "70858": "Bank of America", "831001": "Citigroup", "72971": "Wells Fargo",
         "886982": "Goldman Sachs", "895421": "Morgan Stanley", "1390777": "BNY Mellon", "93751": "State Street"}
CONTROLS = {"320193": "Apple", "34088": "Exxon Mobil", "104169": "Walmart", "200406": "Johnson & Johnson",
            "80424": "Procter & Gamble", "21344": "Coca-Cola", "789019": "Microsoft", "78003": "Pfizer"}
TARGETS = BANKS | CONTROLS


def _cik(v) -> str:
    return str(v).lstrip("0")


def fetch(ds) -> Path:
    local = Path(ds.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        out = local / f"filtered_{y}.jsonl"
        if out.exists():
            continue
        tmp = out.with_suffix(".part")
        with open(tmp, "w", encoding="utf-8") as w:
            for s in SPLITS:
                url = f"https://huggingface.co/datasets/{ds.repository}/resolve/{ds.revision}/{y}/{s}.jsonl"
                for line in stream_lines(url):
                    if '"cik"' not in line:
                        continue
                    rec = json.loads(line)
                    if _cik(rec.get("cik")) in TARGETS:
                        w.write(json.dumps({k: rec.get(k) for k in ("filename", "cik", "year", "section_7")}) + "\n")
        tmp.rename(out)
    return local


def files_sha256(local: Path) -> str:
    return hash_files(local, [local / f"filtered_{y}.jsonl" for y in YEARS])


def ingest(local: Path, ds):
    b = Builder("edgar", "EDGAR-CORPUS")
    for y in YEARS:
        for line in open(local / f"filtered_{y}.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            cik = _cik(rec["cik"])
            is_bank = cik in BANKS
            name = TARGETS[cik]
            for i, passage in enumerate(split_passages(rec.get("section_7") or "", PASSAGE_CHARS)):
                if len(passage) < 200:
                    continue
                b.add(passage, date=str(rec.get("year") or y), source_id=f"{rec['filename']}#{i}",
                      label=name, group=("bank:" if is_bank else "control:") + name, expect={},
                      extra={"cik": cik, "bank": is_bank})
    return b.frames()
