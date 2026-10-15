"""EDGAR-CORPUS (huggingface.co/datasets/eloukas/edgar-corpus, Apache 2.0): 10-K Item 7 (MD&A)
of large US banks (the 8 G-SIBs plus 10 large regionals) and 8 large non-financial companies.

Several G-SIBs (e.g. JPMorgan, Wells Fargo, BNY Mellon) incorporate MD&A by reference to the annual
report, leaving a stub in the 10-K; filings whose Item 7 is under MIN_MDA_CHARS are dropped and
counted. The regional banks keep the positive control populated; each passage records g_sib.

Each year's JSONL (about 1.8 GB) is streamed once and only the target companies are kept
(`filtered_<year>.jsonl`, hashed for the pin). MD&A text is split into passages on paragraph
boundaries. No per-passage expectation (a bank MD&A passage may discuss accounting policy): results are
reported per group, bank vs control (spec §0.14).
"""

from __future__ import annotations

import json
from pathlib import Path

import os
import time

from .common import Builder, hash_files, split_passages, stream_lines

YEARS = (2018, 2019, 2020)
MIN_MDA_CHARS = 5000
SPLITS = ("train", "validate", "test")
PASSAGE_CHARS = 6000
GSIBS = {"19617": "JPMorgan Chase", "70858": "Bank of America", "831001": "Citigroup", "72971": "Wells Fargo",
         "886982": "Goldman Sachs", "895421": "Morgan Stanley", "1390777": "BNY Mellon", "93751": "State Street"}
REGIONALS = {"713676": "PNC", "36104": "U.S. Bancorp", "92230": "Truist", "927628": "Capital One",
             "35527": "Fifth Third", "91576": "KeyCorp", "1281761": "Regions", "36270": "M&T Bank",
             "49196": "Huntington", "759944": "Citizens Financial"}
BANKS = GSIBS | REGIONALS
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
        for attempt in range(1, 6):  # a 1.8 GB stream can drop; restart the year from scratch
            try:
                with open(tmp, "w", encoding="utf-8") as w:
                    for s in SPLITS:
                        url = f"https://huggingface.co/datasets/{ds.repository}/resolve/{ds.revision}/{y}/{s}.jsonl"
                        for line in stream_lines(url, token=os.environ.get("HF_TOKEN")):
                            if '"cik"' not in line:
                                continue
                            rec = json.loads(line)
                            if _cik(rec.get("cik")) in TARGETS:
                                w.write(json.dumps({k: rec.get(k) for k in ("filename", "cik", "year", "section_7")})
                                        + "\n")
                break
            except Exception as e:  # noqa: BLE001 - network errors of any kind
                print(f"edgar {y}: attempt {attempt} failed ({type(e).__name__}); retrying", flush=True)
                time.sleep(30 * attempt)
        else:
            raise RuntimeError(f"edgar {y}: streaming failed after 5 attempts")
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
            if len(rec.get("section_7") or "") < MIN_MDA_CHARS:  # MD&A incorporated by reference: stub only
                continue
            for i, passage in enumerate(split_passages(rec.get("section_7") or "", PASSAGE_CHARS)):
                if len(passage) < 200:
                    continue
                b.add(passage, date=str(rec.get("year") or y), source_id=f"{rec['filename']}#{i}",
                      label=name, group=("bank:" if is_bank else "control:") + name, expect={},
                      extra={"cik": cik, "bank": is_bank, "g_sib": cik in GSIBS})
    return b.frames()
