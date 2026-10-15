"""CFPB Consumer Complaint Database (consumerfinance.gov, public: "freely available for anyone to use").

The database updates daily, so the pin is the SHA-256 of the exact zip snapshot downloaded (its
date is the config `revision`). Only complaints with a consumer narrative are used; the product label
is NOT shown to the model (headline empty, narrative only).

Sampling: per product group, the N complaints with the lowest SHA-256 of their Complaint ID, which is
deterministic and independent of file order. Expectations (spec §0.14) map products onto lending
leaves; groups without a clear mapping carry none.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path

from .common import GSIB, Builder, download, hash_files

URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
ZIP = "complaints.csv.zip"
PER_GROUP = 1500
LEND, CONS, MORT = "business.lending", "lending.consumer", "lending.residential_mortgage"
EXPECT = {
    "mortgage": {GSIB: 1, LEND: 1, MORT: 1, CONS: 0},
    "credit_card": {GSIB: 1, LEND: 1, CONS: 1, MORT: 0},
    "vehicle_loan": {GSIB: 1, LEND: 1, CONS: 1, MORT: 0},
    "personal_loan": {GSIB: 1, LEND: 1, CONS: 1, MORT: 0},
    "deposit_account": {CONS: 0, MORT: 0},
    "money_transfer": {CONS: 0, MORT: 0},
    "student_loan": {},
    "credit_reporting": {},
    "debt_collection": {},
}
US_GSIB_MARKERS = ("JPMORGAN", "BANK OF AMERICA", "CITIBANK", "CITIGROUP", "WELLS FARGO", "GOLDMAN SACHS",
                   "MORGAN STANLEY", "BANK OF NEW YORK MELLON", "STATE STREET")


def group_of(product: str, sub: str) -> str | None:
    p = (product or "").lower()
    if p == "mortgage":
        return "mortgage"
    if p.startswith("credit card"):
        return None if "prepaid" in (sub or "").lower() else "credit_card"
    if p.startswith("vehicle loan"):
        return "vehicle_loan"
    if p.startswith("payday loan") or p == "consumer loan":
        return "personal_loan"
    if p.startswith("checking or savings") or p == "bank account or service":
        return "deposit_account"
    if p.startswith("money transfer"):
        return "money_transfer"
    if p == "student loan":
        return "student_loan"
    if p.startswith("credit reporting"):
        return "credit_reporting"
    if p == "debt collection":
        return "debt_collection"
    return None


def fetch(ds) -> Path:
    local = Path(ds.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    if not (local / ZIP).exists():
        download(URL, local / ZIP)
    return local


def files_sha256(local: Path) -> str:
    return hash_files(local, [local / ZIP])


def ingest(local: Path, ds):
    csv.field_size_limit(10**9)
    best: dict[str, list[tuple[str, dict]]] = {g: [] for g in EXPECT}
    with zipfile.ZipFile(local / ZIP) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
                text = (r.get("Consumer complaint narrative") or "").strip()
                g = group_of(r.get("Product", ""), r.get("Sub-product", ""))
                if not text or g is None:
                    continue
                key = hashlib.sha256(r["Complaint ID"].encode()).hexdigest()
                best[g].append((key, r))
                if len(best[g]) > 4 * PER_GROUP:  # keep memory bounded: retain the lowest keys
                    best[g] = sorted(best[g], key=lambda x: x[0])[:PER_GROUP]
    b = Builder("cfpb", "CFPB")
    for g in sorted(best):
        for _, r in sorted(best[g], key=lambda x: x[0])[:PER_GROUP]:
            company = r.get("Company", "")
            b.add(r["Consumer complaint narrative"], date=r.get("Date received", ""), source_id=r["Complaint ID"],
                  label=f"{r.get('Product', '')} / {r.get('Sub-product', '')}", group=g, expect=EXPECT[g],
                  extra={"company": company, "us_gsib": any(m in company.upper() for m in US_GSIB_MARKERS),
                         "issue": r.get("Issue", "")})
    return b.frames()
