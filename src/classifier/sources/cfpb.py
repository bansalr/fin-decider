"""CFPB Consumer Complaint Database narratives, from a pre-removal archive.

The CFPB stopped publishing complaint narratives on 2026-08-14 (they now sit in its FOIA reading
room), so the live bulk file and API no longer carry the text. This source reads the CC0 mirror
`BEE-spoke-data/consumer-finance-complaints` on Hugging Face (4.7M complaints, original CFPB columns),
pinned to a commit plus the hashes of its parquet files. It is a third-party copy of public-domain
data that the CFPB has since withdrawn as "unverified"; use it for internal benchmarking.

Only complaints with a narrative are used; the product label is NOT shown to the model (headline
empty, narrative only). Sampling: per product group, the N complaints with the lowest SHA-256 of
their Complaint ID (deterministic, independent of file order). Expectations (spec §0.14) map
products onto lending leaves; groups without a clear mapping carry none.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from .common import GSIB, Builder, hash_files, hf_fetch

FILES = [f"data/train-{i:05d}-of-00007.parquet" for i in range(7)]
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
    return hf_fetch(ds.repository, ds.revision, FILES, ds.local_dir)


def files_sha256(local: Path) -> str:
    return hash_files(local, [local / f for f in FILES])


def ingest(local: Path, ds):
    cols = ["Date received", "Product", "Sub-product", "Issue", "Consumer complaint narrative", "Company",
            "Complaint ID"]
    df = (pl.scan_parquet([str(local / f) for f in FILES]).select(cols)
          .filter(pl.col("Consumer complaint narrative").is_not_null()
                  & (pl.col("Consumer complaint narrative").str.strip_chars() != ""))
          .collect())
    df = df.with_columns(
        pl.struct(["Product", "Sub-product"]).map_elements(lambda r: group_of(r["Product"], r["Sub-product"]),
                                                            return_dtype=pl.String).alias("_group"),
        pl.col("Complaint ID").cast(pl.String).map_elements(lambda c: hashlib.sha256(c.encode()).hexdigest(),
                                                            return_dtype=pl.String).alias("_key"),
    ).filter(pl.col("_group").is_not_null())
    best = {g: [r for r in df.filter(pl.col("_group") == g).sort("_key").head(PER_GROUP).iter_rows(named=True)]
            for g in EXPECT}
    b = Builder("cfpb", "CFPB")
    for g in sorted(best):
        for r in best[g]:
            company = r.get("Company", "")
            b.add(r["Consumer complaint narrative"], date=str(r.get("Date received") or ""), source_id=str(r["Complaint ID"]),
                  label=f"{r.get('Product', '')} / {r.get('Sub-product', '')}", group=g, expect=EXPECT[g],
                  extra={"company": company, "us_gsib": any(m in company.upper() for m in US_GSIB_MARKERS),
                         "issue": r.get("Issue", "")})
    return b.frames()
