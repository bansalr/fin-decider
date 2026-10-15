"""Reuters-21578 (UCI ML Repository #137, CC BY 4.0). 1987 Reuters newswire, ModApte split
(LEWISSPLIT TRAIN/TEST with TOPICS="YES"). No ontology expectations: its topics are commodities
and macro, so results are reported per Reuters topic (group = first topic) for comparison only.
"""

from __future__ import annotations

import html
import re
import tarfile
import zipfile
from pathlib import Path

from .common import Builder, download, hash_files

URL = "https://archive.ics.uci.edu/static/public/137/reuters+21578+text+categorization+collection.zip"
ZIP = "reuters21578.zip"
DOC = re.compile(r"<REUTERS([^>]*)>(.*?)</REUTERS>", re.S)
ATTR = re.compile(r'(\w+)="([^"]*)"')


def fetch(ds) -> Path:
    local = Path(ds.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    z = local / ZIP
    if not z.exists():
        download(URL, z)
    return local


def files_sha256(local: Path) -> str:
    return hash_files(local, [local / ZIP])


def _tag(body: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", body, re.S)
    return html.unescape(m.group(1)).strip() if m else ""


def _sgm_texts(local: Path):
    with zipfile.ZipFile(local / ZIP) as zf:
        inner = [n for n in zf.namelist() if n.endswith(".tar.gz")]
        if inner:
            import io

            with tarfile.open(fileobj=io.BytesIO(zf.read(inner[0])), mode="r:gz") as tf:
                for m in sorted(tf.getmembers(), key=lambda m: m.name):
                    if m.name.endswith(".sgm"):
                        yield tf.extractfile(m).read().decode("latin-1")
        else:
            for n in sorted(zf.namelist()):
                if n.endswith(".sgm"):
                    yield zf.read(n).decode("latin-1")


def ingest(local: Path, ds):
    b = Builder("rtr", "Reuters-21578")
    for sgm in _sgm_texts(local):
        for attrs, body in DOC.findall(sgm):
            a = dict(ATTR.findall(attrs))
            if a.get("TOPICS") != "YES" or a.get("LEWISSPLIT") not in ("TRAIN", "TEST"):
                continue
            topics = re.findall(r"<D>(.*?)</D>", _tag(body, "TOPICS"))
            text = _tag(body, "BODY")
            if not text or not topics:
                continue
            b.add(text, headline=_tag(body, "TITLE"), date=_tag(body, "DATE"), source_id=a.get("NEWID", ""),
                  split=a["LEWISSPLIT"].lower(), label=";".join(topics), group=topics[0], expect={})
    return b.frames()
