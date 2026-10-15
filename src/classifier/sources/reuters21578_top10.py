"""Reuters-21578 top-10 topics, ModApte test split: the native-task benchmark (native_benchmarks/reuters_top10).

Same pinned archive as `reuters21578` (identical files_sha256), different selection: the standard "R(10)"
test set, i.e. LEWISSPLIT="TEST", TOPICS="YES" stories carrying at least one of the ten most frequent
topics (2,545 stories, as in LabelFusion, FNP 2026). Unlike the `reuters21578` source, stories without a
<BODY> (title-only briefs) are kept, so the set matches the published one.

Provenance `label` holds the gold top-10 topics (";"-joined, in TOP10 order); `extra.all_topics` keeps the
full topic list. `expect` is empty: this corpus is scored on its own labels by scripts/score_native_reuters.py,
not against the G-SIB ontology.
"""

from __future__ import annotations

import re
from pathlib import Path

from .common import Builder
from .reuters21578 import ATTR, DOC, _sgm_texts, _tag, fetch, files_sha256  # noqa: F401  (same archive)

TOP10 = ("earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn")
_TAGS = re.compile(r"<[^>]+>")


def _text(body: str) -> tuple[str, str]:
    """-> (headline, text). Unprocessed stories (TEXT TYPE="UNPROC") have no TITLE/BODY tags."""
    title, text = _tag(body, "TITLE"), _tag(body, "BODY")
    if not title and not text:
        raw = re.search(r"<TEXT[^>]*>(.*?)</TEXT>", body, re.S)
        import html

        text = html.unescape(_TAGS.sub(" ", raw.group(1))).strip() if raw else ""
    return title, text


def ingest(local: Path, ds):
    b = Builder("rtr10", "Reuters-21578 R(10) ModApte test")
    for sgm in _sgm_texts(local):
        for attrs, body in DOC.findall(sgm):
            a = dict(ATTR.findall(attrs))
            if a.get("TOPICS") != "YES" or a.get("LEWISSPLIT") != "TEST":
                continue
            topics = re.findall(r"<D>(.*?)</D>", _tag(body, "TOPICS"))
            gold = [t for t in TOP10 if t in topics]
            if not gold:
                continue
            headline, text = _text(body)
            b.add(text, headline=headline, date=_tag(body, "DATE"), source_id=a.get("NEWID", ""), split="test",
                  label=";".join(gold), group=gold[0], expect={}, extra={"all_topics": topics})
    return b.frames()
