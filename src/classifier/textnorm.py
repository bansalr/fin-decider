"""Versioned text normalization shared by identity, dedupe, and input construction."""

from __future__ import annotations

import html
import re
import unicodedata

NORMALIZATION_VERSION = "norm_v1"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def normalize(text: str | None, *, unicode: str = "NFC", strip_html: bool = True, collapse_whitespace: bool = True) -> str:
    if text is None:
        return ""
    out = unicodedata.normalize(unicode, str(text))
    if strip_html:
        out = html.unescape(_TAG_RE.sub(" ", out))
    if collapse_whitespace:
        out = _WS_RE.sub(" ", out)
    return out.strip()
