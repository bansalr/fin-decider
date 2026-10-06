"""Model input construction (spec §7). One canonical mode: headline_plus_article."""

from __future__ import annotations

from .adapters.base import ArticleInput
from .config import Input
from .provenance import sha256_text
from .textnorm import normalize


def build_input(article_id: str, headline: str, body: str, cfg: Input) -> ArticleInput:
    n = cfg.normalization
    norm = lambda t: normalize(t, unicode=n.unicode, strip_html=n.strip_html, collapse_whitespace=n.collapse_whitespace)  # noqa: E731
    parts = [p for p in (norm(headline), norm(body)) if p]
    full = cfg.separator.join(parts)
    text = full[: cfg.max_chars]  # truncation_strategy == "head"
    return ArticleInput(
        article_id=article_id,
        text=text,
        input_hash=sha256_text(text),
        original_chars=len(full),
        submitted_chars=len(text),
        truncated=len(text) < len(full),
        truncation_strategy=cfg.truncation_strategy,
    )
