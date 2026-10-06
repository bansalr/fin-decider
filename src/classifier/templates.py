"""Versioned decision templates (spec §19). One template is shared by Jev and Laya
so both receive byte-identical questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .ontology import OntologyNode
from .provenance import sha256_file

_SECTION_RE = re.compile(r"^\[([a-z.]+)\]\s*$", re.M)
_FIELD_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class BooleanTemplate:
    instructions: str
    criteria_true: str
    criteria_false: str
    sha256: str

    def render(self, parent: OntologyNode, child: OntologyNode) -> dict:
        values = {
            "parent_question": parent.question or "",
            "name": child.name,
            "definition": child.definition,
            "includes": "; ".join(child.includes),
            "excludes": "; ".join(child.excludes),
            "synonyms": "; ".join(child.synonyms),
        }
        lines = []
        for line in self.instructions.splitlines():
            fields = _FIELD_RE.findall(line)
            # A line whose placeholders are all empty is dropped (e.g. no synonyms).
            if fields and all(not values[f] for f in fields):
                continue
            lines.append(_FIELD_RE.sub(lambda m: values[m.group(1)], line))
        return {
            "type": "boolean",
            "instructions": "\n".join(lines),
            "criteria": {"true": self.criteria_true, "false": self.criteria_false},
        }


def load_template(path: str | Path, expected_sha256: str) -> BooleanTemplate:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"template {path}: expected sha256 {expected_sha256}, got {digest}")
    text = Path(path).read_text(encoding="utf-8")
    parts = _SECTION_RE.split(text)
    sections = {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}
    missing = {"instructions", "criteria.true", "criteria.false"} - sections.keys()
    if missing:
        raise ValueError(f"template {path}: missing sections {sorted(missing)}")
    return BooleanTemplate(
        instructions=sections["instructions"],
        criteria_true=sections["criteria.true"].strip(),
        criteria_false=sections["criteria.false"].strip(),
        sha256=digest,
    )
