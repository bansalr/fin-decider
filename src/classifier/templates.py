"""Versioned templates (spec §19).

`boolean` templates render one decision question per child; one boolean template is
shared by every decision model, so they receive byte-identical questions. `label`
templates render one label string per child, for encoder classifiers.

Placeholders: {parent_question} {name} {definition} {includes} {excludes}
{synonyms} {yes_no_question} {criteria_true} {criteria_false}. A line whose
placeholders all render empty is dropped. Ontology-backed placeholders that a
template uses must be present on every node it renders (checked up front).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .ontology import Ontology, OntologyNode
from .provenance import sha256_file

_SECTION_RE = re.compile(r"^\[([a-z_.]+)\]\s*$", re.M)
_FIELD_RE = re.compile(r"\{(\w+)\}")

# Placeholders that must have a value on the node (not just be possibly-empty lists).
REQUIRED_NODE_FIELDS = {"yes_no_question", "criteria_true", "criteria_false"}
SECTIONS = {"boolean": ({"instructions"}, {"criteria.true", "criteria.false"}), "label": ({"label"}, set())}


class TemplateError(ValueError):
    pass


def _values(parent: OntologyNode | None, child: OntologyNode) -> dict[str, str]:
    crit = child.criteria_dict
    return {
        "parent_question": (parent.question if parent else None) or "",
        "name": child.name,
        "definition": child.definition,
        "includes": "; ".join(child.includes),
        "excludes": "; ".join(child.excludes),
        "synonyms": "; ".join(child.synonyms),
        "yes_no_question": child.yes_no_question or "",
        "criteria_true": crit.get("true", ""),
        "criteria_false": crit.get("false", ""),
    }


def _fill(text: str, values: dict[str, str]) -> str:
    lines = []
    for line in text.splitlines():
        fields = _FIELD_RE.findall(line)
        if fields and all(not values[f] for f in fields):
            continue
        lines.append(_FIELD_RE.sub(lambda m: values[m.group(1)], line))
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class Template:
    kind: str
    sections: dict[str, str]
    sha256: str

    @property
    def fields(self) -> set[str]:
        return {f for body in self.sections.values() for f in _FIELD_RE.findall(body)}

    def check_ontology(self, onto: Ontology) -> None:
        """Every non-root node must supply each required field this template uses."""
        needed = self.fields & REQUIRED_NODE_FIELDS
        bad = []
        for node in onto.nodes.values():
            if node.parent is None:
                continue
            vals = _values(None, node)
            missing = [f for f in needed if not vals[f]]
            if missing:
                bad.append(f"{node.id}: {missing}")
        if bad:
            raise TemplateError(f"ontology lacks fields used by template: {bad[:5]}{' ...' if len(bad) > 5 else ''}")

    def render(self, parent: OntologyNode, child: OntologyNode) -> dict:
        """Boolean decision question for one child."""
        if self.kind != "boolean":
            raise TemplateError("render() is for boolean templates")
        v = _values(parent, child)
        q = {"type": "boolean", "instructions": _fill(self.sections["instructions"], v)}
        t, f = _fill(self.sections["criteria.true"], v), _fill(self.sections["criteria.false"], v)
        if t or f:
            q["criteria"] = {"true": t, "false": f}
        return q

    def render_label(self, child: OntologyNode) -> str:
        if self.kind != "label":
            raise TemplateError("render_label() is for label templates")
        return _fill(self.sections["label"], _values(None, child))


# Backwards-compatible name used by adapters and tests.
BooleanTemplate = Template


def load_template(path: str | Path, expected_sha256: str, kind: str = "boolean") -> Template:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise TemplateError(f"template {path}: expected sha256 {expected_sha256}, got {digest}")
    text = Path(path).read_text(encoding="utf-8")
    parts = _SECTION_RE.split(text)
    sections = {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}
    required, optional = SECTIONS[kind]
    missing = required - sections.keys()
    if missing:
        raise TemplateError(f"template {path}: missing sections {sorted(missing)}")
    unknown = sections.keys() - required - optional
    if unknown:
        raise TemplateError(f"template {path}: unknown sections {sorted(unknown)}")
    for s in optional:
        sections.setdefault(s, "")
    bad = {f for body in sections.values() for f in _FIELD_RE.findall(body)} - set(_values(None, _DUMMY))
    if bad:
        raise TemplateError(f"template {path}: unknown placeholders {sorted(bad)}")
    return Template(kind=kind, sections=sections, sha256=digest)


_DUMMY = OntologyNode(
    id="x", name="x", type="leaf", definition="x", question=None, includes=(), excludes=(), synonyms=(),
    parent=None, children=(),
)
