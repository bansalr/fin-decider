"""Ontology loading and validation (spec §9, §10, §22).

The ontology file is authoritative and is read in place; this module never
rewrites it. Validation runs before any inference call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .provenance import sha256_file, sha256_obj

NODE_TYPES = {"multi_label", "single_choice", "leaf"}
REQUIRED_FIELDS = ("name", "type", "definition", "includes", "excludes", "synonyms")


class OntologyError(ValueError):
    pass


@dataclass(frozen=True)
class OntologyNode:
    id: str
    name: str
    type: str
    definition: str
    question: str | None
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    synonyms: tuple[str, ...]
    parent: str | None
    children: tuple[str, ...]
    basel: Any = None
    yes_no_question: str | None = None  # v1.1+
    criteria: tuple[tuple[str, str], ...] = ()  # v1.1+: (("false", ...), ("true", ...))

    @property
    def criteria_dict(self) -> dict[str, str]:
        return dict(self.criteria)

    @property
    def is_leaf(self) -> bool:
        return self.type == "leaf"

    def semantic_ir(self) -> dict[str, Any]:
        """The shared representation every adapter receives (spec §18)."""
        ir = {
            "id": self.id,
            "name": self.name,
            "definition": self.definition,
            "includes": list(self.includes),
            "excludes": list(self.excludes),
            "synonyms": list(self.synonyms),
        }
        if self.yes_no_question is not None:
            ir["yes_no_question"] = self.yes_no_question
        if self.criteria:
            ir["criteria"] = self.criteria_dict
        return ir


@dataclass
class Ontology:
    id: str
    version: str
    roots: tuple[str, ...]
    nodes: dict[str, OntologyNode]
    sha256: str
    raw: dict[str, Any] = field(repr=False)

    def node(self, node_id: str) -> OntologyNode:
        return self.nodes[node_id]

    def children(self, node_id: str) -> list[OntologyNode]:
        return [self.nodes[c] for c in self.nodes[node_id].children]

    def depth(self, node_id: str) -> int:
        d, cur = 0, self.nodes[node_id]
        while cur.parent is not None:
            d += 1
            cur = self.nodes[cur.parent]
        return d

    def node_hash(self, node_id: str) -> str:
        """Hash of everything a decision at this node depends on: the node and its children's IR."""
        node = self.nodes[node_id]
        return sha256_obj(
            {
                "node": node.semantic_ir() | {"type": node.type, "question": node.question},
                "children": [c.semantic_ir() | {"type": c.type} for c in self.children(node_id)],
            }
        )


def load_ontology(path: str | Path, expected_sha256: str | None = None) -> Ontology:
    path = Path(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise OntologyError(f"ontology hash mismatch: expected {expected_sha256}, got {digest}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return build_ontology(raw, digest)


def build_ontology(raw: dict[str, Any], digest: str) -> Ontology:
    meta = raw.get("ontology") or {}
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, dict) or not raw_nodes:
        raise OntologyError("ontology has no nodes")
    for key in ("id", "version", "roots"):
        if key not in meta:
            raise OntologyError(f"ontology metadata missing '{key}'")

    nodes: dict[str, OntologyNode] = {}
    for node_id, spec in raw_nodes.items():
        for f in REQUIRED_FIELDS:
            if f not in spec:
                raise OntologyError(f"{node_id}: missing required field '{f}'")
        if spec["type"] not in NODE_TYPES:
            raise OntologyError(f"{node_id}: unknown node type '{spec['type']}'")
        nodes[node_id] = OntologyNode(
            id=node_id,
            name=spec["name"],
            type=spec["type"],
            definition=spec["definition"],
            question=spec.get("question"),
            includes=tuple(spec["includes"]),
            excludes=tuple(spec["excludes"]),
            synonyms=tuple(spec["synonyms"]),
            parent=spec.get("parent"),
            children=tuple(spec.get("children", [])),
            basel=spec.get("basel"),
            yes_no_question=spec.get("yes_no_question"),
            criteria=tuple(sorted((spec.get("criteria") or {}).items())),
        )

    onto = Ontology(
        id=meta["id"], version=meta["version"], roots=tuple(meta["roots"]), nodes=nodes, sha256=digest, raw=raw
    )
    validate(onto)
    return onto


def validate(onto: Ontology) -> None:
    nodes = onto.nodes
    if not onto.roots:
        raise OntologyError("no root nodes")
    for r in onto.roots:
        if r not in nodes:
            raise OntologyError(f"root '{r}' does not exist")
        if nodes[r].parent is not None:
            raise OntologyError(f"root '{r}' has a parent")

    for nid, node in nodes.items():
        if len(set(node.children)) != len(node.children):
            raise OntologyError(f"{nid}: duplicate child reference")
        for c in node.children:
            if c not in nodes:
                raise OntologyError(f"{nid}: child '{c}' does not exist")
            if nodes[c].parent != nid:
                raise OntologyError(f"{c}: parent is '{nodes[c].parent}', but listed as child of '{nid}'")
        if node.criteria and set(node.criteria_dict) != {"true", "false"}:
            raise OntologyError(f"{nid}: criteria must have exactly 'true' and 'false'")
        if node.is_leaf and node.children:
            raise OntologyError(f"{nid}: leaf has children")
        if not node.is_leaf:
            if not node.children:
                raise OntologyError(f"{nid}: non-leaf has no children")
            if not node.question:
                raise OntologyError(f"{nid}: non-leaf has no question")
        if node.parent is not None:
            if node.parent not in nodes:
                raise OntologyError(f"{nid}: parent '{node.parent}' does not exist")
            if nid not in nodes[node.parent].children:
                raise OntologyError(f"{nid}: not listed among parent '{node.parent}' children")
        elif nid not in onto.roots:
            raise OntologyError(f"{nid}: orphan node (no parent, not a root)")

    # Cycle check: walking parents from any node must terminate at a root.
    for nid in nodes:
        seen = set()
        cur: str | None = nid
        while cur is not None:
            if cur in seen:
                raise OntologyError(f"cycle detected through '{cur}'")
            seen.add(cur)
            cur = nodes[cur].parent
