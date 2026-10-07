"""Developer tooling: compare GLiClass label templates on the synthetic dev set.

Not part of the classifier pipeline. Mirrors production: for a (case, parent node)
pair, one forward pass scores every child label of that parent. Selection rule
(pre-declared, spec §0.9): highest balanced accuracy at 0.5; ties -> fewer tokens.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/label_check.py --device mps --dtype float32
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from classifier.adapters.gliclass import GLiClassBackend
from classifier.config import load_config
from classifier.ontology import load_ontology
from classifier.provenance import sha256_file
from classifier.templates import load_template

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/classification.yaml"))
    ap.add_argument("--ontology", default=str(ROOT / "gsib_basel_ontology_v1.1.json"))
    ap.add_argument("--cases", default=str(ROOT / "dev/template_cases_v1.jsonl"))
    ap.add_argument("--templates", nargs="+", default=["label_v1", "label_name", "label_v2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    m = load_config(args.config).models["gliclass"]
    backend = GLiClassBackend(m.model, m.revision, args.device, args.dtype, m.max_length)
    onto = load_ontology(args.ontology)
    cases = [json.loads(x) for x in Path(args.cases).read_text().splitlines() if x.strip()]
    jobs: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for c in cases:
        for child, y in c["expect"].items():
            jobs[(c["id"], onto.node(child).parent)][child] = y
    text = {c["id"]: c["text"] for c in cases}

    ranked = []
    for tname in args.templates:
        path = ROOT / f"templates/gliclass/{tname}.txt"
        t = load_template(path, sha256_file(path), "label")
        t.check_ontology(onto)
        keys = list(jobs)
        items = [(backend.fit(text[cid], labs)[0], labs)
                 for cid, par in keys for labs in [[t.render_label(ch) for ch in onto.children(par)]]]
        tokens = sum(backend.fit(text[cid], [t.render_label(ch) for ch in onto.children(par)])[1] for cid, par in keys)
        res = []
        for i in range(0, len(items), 8):
            res += backend.score_batch(items[i:i + 8])
        pairs = [(y, r[t.render_label(onto.node(ch))]) for (cid, par), r in zip(keys, res) for ch, y in jobs[(cid, par)].items()]
        pos = [p >= 0.5 for y, p in pairs if y]
        neg = [p < 0.5 for y, p in pairs if not y]
        bal = (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2
        ranked.append((bal, -tokens, tname))
        print(f"{tname:12s} bal_acc {bal:.3f}  recall {sum(pos)/len(pos):.2f}  specificity {sum(neg)/len(neg):.2f}  tokens {tokens}")
    print(f"\nselected by rule: {max(ranked)[2]}")


if __name__ == "__main__":
    main()
