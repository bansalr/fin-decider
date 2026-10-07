"""GLiClass (knowledgator) zero-shot encoder, run locally on a GPU/MPS/CPU.

One forward pass scores every child label of a node (multi-label, independent
sigmoids). Labels come from a `label` template (e.g. "{name}: {definition}").

The uni-encoder input is  [labels][SEP][text]  inside a fixed token window, so the
article is truncated from the end to fit after the labels. Each call records the
tokens actually submitted and whether the article was cut.

Concurrency: traversal runs many articles at once; classify() enqueues work and a
single batcher task groups requests into GPU batches (up to batch_size, or what
arrives within batch_wait_ms) and runs them in a worker thread.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import ModelCfg
from ..ontology import OntologyNode
from .base import ArticleInput, ClassificationAdapter, ClassificationContext, NodeDecision, ProviderFailure

TOKEN_MARGIN = 2  # slack for tokenization differences at the label/text boundary


class Backend(Protocol):
    def fit(self, text: str, labels: list[str]) -> tuple[str, int, bool]: ...
    def score_batch(self, items: list[tuple[str, list[str]]]) -> list[dict[str, float]]: ...


class GLiClassBackend:
    """The real model. Imported lazily so the base install has no torch dependency."""

    def __init__(self, model_id: str, revision: str, device: str, dtype: str, max_length: int):
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = GLiClassModel.from_pretrained(model_id, revision=revision)
        model = model.to(device=device, dtype=getattr(torch, dtype)).eval()
        self.pipe = ZeroShotClassificationPipeline(
            model, self.tokenizer, classification_type="multi-label", device=device,
            max_length=max_length, progress_bar=False,
        )
        # The public class wraps an architecture-specific pipeline that builds the inputs.
        self.inner = getattr(self.pipe, "pipe", self.pipe)
        if not getattr(model.config, "prompt_first", False):
            raise RuntimeError("expected a prompt_first GLiClass model (labels before text)")

    def fit(self, text: str, labels: list[str]) -> tuple[str, int, bool]:
        prefix = self.inner.prepare_input("", labels)
        n_prefix = len(self.tokenizer(prefix, add_special_tokens=True)["input_ids"])
        budget = self.max_length - n_prefix - TOKEN_MARGIN
        if budget <= 0:
            raise ProviderFailure("INVALID_REQUEST", "labels_exceed_window", f"{n_prefix} label tokens")
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        if len(ids) <= budget:
            return text, n_prefix + len(ids), False
        cut = enc["offset_mapping"][budget - 1][1]
        return text[:cut], n_prefix + budget, True

    def score_batch(self, items: list[tuple[str, list[str]]]) -> list[dict[str, float]]:
        texts = [t for t, _ in items]
        labels = [lab for _, lab in items]
        with self.torch.inference_mode():
            out = self.pipe(texts, labels, threshold=0.0, batch_size=len(items))
        return [{d["label"]: float(d["score"]) for d in res} for res in out]


@dataclass
class _Job:
    text: str
    labels: list[str]
    future: asyncio.Future
    t0: float


class GLiClassAdapter(ClassificationAdapter):
    def __init__(self, model_key: str, cfg: ModelCfg, backend: Backend | None = None):
        self.model_key = model_key
        self.cfg = cfg
        self._backend = backend
        self._queue: asyncio.Queue[_Job] | None = None
        self._batcher: asyncio.Task | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliclass")

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            c = self.cfg
            self._backend = GLiClassBackend(c.model, c.revision, c.device, c.dtype, c.max_length)  # type: ignore[arg-type]
        return self._backend

    def _ensure_batcher(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._batcher = asyncio.create_task(self._run_batches())
        return self._queue

    async def classify(self, article: ArticleInput, node: OntologyNode, children: list[OntologyNode],
                       ctx: ClassificationContext) -> NodeDecision:
        labels = [ctx.template.render_label(c) for c in children]
        if len(set(labels)) != len(labels):
            raise ProviderFailure("INVALID_REQUEST", "duplicate_labels", node.id)
        loop = asyncio.get_running_loop()
        backend = await loop.run_in_executor(self._pool, lambda: self.backend)  # first call loads weights
        text, n_tokens, truncated = await loop.run_in_executor(None, backend.fit, article.text, labels)
        fut: asyncio.Future = loop.create_future()
        self._ensure_batcher().put_nowait(_Job(text, labels, fut, time.perf_counter()))
        scores_by_label, latency_ms = await fut
        missing = [lab for lab in labels if lab not in scores_by_label]
        if missing:
            raise ProviderFailure("MODEL_ERROR", "missing_label_scores", str(missing))
        return NodeDecision(
            node_id=node.id,
            scores={c.id: scores_by_label[lab] for c, lab in zip(children, labels)},
            model_returned=f"{self.cfg.model}@{self.cfg.revision}",
            routing_provider="local",
            latency_ms=latency_ms,
            input_tokens=n_tokens,
            output_tokens=0,
            cost_usd=0.0,
            model_truncated=truncated,
            model_submitted_tokens=n_tokens,
        )

    async def _run_batches(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        size, wait = self.cfg.batch_size or 1, (self.cfg.batch_wait_ms or 0) / 1000
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + wait
            while len(batch) < size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            items = [(j.text, j.labels) for j in batch]
            try:
                results = await loop.run_in_executor(self._pool, self.backend.score_batch, items)
                if len(results) != len(batch):
                    raise RuntimeError(f"backend returned {len(results)} results for {len(batch)} inputs")
            except Exception as e:  # one bad batch fails its items, not the run
                for j in batch:
                    if not j.future.done():
                        j.future.set_exception(ProviderFailure("MODEL_ERROR", type(e).__name__, str(e)[:300]))
                continue
            now = time.perf_counter()
            for j, r in zip(batch, results):
                if not j.future.done():
                    j.future.set_result((r, (now - j.t0) * 1000))

    async def aclose(self) -> None:
        if self._batcher is not None:
            self._batcher.cancel()
            try:
                await self._batcher
            except asyncio.CancelledError:
                pass
        self._pool.shutdown(wait=False)
