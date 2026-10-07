from __future__ import annotations

from ..config import ModelCfg
from .base import ClassificationAdapter
from .d1 import D1Adapter
from .fake import FakeAdapter
from .jev import JevAdapter
from .laya import LayaAdapter


def make_adapter(model_key: str, cfg: ModelCfg) -> ClassificationAdapter:
    if cfg.adapter == "jev":
        return JevAdapter(model_key, cfg)
    if cfg.adapter == "laya":
        return LayaAdapter(model_key, cfg)
    if cfg.adapter == "d1":
        return D1Adapter(model_key, cfg)
    if cfg.adapter == "fake":
        return FakeAdapter(model_key)
    raise ValueError(f"unknown adapter {cfg.adapter!r}")
