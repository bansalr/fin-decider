from __future__ import annotations

from ..config import ModelCfg
from .base import ClassificationAdapter
from .clef import ClefAdapter
from .d1 import D1Adapter
from .fake import FakeAdapter
from .jev import JevAdapter
from .laya import LayaAdapter
from .systemone import SystemOneAdapter


def make_adapter(model_key: str, cfg: ModelCfg) -> ClassificationAdapter:
    if cfg.adapter == "jev":
        return JevAdapter(model_key, cfg)
    if cfg.adapter == "laya":
        return LayaAdapter(model_key, cfg)
    if cfg.adapter == "clef":
        return ClefAdapter(model_key, cfg)
    if cfg.adapter == "d1":
        return D1Adapter(model_key, cfg)
    if cfg.adapter == "systemone":
        return SystemOneAdapter(model_key, cfg)
    if cfg.adapter == "fake":
        return FakeAdapter(model_key)
    raise ValueError(f"unknown adapter {cfg.adapter!r}")
