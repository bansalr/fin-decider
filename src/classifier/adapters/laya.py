"""Laya (Convai Innovations) through Vercel AI Gateway. Same request shape and
template as Jev; only the configured model ID and provider lock differ."""

from .gateway import GatewayAdapter


class LayaAdapter(GatewayAdapter):
    pass
