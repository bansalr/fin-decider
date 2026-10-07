"""Liquid d1 through Vercel AI Gateway. Same request shape and template as Jev
and Laya; only the configured model ID and provider lock differ."""

from .gateway import GatewayAdapter


class D1Adapter(GatewayAdapter):
    pass
