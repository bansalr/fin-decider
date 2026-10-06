"""Jev (TypeSafe AI) through Vercel AI Gateway. All behavior lives in GatewayAdapter;
model ID, provider lock, and limits come from configuration."""

from .gateway import GatewayAdapter


class JevAdapter(GatewayAdapter):
    pass
