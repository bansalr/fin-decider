"""Mo (Modaic) through Vercel AI Gateway. Same /v1/evaluate request as Jev; all behavior lives in
GatewayAdapter, and the model ID and provider lock come from configuration."""

from .gateway import GatewayAdapter


class MoAdapter(GatewayAdapter):
    pass
