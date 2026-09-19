"""NVIDIA NIM / build.nvidia.com free hosted inference."""

from __future__ import annotations

from ..types import AccessClass, AccessEvidence, ResourceOffer
from .compat_transport import discover_openai_compat, env_key


def discover(*, free_only: bool = False) -> list[ResourceOffer]:
    key = env_key("NVIDIA_API_KEY", "NGC_API_KEY", "NIM_API_KEY")
    return discover_openai_compat(
        transport="nvidia-nim",
        url="https://integrate.api.nvidia.com/v1/models",
        api_key=key,
        access_class=AccessClass.FREE_TIER,
        evidence=AccessEvidence.PROVIDER_PLAN,
        free_only=free_only,
        concurrency=5,
        latency_ms=1000.0,
    )
