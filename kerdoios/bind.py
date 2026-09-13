"""Bind an ExecutionPlan onto Hermes and OMP config. Does not call models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .types import ExecutionPlan, ResourceOffer

_NATIVE = frozenset(
    {"nvidia", "groq", "cerebras", "xai", "anthropic", "openai", "nous", "openrouter"}
)
_STRIP_FREE = frozenset({"nvidia", "groq", "cerebras"})
_FALLBACK_BLOCK = re.compile(r"(?m)^fallback_providers:\n(?:[ \t].*\n)*")


def _strip_free_suffix(model: str) -> str:
    if model.endswith(":free"):
        return model[: -len(":free")]
    return model


def _offers_by_id(offers: list[ResourceOffer]) -> dict[str, ResourceOffer]:
    return {offer.id: offer for offer in offers}


def _ordered_offers(plan: ExecutionPlan, offers: list[ResourceOffer]) -> list[ResourceOffer]:
    lookup = _offers_by_id(offers)
    ordered: list[ResourceOffer] = []
    seen: set[str] = set()
    for offer_id in [p.offer_id for p in plan.placements] + list(plan.fallbacks):
        offer = lookup.get(offer_id)
        if offer is None or offer.id in seen:
            continue
        seen.add(offer.id)
        ordered.append(offer)
    return ordered


def hermes_fallback_entries(plan: ExecutionPlan, offers: list[ResourceOffer]) -> list[dict[str, str]]:
    """Hermes fallback_providers rows. Origin-only OpenRouter ids stay on openrouter."""
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for offer in _ordered_offers(plan, offers):
        model = offer.model or offer.id
        provider = offer.provider
        if provider in _STRIP_FREE:
            model = _strip_free_suffix(model)
        if provider not in _NATIVE:
            provider = "openrouter"
        key = (provider, model)
        if key in seen:
            continue
        seen.add(key)
        entries.append({"provider": provider, "model": model})
    return entries


def omp_model_roles(plan: ExecutionPlan, offers: list[ResourceOffer]) -> dict[str, str]:
    """OMP modelRoles slugs (provider/model). First placement is default; rest are smol/tiny."""
    slugs: list[str] = []
    seen: set[str] = set()
    for offer in _ordered_offers(plan, offers):
        model = offer.model or offer.id
        provider = offer.provider
        if provider in _STRIP_FREE:
            model = _strip_free_suffix(model)
        if provider not in _NATIVE:
            slug = model if "/" in model else f"{provider}/{model}"
        elif model.startswith(f"{provider}/"):
            slug = model
        else:
            slug = f"{provider}/{model}"
        if slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    if not slugs:
        raise ValueError("plan has no bindable placements or fallbacks")
    default = slugs[0]
    smol = slugs[1] if len(slugs) > 1 else default
    tiny = slugs[2] if len(slugs) > 2 else smol
    return {"default": default, "smol": smol, "tiny": tiny, "task": default}


def apply_hermes_config(path: Path, plan: ExecutionPlan, offers: list[ResourceOffer]) -> None:
    """Replace fallback_providers. Leave the primary model: block untouched."""
    entries = hermes_fallback_entries(plan, offers)
    if not entries:
        raise ValueError("plan has no bindable placements or fallbacks")
    block = "fallback_providers:\n" + "".join(
        f"  - provider: {entry['provider']}\n    model: {entry['model']}\n" for entry in entries
    )
    target = Path(path)
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    if _FALLBACK_BLOCK.search(text):
        text = _FALLBACK_BLOCK.sub(block, text, count=1)
    else:
        text = (text.rstrip() + "\n" if text.strip() else "") + block
    target.write_text(text, encoding="utf-8")


def apply_omp_profile(path: Path, plan: ExecutionPlan, offers: list[ResourceOffer]) -> None:
    """Write an OMP overlay profile. Does not call models."""
    roles = omp_model_roles(plan, offers)
    lines = [
        "# Kerdoios allotment overlay. Does not call models.\n",
        "modelRoles:\n",
    ]
    for key, value in roles.items():
        lines.append(f"  {key}: {value}\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


def apply_targets(
    plan: ExecutionPlan,
    offers: list[ResourceOffer],
    *,
    hermes_config: Path | None = None,
    omp_config: Path | None = None,
) -> dict[str, Any]:
    written: list[str] = []
    if hermes_config is not None:
        apply_hermes_config(hermes_config, plan, offers)
        written.append(str(hermes_config))
    if omp_config is not None:
        apply_omp_profile(omp_config, plan, offers)
        written.append(str(omp_config))
    if not written:
        raise ValueError("apply requires --hermes-config and/or --omp-config")
    return {
        "hermes": hermes_fallback_entries(plan, offers),
        "omp": omp_model_roles(plan, offers),
        "written": written,
    }
