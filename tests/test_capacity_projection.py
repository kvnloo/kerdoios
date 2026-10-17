from __future__ import annotations

import json
import stat

from kerdoios.capacity_projection import SCHEMA, build_projection, write_projection
from kerdoios.quota import QuotaDimension, QuotaState
from kerdoios.types import CapabilityProfile, Capacity, Economics, ResourceOffer, Telemetry


def offer(*, quota=None, remaining=0.0, source="test"):
    return ResourceOffer(
        id="openrouter/free-model",
        provider="openrouter",
        resource_type="llm",
        model="free-model",
        local=False,
        capabilities=CapabilityProfile(),
        capacity=Capacity(),
        economics=Economics(
            remaining_free_quota=remaining,
            quota=quota,
        ),
        telemetry=Telemetry(),
        source=source,
    )


def test_unknown_legacy_zero_is_not_projected_as_exhausted():
    payload = build_projection([offer()], now=10.0)
    assert payload["schema"] == SCHEMA
    assert payload["entries"][0]["quota"] == {}


def test_dimensional_quota_and_reset_are_preserved():
    quota = QuotaState(
        provider="openrouter",
        model="free-model",
        observed_at=5.0,
        dimensions={
            "rpd": QuotaDimension(
                limit=50,
                remaining=12,
                reset_at=100.0,
                source="provider_header",
            )
        },
    )
    row = build_projection([offer(quota=quota)], now=10.0)["entries"][0]
    assert row["quota"]["dimensions"]["rpd"]["remaining"] == 12
    assert row["quota"]["dimensions"]["rpd"]["source"] == "provider_header"
    assert row["reset_at"] == 100.0
    assert row["observed_at"] == 5.0


def test_positive_legacy_capacity_is_preserved_without_inventing_provenance():
    row = build_projection([offer(remaining=7.0)], now=10.0)["entries"][0]
    assert row["quota"] == {"remaining_free_quota": 7.0}


def test_private_atomic_write(tmp_path):
    payload = build_projection([], now=10.0)
    path = write_projection(payload, tmp_path / "capacity.json")
    assert json.loads(path.read_text())["schema"] == SCHEMA
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
