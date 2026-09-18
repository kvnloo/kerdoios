from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.allocation_request import work_requirement_from_allocation_request
from kerdoios.observed import (
    Observation,
    import_allocation_observations,
    record,
    tokens_per_verified_task,
)
from kerdoios.optimize import plan
from kerdoios.providers.fixture import fixture_offers
from kerdoios.types import Mode, QuotaConstraint, WorkRequirement


class ObservationV2Tests(unittest.TestCase):
    def test_execution_completed_without_verified_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            record(
                Observation(
                    provider="fixture",
                    model="cheap",
                    task_type="coding",
                    completed=True,
                    actual_cost=0.01,
                    input_tokens=100,
                    output_tokens=50,
                    execution_completed=True,
                    verified_success=None,
                ),
                path=path,
            )
            # V2: null verified must not count
            stats = tokens_per_verified_task(path=path)
            self.assertEqual(stats.get("verified_tasks", stats.get("n_verified", 0)), 0)

    def test_verified_true_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            record(
                Observation(
                    provider="fixture",
                    model="cheap",
                    task_type="coding",
                    completed=True,
                    actual_cost=0.02,
                    input_tokens=100,
                    output_tokens=50,
                    execution_completed=True,
                    verified_success=True,
                ),
                path=path,
            )
            stats = tokens_per_verified_task(path=path)
            n = stats.get("verified_tasks", stats.get("n_verified", stats.get("tasks")))
            self.assertTrue(n and int(n) >= 1)

    def test_import_allocation_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jsonl"
            dest = Path(tmp) / "out.jsonl"
            row = {
                "schema": "z0int.allocation_observation.v1",
                "provider": "openrouter",
                "model": "x",
                "execution_completed": True,
                "verified_success": None,
                "actual_cost": 0.0,
                "capability_id": "coding.edit",
                "input_tokens": 10,
                "output_tokens": 5,
            }
            src.write_text(json.dumps(row) + "\n")
            result = import_allocation_observations(src, dest=dest)
            self.assertEqual(result["imported"], 1)
            stats = tokens_per_verified_task(path=dest)
            n = stats.get("verified_tasks", stats.get("n_verified", 0))
            self.assertEqual(int(n or 0), 0)


class AllocationRequestTests(unittest.TestCase):
    def test_accept_allocation_request_v1(self) -> None:
        doc = {
            "schema": "z0int.allocation_request.v1",
            "capability_id": "coding.edit",
            "requirement": {
                "coding": 0.9,
                "reasoning": 0.5,
                "parallelism": 3,
                "mode": "cheap",
            },
            "quotas": [
                {"name": "rpm", "limit": 60, "unit": "requests", "group": "shared-moa"},
                {"name": "tpm", "limit": 100000, "unit": "tokens", "group": "shared-moa"},
            ],
            "join_policy": "all",
        }
        req = work_requirement_from_allocation_request(doc)
        self.assertEqual(req.capability_id, "coding.edit")
        self.assertEqual(len(req.quotas), 2)
        self.assertEqual(req.quotas[0].group, "shared-moa")
        self.assertEqual(req.join_policy, "all")

    def test_shared_quota_moa_single_reservation(self) -> None:
        offers = fixture_offers()
        req = WorkRequirement(
            coding=0.5,
            reasoning=0.5,
            parallelism=2,
            mode=Mode.CHEAP,
            quotas=(
                QuotaConstraint(name="rpm", limit=30, unit="requests", group="moa"),
                QuotaConstraint(name="rpm-dup", limit=30, unit="requests", group="moa"),
            ),
            join_policy="all",
        )
        built = plan(offers, req)
        d = built.to_dict()
        self.assertEqual(d["schema"], "kerdoios.execution_plan.v2")
        self.assertIn("quota_reservations", d)
        self.assertIn("retry_policy", d)
        self.assertEqual(d["join_policy"], "all")
        # shared group reserved once
        groups = [r["group"] for r in d["quota_reservations"]]
        self.assertEqual(groups.count("moa"), 1)


if __name__ == "__main__":
    unittest.main()
