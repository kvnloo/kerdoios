from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios.aodl import AodlIngestError, work_requirement_from_aodl
from kerdoios.types import ExecutionPlan, Mode, WorkRequirement


def _compact_spec(**work: object) -> dict:
    return {"specVersion": "0.2", "work": dict(work)}


class MapTests(unittest.TestCase):
    def test_maps_work_spec_onto_requirement(self) -> None:
        req = work_requirement_from_aodl(
            _compact_spec(
                context=128_000,
                tools=["github"],
                privacy="local_only",
                parallelism=8,
                budget=0.5,
            )
        )
        self.assertEqual(req.context, 128_000)
        self.assertEqual(req.tools, ("github",))
        self.assertTrue(req.tool_use)
        self.assertEqual(req.privacy, "local_only")
        self.assertEqual(req.parallelism, 8)
        self.assertEqual(req.maximum_cost, 0.5)

    def test_maps_constraints_and_tool_nodes(self) -> None:
        req = work_requirement_from_aodl(
            {
                "specVersion": "0.2",
                "graphId": "fanout",
                "policies": {"dynamic": {"maxChildren": 12}},
                "constraints": {
                    "budgets": {"usd": 0.25},
                    "privacy": "confidential",
                    "context": 64_000,
                    "termination": {"on": "verifier.succeeded"},
                },
                "intentGraph": {
                    "nodes": [
                        {"id": "github", "kind": "tool"},
                        {"id": "worker", "kind": "executor"},
                    ]
                },
            }
        )
        self.assertEqual(req.context, 64_000)
        self.assertEqual(req.tools, ("github",))
        self.assertTrue(req.tool_use)
        self.assertEqual(req.privacy, "confidential")
        self.assertEqual(req.parallelism, 12)
        self.assertEqual(req.maximum_cost, 0.25)

    def test_work_block_wins_over_constraints(self) -> None:
        req = work_requirement_from_aodl(
            {
                "specVersion": "0.2",
                "work": {"privacy": "local_only", "parallelism": 3, "budget": 1.0},
                "constraints": {"budgets": {"usd": 9.0}, "privacy": "public"},
            }
        )
        self.assertEqual(req.privacy, "local_only")
        self.assertEqual(req.parallelism, 3)
        self.assertEqual(req.maximum_cost, 1.0)

    def test_defaults_fill_unspecified_fields(self) -> None:
        defaults = WorkRequirement(
            coding=0.9,
            reasoning=0.4,
            context=8_192,
            parallelism=2,
            privacy="public",
            mode=Mode.CHEAP,
            tools=("github",),
            tool_use=True,
        )
        req = work_requirement_from_aodl(
            _compact_spec(privacy="confidential", budget=0.1),
            defaults=defaults,
        )
        self.assertEqual(req.privacy, "confidential")
        self.assertEqual(req.maximum_cost, 0.1)
        self.assertEqual(req.parallelism, 2)
        self.assertEqual(req.context, 8_192)
        self.assertEqual(req.mode, Mode.CHEAP)
        self.assertEqual(req.coding, 0.9)

    def test_zero_usd_budget_is_a_cap_not_missing(self) -> None:
        req = work_requirement_from_aodl(_compact_spec(budget=0))
        self.assertEqual(req.maximum_cost, 0.0)

    def test_empty_tools_means_no_tool_use(self) -> None:
        req = work_requirement_from_aodl(_compact_spec(tools=[]))
        self.assertEqual(req.tools, ())
        self.assertFalse(req.tool_use)


class InvalidTests(unittest.TestCase):
    def test_non_object_fails(self) -> None:
        with self.assertRaises(AodlIngestError):
            work_requirement_from_aodl(["not", "a", "doc"])

    def test_missing_spec_version_fails(self) -> None:
        with self.assertRaises(AodlIngestError):
            work_requirement_from_aodl({"work": {"privacy": "public"}})

    def test_unknown_privacy_fails(self) -> None:
        with self.assertRaises(AodlIngestError):
            work_requirement_from_aodl(_compact_spec(privacy="secret"))

    def test_bad_tools_type_fails(self) -> None:
        with self.assertRaises(AodlIngestError):
            work_requirement_from_aodl(_compact_spec(tools="github"))

    def test_non_positive_parallelism_fails(self) -> None:
        with self.assertRaises(AodlIngestError):
            work_requirement_from_aodl(_compact_spec(parallelism=0))

    def test_non_numeric_budget_fails(self) -> None:
        with self.assertRaises(AodlIngestError):
            work_requirement_from_aodl(_compact_spec(budget="cheap"))


class CliTests(unittest.TestCase):
    def _plan(self, argv: list[str], *, captured: dict) -> int:
        built = ExecutionPlan(
            estimated_cost=0.0,
            estimated_duration_seconds=0.0,
            confidence=0.0,
            placements=[],
            fallbacks=[],
            rejections=[],
            mode="cheap",
        )

        def fake_plan(_offers, req, **_kwargs):
            captured["req"] = req
            return built

        from kerdoios.__main__ import main

        with (
            patch("kerdoios.__main__.discover_all", return_value=[]),
            patch("kerdoios.__main__.plan", side_effect=fake_plan),
        ):
            return main(argv)

    def test_plan_aodl_path_maps_requirement(self) -> None:
        spec = _compact_spec(
            context=200_000,
            tools=["github"],
            privacy="local_only",
            parallelism=11,
            budget=0.4,
        )
        captured: dict = {}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(spec, handle)
            path = handle.name
        try:
            rc = self._plan(["plan", "--aodl", path, "--workers", "3"], captured=captured)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(rc, 0)
        req = captured["req"]
        self.assertEqual(req.parallelism, 11)
        self.assertEqual(req.privacy, "local_only")
        self.assertEqual(req.context, 200_000)
        self.assertEqual(req.maximum_cost, 0.4)

    def test_explain_aodl_stdin(self) -> None:
        spec = _compact_spec(privacy="confidential", parallelism=4)
        captured: dict = {}
        from kerdoios.__main__ import main

        built = ExecutionPlan(
            estimated_cost=0.0,
            estimated_duration_seconds=0.0,
            confidence=0.0,
            placements=[],
            fallbacks=[],
            rejections=[],
            mode="cheap",
        )

        def fake_plan(_offers, req, **_kwargs):
            captured["req"] = req
            return built

        with (
            patch("kerdoios.__main__.discover_all", return_value=[]),
            patch("kerdoios.__main__.plan", side_effect=fake_plan),
            patch("kerdoios.__main__.explain", return_value="ok"),
            patch("sys.stdin", io.StringIO(json.dumps(spec))),
        ):
            rc = main(["explain", "--aodl", "-"])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["req"].privacy, "confidential")
        self.assertEqual(captured["req"].parallelism, 4)

    def test_invalid_aodl_file_fails_clearly(self) -> None:
        from kerdoios.__main__ import main

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write("{")
            path = handle.name
        err = io.StringIO()
        try:
            with (
                patch("kerdoios.__main__.discover_all", return_value=[]),
                patch("sys.stderr", err),
            ):
                rc = main(["plan", "--aodl", path])
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertNotEqual(rc, 0)
        self.assertIn("aodl:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
