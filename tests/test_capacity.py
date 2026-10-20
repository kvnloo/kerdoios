from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kerdoios.capacity import build_capacity, latest_nous_notice, load_state, observe

PACE = {
    "at": "2026-10-20T08:00:00+00:00",
    "z0_factory": {"prefer": ["claude"], "avoid": ["codex"]},
    "lanes": {
        "claude": {"band": "under-burn", "max_workflow_agents": 12, "sustainable_pct_per_h": 1.0,
                   "observed_pct_per_h": 0.4, "pct_budget_next_hour": 1.0, "hours_to_reset": 48},
        "codex": {"band": "hard-throttle", "max_workflow_agents": 0, "sustainable_pct_per_h": 0.6,
                  "observed_pct_per_h": 1.2, "pct_budget_next_hour": 0.6, "hours_to_reset": 160},
    },
}
NOW = 1_800_000_000.0


class CapacityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.notice = self.tmp / "run.txt"
        self.notice.write_text("  • You've used $150.00 of your $220.00 cap\n")
        self.free = self.tmp / "catalog.json"
        self.free.write_text(json.dumps({"llm_apis": [{"provider": "groq", "source_ids": [6], "models": [
            {"model": "openai/gpt-oss-120b", "rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}]}]}))

    def cap(self, **kw):
        base: dict[str, Any] = dict(pace=PACE, notice_globs=[str(self.notice)], free_tier=self.free, now=NOW)
        base.update(kw)
        return build_capacity(**base)["lanes"]

    def test_under_burn_subscription_offers_agents_and_headroom(self) -> None:
        opus = self.cap()["opus"]
        self.assertEqual(opus["slots_now"], 12)
        self.assertAlmostEqual(opus["pacing"]["headroom_pct_per_h"], 0.6)

    def test_recent_burst_is_reported_beside_the_band(self) -> None:
        hist = self.tmp / "burn-pace.jsonl"
        rows = [("2026-01-15T05:00:00+00:00", 47.0), ("2026-01-15T08:00:00+00:00", 52.0)]
        hist.write_text("".join(json.dumps({"at": at, "lanes": {"claude": {"weekly_used_pct": p}}}) + "\n"
                                for at, p in rows))
        from datetime import datetime
        now = datetime.fromisoformat("2026-01-15T08:10:00+00:00").timestamp()
        opus = self.cap(burn_pace_history=hist, now=now)["opus"]
        self.assertAlmostEqual(opus["pacing"]["recent_pct_per_h"], 1.667, places=2)
        self.assertIn("warning", opus["pacing"])
        self.assertEqual(opus["slots_now"], 12)  # burn-pace stays the authority on the band

    def test_hard_throttle_or_avoided_pool_gets_no_slots(self) -> None:
        sol = self.cap()["sol"]
        self.assertEqual(sol["slots_now"], 0)
        self.assertIn("hard-throttle", sol["reason"])

    def test_nous_lane_needs_a_spend_notice_and_stops_at_the_cap(self) -> None:
        self.assertEqual(self.cap()["haiku"]["spend"]["remaining_usd"], 70.0)
        self.assertEqual(self.cap(notice_globs=[])["haiku"]["slots_now"], 0)
        self.notice.write_text("You've used $220.00 of your $220.00 cap")
        self.assertEqual(self.cap()["luna"]["reason"], "Nous spend cap reached")

    def test_published_limits_size_api_lane_by_tokens_not_blanket_one(self) -> None:
        groq = self.cap()["groq"]["limits"]
        self.assertEqual(groq["jobs_per_min"]["value"], 3.2)  # 8000 TPM / 2500 tokens < 30 RPM
        self.assertEqual(groq["window_jobs"]["value"], 80)  # 200K TPD / 2500
        self.assertEqual(groq["rpm"]["classification"], "published")

    def test_unknown_lane_gets_one_trial_slot_not_unlimited(self) -> None:
        glm = self.cap()["glm"]
        self.assertEqual(glm["slots_now"], 1)
        self.assertEqual(glm["limits"]["max_concurrency"]["confidence"], "low")

    def test_retired_model_blocks_lane(self) -> None:
        nim = self.cap(models={"nim": ["other/model"]})["nim"]
        self.assertEqual(nim["slots_now"], 0)
        self.assertIn("not in provider", nim["reason"])

    def test_nim_picks_first_listed_preferred_model_and_skips_unsupported(self) -> None:
        listed = ["z-ai/glm-5.3", "nvidia/nemotron-3.5-lightning-30b-a3b"]  # ultra retired from the list
        nim = self.cap(models={"nim": listed})["nim"]
        self.assertEqual(nim["model"], "z-ai/glm-5.3")
        self.assertGreater(nim["slots_now"], 0)
        path = self.tmp / "state.json"
        observe("nim", "unsupported_model", model="z-ai/glm-5.3", now=NOW, path=path)
        nim = self.cap(models={"nim": listed}, state=load_state(path), now=NOW + 10)["nim"]
        self.assertEqual(nim["model"], "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(nim["state"]["jobs_24h"], 0)  # a 404 is not a spent request window
        observe("nim", "unsupported_model", model="nvidia/nemotron-3.5-lightning-30b-a3b", now=NOW, path=path)
        nim = self.cap(models={"nim": listed}, state=load_state(path), now=NOW + 10)["nim"]
        self.assertEqual(nim["slots_now"], 0)

    def test_published_rpm_is_paced_below_the_ceiling(self) -> None:
        self.free.write_text(json.dumps({"llm_apis": [{"provider": "nvidia", "source_ids": [1], "models": [
            {"model": "*", "rpm": 40, "rpd": 10000}]}]}))
        nim = self.cap()["nim"]["limits"]
        self.assertEqual(nim["jobs_per_min"]["value"], 36.0)
        self.assertEqual(nim["window_jobs"]["value"], 10000)

    def test_measured_job_latency_sizes_api_concurrency(self) -> None:
        self.free.write_text(json.dumps({"llm_apis": [{"provider": "nvidia", "source_ids": [1], "models": [
            {"model": "*", "rpm": 40, "rpd": 10000}]}]}))
        path = self.tmp / "state.json"
        before = self.cap(state=load_state(path))["nim"]["slots_now"]
        for i in range(5):
            observe("nim", "ok", seconds=12.0, now=NOW + i, path=path)
        nim = self.cap(state=load_state(path), now=NOW + 10)["nim"]
        self.assertEqual(nim["latency_measured_ms"], 12000)
        self.assertEqual(nim["slots_now"], 8)  # 36 jobs/min x 12 s = 7.2 in flight, +1, capped at 8
        self.assertLess(before, 8)

    def test_quota_stop_rests_lane_at_once_then_probes_one_slot(self) -> None:
        path = self.tmp / "state.json"
        observe("minimax", "ok", now=NOW, path=path)
        observe("minimax", "quota", now=NOW + 1, path=path)
        lane = self.cap(state=load_state(path), now=NOW + 60)["minimax"]
        self.assertEqual(lane["slots_now"], 0)
        self.assertIn("cooling down", lane["reason"])
        self.assertLessEqual(self.cap(state=load_state(path), now=NOW + 3700)["minimax"]["slots_now"], 1)

    def test_metered_lane_waits_for_explicit_permission(self) -> None:
        self.assertEqual(self.cap()["cerebras"]["slots_now"], 0)
        self.assertGreater(self.cap(allow_metered=True)["cerebras"]["slots_now"], 0)

    def test_throttles_cool_down_then_spent_quota_rests_an_hour(self) -> None:
        path = self.tmp / "state.json"
        observe("ds", "ok", now=NOW, path=path)
        observe("ds", "rate_limited", now=NOW + 1, retry_after_s=30, path=path)
        state = load_state(path)
        self.assertEqual(state["lanes"]["ds"]["cooldown_until"], NOW + 31)
        self.assertEqual(self.cap(state=state, now=NOW + 5)["ds"]["slots_now"], 0)
        self.assertEqual(self.cap(state=state, now=NOW + 40)["ds"]["slots_now"], 2)
        observe("ds", "rate_limited", now=NOW + 40, path=path)
        observe("ds", "rate_limited", now=NOW + 200, path=path)
        state = load_state(path)
        self.assertEqual(state["lanes"]["ds"]["quota_suspected"]["ok_jobs_before"], 1)
        self.assertEqual(state["lanes"]["ds"]["cooldown_until"], NOW + 200 + 3600)
        self.assertEqual(self.cap(state=state, now=NOW + 4000)["ds"]["slots_now"], 1)  # one probe, not two
        observe("ds", "ok", now=NOW + 4001, path=path)
        self.assertEqual(self.cap(state=load_state(path), now=NOW + 4002)["ds"]["slots_now"], 2)

    def test_window_usage_counts_against_the_lane_budget(self) -> None:
        path = self.tmp / "state.json"
        for i in range(30):
            observe("ds", "ok", now=NOW + i, path=path)
        ds = self.cap(state=load_state(path), now=NOW + 60)["ds"]
        self.assertEqual(ds["window_jobs_left"], 0)
        self.assertEqual(ds["slots_now"], 0)

    def test_trial_lane_grows_on_clean_runs_and_falls_back_on_a_throttle(self) -> None:
        path = self.tmp / "state.json"
        for i in range(10):
            observe("minimax", "ok", now=NOW + i, path=path)
        self.assertEqual(self.cap(state=load_state(path), now=NOW + 60)["minimax"]["slots_now"], 3)
        observe("minimax", "rate_limited", now=NOW + 61, path=path)
        self.assertEqual(self.cap(state=load_state(path), now=NOW + 4000)["minimax"]["slots_now"], 1)

    def test_executor_notice_newer_than_files_wins(self) -> None:
        path = self.tmp / "state.json"
        observe("haiku", "ok", now=NOW, nous_used=160.0, nous_cap=220.0, path=path)
        self.notice.touch()
        import os
        os.utime(self.notice, (NOW - 100, NOW - 100))
        self.assertEqual(self.cap(state=load_state(path))["haiku"]["spend"]["used_usd"], 160.0)
        notice = latest_nous_notice([str(self.notice)])
        assert notice is not None
        self.assertEqual(notice["used_usd"], 150.0)


if __name__ == "__main__":
    unittest.main()
