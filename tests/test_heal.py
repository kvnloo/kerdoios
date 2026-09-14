"""Self-healing batch executor: retry, failover, per-task isolation, auto-record.

Red-phase tests: kerdoios.heal does not exist yet; every test here must
fail on import until the executor is implemented.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUB_WORKER = """\
import json, sys, time

def main():
    script_path, state_path = sys.argv[1], sys.argv[2]
    task = json.load(sys.stdin)
    tid = str(task.get("id"))
    script = json.load(open(script_path))
    try:
        state = json.load(open(state_path))
    except (OSError, ValueError):
        state = {}
    idx = state.get(tid, 0)
    behaviors = script.get(tid, [{"exit": 0, "out": "stub-ok"}])
    behavior = behaviors[min(idx, len(behaviors) - 1)]
    state[tid] = idx + 1
    json.dump(state, open(state_path, "w"))
    if behavior.get("sleep"):
        time.sleep(behavior["sleep"])
    out = behavior.get("out", "")
    if out:
        sys.stdout.write(out if out.endswith("\\n") else out + "\\n")
        sys.stdout.flush()
    sys.exit(behavior.get("exit", 0))

main()
"""


def write_stub(tmp: Path) -> Path:
    path = tmp / "stub_worker.py"
    path.write_text(STUB_WORKER, encoding="utf-8")
    return path


def write_script(tmp: Path, behaviors: dict) -> tuple[Path, Path]:
    script = tmp / "script.json"
    script.write_text(json.dumps(behaviors), encoding="utf-8")
    state = tmp / "state.json"
    return script, state


def worker_cmd(stub: Path, script: Path, state: Path) -> str:
    return f"{sys.executable} {stub} {script} {state} --model {{model}}"


def make_plan(primary=("openrouter", "model-a"), fallback=("openrouter", "model-b"),
              policy=None) -> dict:
    pol = {
        "max_retries": 2,
        "backoff_s": 0.01,
        "attempt_timeout_s": 30.0,
        "retry_on": ["rate_limited", "timeout", "upstream_5xx", "empty_response"],
    }
    if policy:
        pol.update(policy)
    return {
        "estimated_cost": 0.0,
        "estimated_duration_seconds": 0.0,
        "confidence": 0.9,
        "mode": "free",
        "unplaced_workers": 0,
        "rejections": [],
        "fallbacks": [],
        "placements": [
            {
                "offer_id": "openrouter/model-a",
                "provider": primary[0],
                "model": primary[1],
                "workers": 4,
                "estimated_cost": 0.0,
                "reasons": ["test"],
                "fallbacks": [{"provider": fallback[0], "model": fallback[1]}],
                "retry_policy": pol,
            }
        ],
    }


def run_heal(plan, tasks, cmd, **kwargs):
    from kerdoios.heal import run_batch

    return run_batch(plan, tasks, cmd, **kwargs)


class HealRetryTests(unittest.TestCase):
    def test_transient_429_heals_on_retry(self) -> None:
        from kerdoios.heal import run_batch  # noqa: F401  (import must exist)

        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {"t1": [{"exit": 3}, {"exit": 3}, {"exit": 0, "out": "healed"}]})
            log = tmpp / "observed.jsonl"
            results = run_batch(
                make_plan(), [{"id": "t1"}],
                worker_cmd(stub, script, state), observed_log=log,
            )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r["ok"], msg=json.dumps(r, indent=2))
        self.assertEqual(r["model"], "model-a")
        self.assertEqual(r["output"], "healed")
        self.assertEqual(len(r["attempts"]), 3)
        self.assertEqual([a["reason"] for a in r["attempts"]],
                         ["rate_limited", "rate_limited", None])

    def test_empty_response_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {"t1": [{"exit": 6}, {"exit": 0, "out": "recovered"}]})
            results = run_heal(make_plan(), [{"id": "t1"}], worker_cmd(stub, script, state),
                               observed_log=tmpp / "o.jsonl")
        self.assertTrue(results[0]["ok"])
        self.assertEqual(len(results[0]["attempts"]), 2)

    def test_failover_when_retries_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {
                "t1": [{"exit": 3}, {"exit": 3}, {"exit": 0, "out": "via-fallback"}],
            })
            plan = make_plan(policy={"max_retries": 1, "backoff_s": 0.01})
            results = run_heal(plan, [{"id": "t1"}], worker_cmd(stub, script, state),
                               observed_log=tmpp / "o.jsonl")
        r = results[0]
        self.assertTrue(r["ok"], msg=json.dumps(r, indent=2))
        self.assertEqual(r["model"], "model-b")
        # 2 attempts on primary (1 retry), then failover succeeds on its first try
        self.assertEqual(len(r["attempts"]), 3)
        self.assertEqual(r["attempts"][-1]["model"], "model-b")

    def test_worker_crash_fails_over_without_burning_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {
                "t1": [{"exit": 1, "out": " Traceback (most recent call last): boom"},
                       {"exit": 0, "out": "via-fallback"}],
            })
            results = run_heal(make_plan(), [{"id": "t1"}], worker_cmd(stub, script, state),
                               observed_log=tmpp / "o.jsonl")
        r = results[0]
        self.assertTrue(r["ok"])
        # crash is not in retry_on: exactly one primary attempt, then failover
        primary_attempts = [a for a in r["attempts"] if a["model"] == "model-a"]
        self.assertEqual(len(primary_attempts), 1)
        self.assertEqual(primary_attempts[0]["reason"], "worker_crash")

    def test_timeout_reason_on_hung_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {"t1": [{"sleep": 30, "exit": 0}]})
            plan = make_plan(policy={"max_retries": 0, "backoff_s": 0.01, "attempt_timeout_s": 0.5})
            plan["placements"][0]["fallbacks"] = []
            results = run_heal(plan, [{"id": "t1"}], worker_cmd(stub, script, state),
                               observed_log=tmpp / "o.jsonl")
        r = results[0]
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "timeout")


class HealBatchRobustnessTests(unittest.TestCase):
    def test_batch_completes_with_failure_stubs_when_everything_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {
                "t1": [{"exit": 3}], "t2": [{"exit": 5}],
            })
            plan = make_plan(policy={"max_retries": 0, "backoff_s": 0.01})
            plan["placements"][0]["fallbacks"] = []
            tasks = [{"id": "t1"}, {"id": "t2"}]
            results = run_heal(plan, tasks, worker_cmd(stub, script, state),
                               observed_log=tmpp / "o.jsonl")
        self.assertEqual(len(results), 2)
        by_id = {r["task_id"]: r for r in results}
        self.assertFalse(by_id["t1"]["ok"])
        self.assertEqual(by_id["t1"]["reason"], "rate_limited")
        self.assertIn("error", by_id["t1"])
        self.assertFalse(by_id["t2"]["ok"])
        self.assertEqual(by_id["t2"]["reason"], "upstream_5xx")

    def test_one_crashing_task_does_not_kill_the_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {
                # exit 2 with non-utf8-ish garbage is unparseable -> worker_crash
                "t1": [{"exit": 2, "out": "\x00\x01not-json-at-all"}],
                "t2": [{"exit": 0, "out": "fine"}],
            })
            plan = make_plan(policy={"max_retries": 0, "backoff_s": 0.01})
            plan["placements"][0]["fallbacks"] = []
            results = run_heal(plan, [{"id": "t1"}, {"id": "t2"}],
                               worker_cmd(stub, script, state),
                               observed_log=tmpp / "o.jsonl", jobs=2)
        by_id = {r["task_id"]: r for r in results}
        self.assertEqual(len(results), 2)
        self.assertFalse(by_id["t1"]["ok"])
        self.assertTrue(by_id["t2"]["ok"])
        self.assertEqual(by_id["t2"]["output"], "fine")

    def test_attempts_auto_record_with_reasons(self) -> None:
        from kerdoios.observed import load_observations

        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {"t1": [{"exit": 3}, {"exit": 0, "out": "ok"}]})
            log = tmpp / "observed.jsonl"
            run_heal(make_plan(), [{"id": "t1"}], worker_cmd(stub, script, state),
                     observed_log=log, task_type="research")
            rows = load_observations(path=log)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].reason, "rate_limited")
        self.assertFalse(rows[0].completed)
        self.assertTrue(rows[0].retried is False)
        self.assertTrue(rows[1].completed)
        self.assertTrue(rows[1].retried)
        self.assertEqual(rows[1].task_type, "research")


class HealPlanContractTests(unittest.TestCase):
    def test_plan_placements_carry_fallbacks_and_retry_policy(self) -> None:
        from kerdoios.optimize import plan
        from kerdoios.providers.fixture import fixture_offers
        from kerdoios.types import Mode, WorkRequirement

        req = WorkRequirement(coding=0.6, reasoning=0.6, tool_use=True,
                              tools=("github",), context=128_000,
                              parallelism=4, mode=Mode.CHEAP)
        built = plan(fixture_offers(), req)
        self.assertTrue(built.placements)
        p = built.placements[0]
        d = p.to_dict()
        self.assertIn("fallbacks", d)
        self.assertIn("retry_policy", d)
        self.assertTrue(d["fallbacks"], "primary placement must name substitutes")
        for fb in d["fallbacks"]:
            self.assertIn("provider", fb)
            self.assertIn("model", fb)
        self.assertNotIn(p.offer_id, [f.get("model") and f"{f['provider']}/{f['model']}" for f in d["fallbacks"]])
        rp = d["retry_policy"]
        for key in ("max_retries", "backoff_s", "retry_on"):
            self.assertIn(key, rp)
        for reason in ("rate_limited", "timeout", "upstream_5xx", "empty_response"):
            self.assertIn(reason, rp["retry_on"])

    def test_run_cli_end_to_end(self) -> None:
        from kerdoios.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            stub = write_stub(tmpp)
            script, state = write_script(tmpp, {"t1": [{"exit": 3}, {"exit": 0, "out": "cli-ok"}]})
            plan_path = tmpp / "plan.json"
            plan_path.write_text(json.dumps(make_plan()), encoding="utf-8")
            tasks_path = tmpp / "tasks.jsonl"
            tasks_path.write_text(json.dumps({"id": "t1", "prompt": "hi"}) + "\n", encoding="utf-8")
            out_path = tmpp / "results.jsonl"
            log = tmpp / "observed.jsonl"
            rc = main([
                "run",
                "--plan", str(plan_path),
                "--tasks", str(tasks_path),
                "--worker-cmd", worker_cmd(stub, script, state),
                "--out", str(out_path),
                "--observed-log", str(log),
            ])
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ok"])
        self.assertEqual(rows[0]["output"], "cli-ok")


if __name__ == "__main__":
    unittest.main()
