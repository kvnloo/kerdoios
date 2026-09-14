"""Self-healing batch executor for ExecutionPlans.

Why this exists: a batch of parallel model calls dies completely when one
call 429s or returns empty content and the loop has no recovery. `run`
wraps each task in retry-with-backoff plus automatic failover through the
plan's per-placement fallback chain, so the batch always completes — every
task yields a result or a failure stub.

Why it looks like this: kerdoios allots compute; it never speaks to a
model API (see tests/test_no_model_execution.py). `run` keeps that line.
The operator supplies --worker-cmd, a command that performs ONE attempt of
ONE task against ONE model. kerdoios contributes only the healing loop
around it: per-task isolation, retries, failover, and auto-recorded
outcomes so the next --observed plan routes around sick models.

Worker contract:
  argv: --worker-cmd with {model}, {provider}, {task_id} substituted; the
        task JSON object is piped on stdin. Env also carries
        KERDOIOS_MODEL, KERDOIOS_PROVIDER, KERDOIOS_TASK_ID.
  exit 0: attempt succeeded; stdout is the result payload.
  exit 3: rate_limited   exit 4: timeout
  exit 5: upstream_5xx   exit 6: empty_response
  any other nonzero exit, or output that cannot be interpreted:
        worker_crash.
  A JSON object on the last stdout line like
        {"reason": "rate_limited", "error": "...", "cost": 0.001}
  overrides the exit-code mapping when it names a known reason, and
  "cost" is recorded as the attempt's actual cost.
  A kerdoios-side per-attempt timeout counts as timeout.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .observed import Observation, record
from .types import DEFAULT_RETRY_POLICY, RETRYABLE_REASONS

EXIT_REASONS = {
    3: "rate_limited",
    4: "timeout",
    5: "upstream_5xx",
    6: "empty_response",
}
KNOWN_REASONS = set(RETRYABLE_REASONS) | {"worker_crash", "unknown"}

# Cap exponential backoff so a pathological retry policy cannot sleep forever.
MAX_BACKOFF_S = 300.0


def _parse_envelope(text: str) -> dict:
    """Best-effort parse of a trailing JSON envelope from worker output."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
        return {}
    return {}


def _classify(proc: subprocess.CompletedProcess, timed_out: bool) -> tuple[str | None, str, float]:
    """Return (reason, error, cost). reason is None on success."""
    if timed_out:
        return "timeout", "attempt exceeded attempt_timeout_s", 0.0
    out = proc.stdout or ""
    err = (proc.stderr or "").strip()
    envelope = _parse_envelope(out)
    env_reason = envelope.get("reason")
    cost = envelope.get("cost", 0.0)
    try:
        cost = float(cost)
    except (TypeError, ValueError):
        cost = 0.0
    if proc.returncode == 0:
        return None, "", cost
    if isinstance(env_reason, str) and env_reason in KNOWN_REASONS:
        return env_reason, str(envelope.get("error") or err or out[-500:]), cost
    reason = EXIT_REASONS.get(proc.returncode, "worker_crash")
    detail = err or out[-500:].strip() or f"exit {proc.returncode}"
    return reason, detail, cost


def _attempt(worker_cmd: str, provider: str, model: str, task: dict, timeout_s: float) -> dict:
    """Run one worker attempt in a subprocess. Never raises."""
    task_id = str(task.get("id", ""))
    argv = [
        token.format(model=model, provider=provider, task_id=task_id)
        for token in shlex.split(worker_cmd)
    ]
    env = dict(os.environ)
    env.update(
        KERDOIOS_MODEL=model,
        KERDOIOS_PROVIDER=provider,
        KERDOIOS_TASK_ID=task_id,
    )
    start = time.time()
    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(task).encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout_s,
        )
        reason, error, cost = _classify(proc, timed_out=False)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        tail = (exc.stdout or b"").decode("utf-8", "replace")[-500:]
        return {
            "provider": provider, "model": model, "ok": False,
            "reason": "timeout", "error": f"attempt exceeded {timeout_s}s: {tail}",
            "cost": 0.0, "elapsed_s": round(elapsed, 2),
        }
    except Exception as exc:  # noqa: BLE001 — worker launch itself failed; still a result
        # The worker could not even start (bad command, OSError). Treat as
        # a crash of this attempt, not of the batch.
        return {
            "provider": provider, "model": model, "ok": False,
            "reason": "worker_crash", "error": f"{type(exc).__name__}: {exc}",
            "cost": 0.0, "elapsed_s": round(time.time() - start, 2),
        }
    elapsed = time.time() - start
    if reason is None:
        return {
            "provider": provider, "model": model, "ok": True,
            "reason": None, "output": (proc.stdout or "").decode("utf-8", "replace").strip(),
            "cost": cost, "elapsed_s": round(elapsed, 2),
        }
    return {
        "provider": provider, "model": model, "ok": False,
        "reason": reason, "error": error,
        "cost": cost, "elapsed_s": round(elapsed, 2),
    }


def _resolve_policy(placement: dict, overrides: dict) -> dict:
    policy = dict(DEFAULT_RETRY_POLICY)
    policy.update(placement.get("retry_policy") or {})
    policy.update({k: v for k, v in overrides.items() if v is not None})
    return policy


def _run_task(plan_task: tuple[dict, dict], worker_cmd: str, task_type: str,
              observed_log: Path | None, policy_overrides: dict) -> dict:
    """Execute one task through primary + fallbacks. Never raises: the
    batch contract is that every task yields a result or a failure stub."""
    placement, task = plan_task
    task_id = str(task.get("id", ""))
    chain = [{"provider": placement["provider"], "model": placement["model"]}]
    chain.extend(placement.get("fallbacks") or [])
    policy = _resolve_policy(placement, policy_overrides)
    max_retries = int(policy.get("max_retries", 0))
    backoff_s = float(policy.get("backoff_s", 10.0))
    timeout_s = float(policy.get("attempt_timeout_s", 300.0))
    retry_on = set(policy.get("retry_on", [])) or set(RETRYABLE_REASONS)

    attempts: list[dict] = []
    start = time.time()
    try:
        for cand in chain:
            for retry in range(max_retries + 1):
                outcome = _attempt(worker_cmd, cand["provider"], cand["model"], task, timeout_s)
                outcome["attempt"] = len(attempts) + 1
                attempts.append({k: v for k, v in outcome.items() if k != "output"})
                record(
                    Observation(
                        provider=cand["provider"],
                        model=cand["model"],
                        task_type=task.get("type", task_type),
                        completed=outcome["ok"],
                        actual_cost=outcome["cost"],
                        retried=len(attempts) > 1,
                        reason=outcome["reason"],
                    ),
                    path=observed_log,
                )
                if outcome["ok"]:
                    return {
                        "task_id": task_id, "ok": True,
                        "provider": cand["provider"], "model": cand["model"],
                        "output": outcome["output"], "reason": None,
                        "attempts": attempts,
                        "elapsed_s": round(time.time() - start, 2),
                    }
                if outcome["reason"] not in retry_on:
                    break  # deterministic failure: fail over, don't burn retries
                if retry < max_retries:
                    time.sleep(min(MAX_BACKOFF_S, backoff_s * (2 ** retry)))
            # inner loop ended without success (break or retries exhausted):
            # fall through to the next candidate in the chain
    except Exception as exc:  # noqa: BLE001 — per-task isolation: one task never kills the batch
        attempts.append({"reason": "worker_crash", "error": f"{type(exc).__name__}: {exc}"})
    last = attempts[-1] if attempts else {}
    return {
        "task_id": task_id, "ok": False,
        "provider": last.get("provider"), "model": last.get("model"),
        "error": last.get("error", "no attempts ran"),
        "reason": last.get("reason", "worker_crash"),
        "attempts": attempts,
        "elapsed_s": round(time.time() - start, 2),
    }


def run_batch(plan: dict, tasks: list[dict], worker_cmd: str, *,
              jobs: int | None = None, task_type: str = "unknown",
              observed_log: Path | str | None = None,
              max_retries: int | None = None, backoff_s: float | None = None,
              attempt_timeout_s: float | None = None,
              out: Path | str | None = None) -> list[dict]:
    """Run every task with healing. Returns one result dict per task, in
    task order. Always returns: failures become stubs, never exceptions."""
    placements = plan.get("placements") or []
    if not placements:
        raise ValueError("plan has no placements")
    assignments = [(placements[i % len(placements)], task) for i, task in enumerate(tasks)]
    policy_overrides = {
        "max_retries": max_retries,
        "backoff_s": backoff_s,
        "attempt_timeout_s": attempt_timeout_s,
    }
    log_path = Path(observed_log) if observed_log else None
    workers = jobs or min(32, max(1, len(tasks)))
    by_idx: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_run_task, item, worker_cmd, task_type, log_path, policy_overrides): i
            for i, item in enumerate(assignments)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                by_idx[i] = future.result()
            except Exception as exc:  # noqa: BLE001 — absolute last resort; the batch completes
                task = tasks[i]
                by_idx[i] = {
                    "task_id": str(task.get("id", i)), "ok": False,
                    "provider": None, "model": None,
                    "error": f"healer bug: {type(exc).__name__}: {exc}",
                    "reason": "worker_crash", "attempts": [],
                    "elapsed_s": 0.0,
                }
    results = [by_idx[i] for i in range(len(tasks))]
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            for result in results:
                fh.write(json.dumps(result) + "\n")
    ok = sum(1 for r in results if r["ok"])
    print(f"heal: {ok}/{len(results)} ok, "
          f"{sum(len(r['attempts']) for r in results)} attempts", file=sys.stderr)
    return results
