"""Live per-lane capacity: what each execution lane can absorb right now.

Kerdoios owns provider quota truth; this module folds every capacity source the
host has into one ``kerdoios.capacity.v1`` document that a dispatcher can read:

* ``burn-pace --json`` for paid subscription pools (band, agent ceiling, hourly
  budget, reset). Paid pools should finish each window near 100%, so the
  useful number is the *headroom*: sustainable %/h minus observed %/h.
* Hermes "You've used $X of your $Y cap" notices for the Nous Portal spend cap,
  which burn-pace does not track.
* The offline rate-limit catalog (``catalog.sqlite`` + ``kerdoios-historical.json``)
  for measured latency, throttle and retry-after evidence, and provider headers.
* A published free-tier catalog (``catalog.json``) for documented RPM/RPD/TPM.
* Kitchen ``runs.jsonl`` files for measured per-lane batch outcomes.
* A small live state file (``observe``) where executors report throttles, so a
  lane that just hit a quota is cooled down for every later caller.

Every limit carries ``classification`` (measured | published | inferred |
unknown), ``confidence`` and ``evidence``. Unknown never means unlimited: an
unknown lane gets one trial slot until outcomes say otherwise.

Nothing here executes a model. The only network call is an optional GET of a
provider's model list, so retired model ids are caught before dispatch. A listed
model is necessary, not sufficient: on 2026-10-20 NVIDIA's /v1/models listed
nemotron-3-super and kimi-k2.6, yet both answered 404 "Function not found for
account". Executors must report such failures through ``observe``.
"""

from __future__ import annotations

import fcntl
import glob
import json
import math
import os
import re
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

CAPACITY_SCHEMA = "kerdoios.capacity.v1"
STATE_SCHEMA = "kerdoios.capacity_state.v1"

# Execution lanes the factory can dispatch to. ``pool`` groups lanes that share
# one quota (haiku and luna both spend the Nous cap). ``data_policy`` is the
# provider's published retention stance, which the dispatcher maps to privacy.
LANE_SPECS: dict[str, dict[str, Any]] = {
    "opus": {"pool": "claude", "provider": "anthropic", "model": "claude-opus-5-5",
             "cost_class": "subscription", "catalog_provider": "anthropic", "data_policy": "subscription"},
    "sol": {"pool": "codex", "provider": "openai-codex", "model": "gpt-6.1-sol",
            "cost_class": "subscription", "catalog_provider": "openai-codex", "data_policy": "subscription"},
    "haiku": {"pool": "nous", "provider": "nous", "model": "anthropic/claude-haiku-5.5",
              "cost_class": "capped", "catalog_provider": "nous", "data_policy": "subscription",
              "usd_per_job_prior": 0.04},
    "luna": {"pool": "nous", "provider": "nous", "model": "openai/gpt-6-luna",
             "cost_class": "capped", "catalog_provider": "nous", "data_policy": "subscription",
             "usd_per_job_prior": 0.06},
    "ds": {"pool": "opencode-free:deepseek-v4.1-flash-free", "provider": "opencode-zen",
           "model": "deepseek-v4.1-flash-free", "cost_class": "free", "catalog_provider": "opencode-zen",
           "data_policy": "zero-retention", "kitchen_lane": "ds"},
    "glm": {"pool": "opencode-free:glm-5.3-flash-free", "provider": "opencode-zen",
            "model": "glm-5.3-flash-free", "cost_class": "free", "catalog_provider": "opencode-zen",
            "data_policy": "zero-retention", "kitchen_lane": "glm"},
    "minimax": {"pool": "opencode-free:minimax-m3.1-flash-free", "provider": "opencode-zen",
                "model": "minimax-m3.1-flash-free", "cost_class": "free", "catalog_provider": "opencode-zen",
                "data_policy": "unstated", "kitchen_lane": "minimax"},
    "groq": {"pool": "groq-free:openai/gpt-oss-120b", "provider": "groq", "model": "openai/gpt-oss-120b",
             "cost_class": "free", "catalog_provider": "groq", "data_policy": "unstated",
             "tokens_per_job_prior": 2500, "models_url": "https://api.groq.com/openai/v1/models",
             "key_env": "GROQ_API_KEY"},
    "nim": {"pool": "nvidia-free", "provider": "nvidia", "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "cost_class": "free", "catalog_provider": "nvidia", "data_policy": "trial-no-confidential",
            "tokens_per_job_prior": 2500, "models_url": "https://integrate.api.nvidia.com/v1/models",
            "key_env": "NVIDIA_API_KEY"},
    "cerebras": {"pool": "cerebras", "provider": "cerebras", "model": "gpt-oss-120b",
                 "cost_class": "metered", "catalog_provider": "cerebras", "data_policy": "unstated",
                 "tokens_per_job_prior": 2500, "usd_per_job_prior": 0.002},
}

# Measured or owner-observed facts that no file on disk states yet. Each is cited.
LANE_EVIDENCE: dict[str, dict[str, Any]] = {
    "ds": {
        "max_concurrency": (2, "measured", "medium",
                            "kitchen waves 2026-10-19: 2 concurrent ran clean; 4 concurrent rate-limited about half"),
        "window_jobs": (30, "inferred", "low",
                        "'Rate limit exceeded' after ~35 long jobs at both 2 and 4 concurrent: a quota, not a concurrency cap"),
    },
    "haiku": {"max_concurrency": (3, "measured", "medium", "kitchen wave 2: 3 concurrent Haiku QA jobs, no 429")},
    "luna": {"max_concurrency": (3, "inferred", "low", "same Nous pool as haiku; kitchen preflight only")},
}

DEFAULT_STATE_PATH = Path.home() / ".hermes" / "cache" / "kerdoios" / "capacity_state.json"
NOTICE = re.compile(r"You.ve used \$([0-9][0-9,]*\.?[0-9]*) of your \$([0-9][0-9,]*\.?[0-9]*) cap")
BURN_PACE_TTL_S = 600
COOLDOWN_CAP_S = 3600
SUSPECT_QUOTA_AFTER = 3  # consecutive throttles after backoff mean a spent window
TRIAL_STEP, TRIAL_MAX = 5, 4  # a low-confidence free lane earns one slot per 5 clean jobs, up to 4


def _limit(value: Any, classification: str, confidence: str, evidence: Any) -> dict[str, Any]:
    return {"value": value, "classification": classification, "confidence": confidence, "evidence": evidence}


def _unknown(reason: str) -> dict[str, Any]:
    return _limit(None, "unknown", "none", reason)


# ---------------------------------------------------------------- live state

def state_path() -> Path:
    override = os.environ.get("KERDOIOS_CAPACITY_STATE")
    return Path(override).expanduser() if override else DEFAULT_STATE_PATH


@contextmanager
def _locked_state(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = load_state(path)
        yield state
        state["saved_at"] = time.time()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)


def load_state(path: Path | None = None) -> dict[str, Any]:
    target = path or state_path()
    try:
        state = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        state = {"schema": STATE_SCHEMA}
    state.setdefault("lanes", {})
    return state


def observe(lane: str, status: str, *, retry_after_s: float | None = None, now: float | None = None,
            nous_used: float | None = None, nous_cap: float | None = None,
            path: Path | None = None) -> dict[str, Any]:
    """Record one executor outcome. Throttles set a cooldown every caller honours.

    ``status``: ok | rate_limited | error. A provider Retry-After wins; otherwise
    backoff doubles per consecutive throttle from 60 s. After
    SUSPECT_QUOTA_AFTER consecutive throttles the window is treated as spent and
    the lane rests for COOLDOWN_CAP_S, because retrying a spent quota only
    produces a 429 storm.
    """
    clock = time.time() if now is None else now
    with _locked_state(path or state_path()) as state:
        row = state["lanes"].setdefault(lane, {"events": [], "consecutive_throttles": 0})
        row["events"] = [e for e in row.get("events", []) if clock - e[0] < 48 * 3600]
        row["events"].append([clock, status])
        if status == "rate_limited":
            row["consecutive_throttles"] = int(row.get("consecutive_throttles", 0)) + 1
            n = row["consecutive_throttles"]
            if n >= SUSPECT_QUOTA_AFTER:
                wait = COOLDOWN_CAP_S
                ok_since = _ok_since_last_exhaustion(row["events"][:-1])
                row["quota_suspected"] = {"at": clock, "ok_jobs_before": ok_since}
            else:
                wait = retry_after_s if retry_after_s and retry_after_s > 0 else 60 * 2 ** (n - 1)
            row["cooldown_until"] = max(float(row.get("cooldown_until") or 0), clock + min(wait, COOLDOWN_CAP_S))
        elif status == "ok":
            row["consecutive_throttles"] = 0
            row["last_ok"] = clock
        if nous_used is not None and nous_cap is not None:
            state["nous_cap"] = {"used_usd": nous_used, "cap_usd": nous_cap, "observed_at": clock,
                                 "source": f"executor notice via {lane}"}
        return dict(row)


def _ok_since_last_exhaustion(events: list) -> int:
    """Successful jobs between the previous throttle run and the current one: the learned window size."""
    count, in_current_run = 0, True
    for _, status in reversed(events):
        if status == "rate_limited":
            if in_current_run:
                continue
            break
        in_current_run = False
        count += status == "ok"
    return count


# ---------------------------------------------------------------- sources

def burn_pace(cache: dict[str, Any] | None = None, *, command: Iterable[str] = ("burn-pace", "--json"),
              now: float | None = None) -> dict[str, Any]:
    """Read burn-pace, reusing a cached reading for BURN_PACE_TTL_S (it shells out to codexbar)."""
    clock = time.time() if now is None else now
    if cache and clock - float(cache.get("read_at", 0)) < BURN_PACE_TTL_S and cache.get("data"):
        return cache["data"]
    try:
        out = subprocess.run(list(command), capture_output=True, text=True, timeout=120).stdout
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return (cache or {}).get("data") or {}
    if cache is not None:
        cache.update({"read_at": clock, "data": data})
    return data


def latest_nous_notice(patterns: Iterable[str]) -> dict[str, Any] | None:
    """Newest Hermes spend-cap notice among files matching ``patterns`` (by mtime)."""
    files = sorted({f for p in patterns for f in glob.glob(os.path.expanduser(p), recursive=True)},
                   key=lambda f: os.path.getmtime(f), reverse=True)
    for name in files[:400]:
        try:
            hits = NOTICE.findall(Path(name).read_text(errors="replace"))
        except OSError:
            continue
        if hits:
            used, cap = (float(x.replace(",", "")) for x in hits[-1])
            return {"used_usd": used, "cap_usd": cap, "observed_at": os.path.getmtime(name), "source": name}
    return None


def catalog_evidence(db: Path | None) -> dict[str, dict[str, Any]]:
    """Per catalog-provider latency and throttle evidence from the offline catalog."""
    if not db or not Path(db).exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for prov, n, med in con.execute(
                "SELECT provider, count(*), avg(duration_ms) FROM requests WHERE duration_ms > 0 GROUP BY provider"):
            out.setdefault(prov, {})["requests"] = {"n": n, "mean_duration_ms": round(med or 0)}
        for prov, kind, n, ra_avg, ra_max in con.execute(
                "SELECT provider, kind, count(*), avg(retry_after_s), max(retry_after_s) FROM events "
                "WHERE primary_observation = 1 GROUP BY provider, kind"):
            row = out.setdefault(prov, {}).setdefault("events", {})
            row[kind] = {"n": n, "retry_after_avg_s": None if ra_avg is None else round(ra_avg, 1),
                         "retry_after_max_s": ra_max}
    finally:
        con.close()
    return out


def historical_headers(path: Path | None) -> dict[tuple[str, str], dict[str, float]]:
    """Provider-header limits from the catalog's kerdoios.quota_ledger.v1 export (limits only)."""
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    out: dict[tuple[str, str], dict[str, float]] = {}
    for entry in data.get("entries", []):
        dims = (entry.get("quota") or {}).get("dimensions") or {}
        limits = {k: v["limit"] for k, v in dims.items()
                  if isinstance(v, dict) and v.get("source") == "provider_header" and v.get("limit")}
        if limits:
            out[(entry.get("provider", ""), entry.get("model", ""))] = limits
    return out


def published_limits(path: Path | None) -> dict[str, dict[str, Any]]:
    """Published free-tier limits keyed by lane provider/model from catalog.json."""
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    out: dict[str, dict[str, Any]] = {}
    for p in data.get("llm_apis", []):
        for m in p.get("models", []) or [{}]:
            key = f"{p.get('provider')}/{m.get('model', '*')}"
            out[key] = {**{k: m.get(k) for k in ("rpm", "rpd", "tpm", "tpd")},
                        "source_ids": p.get("source_ids"), "data_policy": p.get("data_policy")}
    return out


def kitchen_outcomes(patterns: Iterable[str]) -> dict[str, dict[str, int]]:
    """Measured per-lane job outcomes from kitchen runs.jsonl logs (latest status per job)."""
    latest: dict[tuple[str, str], str] = {}
    for name in sorted({f for p in patterns for f in glob.glob(os.path.expanduser(p))}):
        try:
            for line in Path(name).read_text().splitlines():
                row = json.loads(line)
                latest[(name, row["id"])] = (row.get("lane") or "?") + "\t" + (row.get("status") or "?")
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    out: dict[str, dict[str, int]] = {}
    for value in latest.values():
        lane, status = value.split("\t")
        out.setdefault(lane, {}).setdefault(status, 0)
        out[lane][status] += 1
    return out


def live_models(spec: Mapping[str, Any], env: Mapping[str, str]) -> list[str] | None:
    """GET the provider's model list; None when unreachable or no key (never inference)."""
    from .providers.http import get_json

    url, key = spec.get("models_url"), env.get(spec.get("key_env", ""))
    if not url or not key:
        return None
    body = get_json(url, api_key=key, timeout=15)
    if not body:
        return None
    return sorted(str(m.get("id")) for m in body.get("data", []) if isinstance(m, dict) and m.get("id"))


def load_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE pairs from a dotenv file. Values stay in memory, never in output."""
    env: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*)\s*$", line)
            if m:
                env[m.group(1)] = m.group(2).strip().strip("'\"")
    except OSError:
        pass
    return env


# ---------------------------------------------------------------- model

def recent_rate(history: Path | None, pool: str, *, now: float, span_h: float = 6.0) -> float | None:
    """Weekly-%/h over the last ``span_h`` hours from burn-pace's own sample log (>= 1 h of samples).

    burn-pace bands use the window *average*, which lags a burst: on 2026-10-20 the
    average said 0.42 %/h while the last three hours ran at 1.66 %/h. Reported beside
    the band as a warning; the band stays burn-pace's call.
    """
    if not history or not Path(history).exists():
        return None
    from datetime import datetime

    points = []
    for line in Path(history).read_text().splitlines()[-2000:]:
        try:
            row = json.loads(line)
            ts = datetime.fromisoformat(row["at"]).timestamp()
            pct = float(row["lanes"][pool]["weekly_used_pct"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if now - span_h * 3600 <= ts <= now:
            points.append((ts, pct))
    if len(points) < 2 or points[-1][0] - points[0][0] < 3600 or points[-1][1] < points[0][1]:
        return None
    return round((points[-1][1] - points[0][1]) / ((points[-1][0] - points[0][0]) / 3600), 3)


def _subscription_lane(spec: Mapping[str, Any], pace: Mapping[str, Any],
                       recent: float | None = None) -> dict[str, Any]:
    pool = (pace.get("lanes") or {}).get(spec["pool"])
    z = pace.get("z0_factory") or {}
    if not pool:
        return {"limits": {"max_concurrency": _unknown("burn-pace has no reading for this pool")},
                "pacing": None, "blocked": "no burn-pace reading"}
    avoid = spec["pool"] in (z.get("avoid") or [])
    agents = int(pool.get("max_workflow_agents") or 0)
    headroom = max(0.0, float(pool.get("sustainable_pct_per_h") or 0) - float(pool.get("observed_pct_per_h") or 0))
    pacing = {k: pool.get(k) for k in ("band", "weekly_used_pct", "hours_to_reset", "sustainable_pct_per_h",
                                       "observed_pct_per_h", "pace_ratio", "pct_budget_next_hour")}
    # The pool lands at 100% at reset if usage rises to the sustainable rate, so the
    # dispatchable budget is the gap between that rate and what is already burning.
    pacing["headroom_pct_per_h"] = round(headroom, 3)
    pacing["avoid"] = avoid
    pacing["recent_pct_per_h"] = recent
    if recent is not None and recent > float(pool.get("sustainable_pct_per_h") or 0):
        pacing["warning"] = (f"last hours burned {recent} %/h, above the sustainable "
                             f"{pool.get('sustainable_pct_per_h')} %/h; the band uses the window average")
    blocked = None
    if avoid or pool.get("band") == "hard-throttle" or agents <= 0:
        blocked = f"burn-pace band {pool.get('band')}{' (avoid)' if avoid else ''}"
    return {"limits": {"max_concurrency": _limit(agents, "measured", "medium",
                                                 "burn-pace max_workflow_agents (codexbar plan usage)")},
            "pacing": pacing, "blocked": blocked}


def _api_lane(spec: Mapping[str, Any], published: Mapping[str, Any], headers: Mapping[str, float],
              latency_ms: float | None) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    for dim in ("rpm", "rpd", "tpm", "tpd"):
        if headers.get(dim):
            limits[dim] = _limit(headers[dim], "measured", "medium", "provider_header (rate-limit catalog)")
        elif published.get(dim):
            limits[dim] = _limit(published[dim], "published", "medium",
                                 f"free-tier catalog sources {published.get('source_ids')}")
    tokens = float(spec.get("tokens_per_job_prior") or 2500)
    rate = [limits[d]["value"] for d in ("rpm",) if d in limits]
    if "tpm" in limits:
        rate.append(limits["tpm"]["value"] / tokens)
    jobs_per_min = min(rate) if rate else None
    if jobs_per_min:
        limits["jobs_per_min"] = _limit(round(jobs_per_min, 2), "inferred", "medium",
                                        f"min(rpm, tpm / {int(tokens)} tokens per job)")
        secs = (latency_ms or 2000) / 1000
        # Little's law: in-flight = arrival rate x latency, plus one for jitter.
        conc = max(1, min(8, math.ceil(jobs_per_min * secs / 60) + 1))
        limits["max_concurrency"] = _limit(conc, "inferred", "medium",
                                           f"Little's law at {jobs_per_min:.1f} jobs/min x {secs:.1f}s latency")
    else:
        limits["max_concurrency"] = _limit(1, "inferred", "low", "no published or measured rate; one trial slot")
    if "rpd" in limits:
        limits["window_jobs"] = _limit(limits["rpd"]["value"], limits["rpd"]["classification"], "medium", "requests/day")
    if "tpd" in limits:
        by_tokens = int(limits["tpd"]["value"] / tokens)
        if "window_jobs" not in limits or by_tokens < limits["window_jobs"]["value"]:
            limits["window_jobs"] = _limit(by_tokens, "inferred", "medium", f"tpd / {int(tokens)} tokens per job")
    return limits


def build_capacity(*, pace: Mapping[str, Any] | None = None, state: Mapping[str, Any] | None = None,
                   burn_pace_history: Path | None = None,
                   catalog_db: Path | None = None, historical: Path | None = None,
                   free_tier: Path | None = None, notice_globs: Iterable[str] = (),
                   kitchen_runs: Iterable[str] = (), models: Mapping[str, list[str] | None] | None = None,
                   allow_metered: bool = False, now: float | None = None) -> dict[str, Any]:
    clock = time.time() if now is None else now
    pace = pace or {}
    state = state or {"lanes": {}}
    evidence = catalog_evidence(catalog_db)
    headers = historical_headers(historical)
    published = published_limits(free_tier)
    measured = kitchen_outcomes(kitchen_runs)
    notice = latest_nous_notice(notice_globs)
    live_cap = state.get("nous_cap")
    if live_cap and (not notice or live_cap.get("observed_at", 0) >= notice.get("observed_at", 0)):
        notice = live_cap
    lanes: dict[str, Any] = {}
    for lane, spec in LANE_SPECS.items():
        ev = evidence.get(spec["catalog_provider"], {})
        latency = (ev.get("requests") or {}).get("mean_duration_ms")
        row: dict[str, Any] = {k: spec[k] for k in ("pool", "provider", "model", "cost_class", "data_policy")}
        row["evidence"] = {"catalog": ev or None, "kitchen_runs": measured.get(spec.get("kitchen_lane", lane))}
        row["latency_prior_ms"] = latency
        blocked = None
        if spec["cost_class"] == "subscription":
            sub = _subscription_lane(spec, pace, recent_rate(burn_pace_history, spec["pool"], now=clock))
            row["limits"], row["pacing"], blocked = sub["limits"], sub["pacing"], sub["blocked"]
        elif spec["cost_class"] == "capped":
            fixed = LANE_EVIDENCE.get(lane, {}).get("max_concurrency")
            row["limits"] = {"max_concurrency": _limit(*fixed) if fixed else _limit(1, "inferred", "low", "trial slot")}
            nous_429 = (ev.get("events") or {}).get("rate_limit_429")
            if nous_429:
                row["limits"]["retry_after_s"] = _limit(nous_429["retry_after_max_s"], "measured", "medium",
                                                        f"max retry_after over {nous_429['n']} catalog 429s")
            if notice:
                remaining = round(notice["cap_usd"] - notice["used_usd"], 2)
                row["spend"] = {**notice, "remaining_usd": remaining,
                                "usd_per_job_prior": spec.get("usd_per_job_prior")}
                if remaining <= 0:
                    blocked = "Nous spend cap reached"
            else:
                row["spend"] = None
                blocked = "no Nous spend-cap notice found; refusing to spend blind"
        else:
            key = f"{spec['provider']}/{spec['model']}"
            pub = published.get(key) or published.get(f"{spec['provider']}/*") or {}
            if lane in LANE_EVIDENCE:
                row["limits"] = {k: _limit(*v) for k, v in LANE_EVIDENCE[lane].items()}
            elif spec.get("tokens_per_job_prior"):
                row["limits"] = _api_lane(spec, pub, headers.get((spec["provider"], spec["model"]), {}), latency)
            else:
                row["limits"] = {"max_concurrency": _limit(1, "inferred", "low",
                                                           "no published limit or measured run; one trial slot")}
            if spec["cost_class"] == "metered":
                row["spend"] = {"usd_per_job_prior": spec.get("usd_per_job_prior"), "balance": "unknown"}
                if not allow_metered:
                    blocked = "metered lane with unknown balance; pass allow_metered to use"
        if models is not None and lane in models:
            listed = models[lane]
            row["model_listed"] = None if listed is None else spec["model"] in listed
            if listed is not None and spec["model"] not in listed:
                blocked = f"model {spec['model']} not in provider /v1/models (retired?)"
        # Live overlay: cooldowns and window usage reported by executors.
        live = (state.get("lanes") or {}).get(lane, {})
        events = [e for e in live.get("events", []) if clock - e[0] < 24 * 3600]
        used_24h = sum(1 for e in events if e[1] in ("ok", "error"))
        cooldown = float(live.get("cooldown_until") or 0)
        row["state"] = {"jobs_24h": used_24h, "throttles_24h": sum(1 for e in events if e[1] == "rate_limited"),
                        "cooldown_until": cooldown if cooldown > clock else None,
                        "quota_suspected": live.get("quota_suspected")}
        if cooldown > clock and not blocked:
            blocked = f"cooling down {int(cooldown - clock)}s after throttling"
        ok_24h = sum(1 for e in events if e[1] == "ok")
        mc = row["limits"].get("max_concurrency", {})
        if mc.get("confidence") in ("low", "none") and spec["cost_class"] == "free" and ok_24h >= TRIAL_STEP \
                and row["state"]["throttles_24h"] == 0:
            # Additive increase on clean evidence; any throttle in 24 h drops back to the trial slot.
            grown = min(TRIAL_MAX, int(mc.get("value") or 1) + ok_24h // TRIAL_STEP)
            row["limits"]["max_concurrency"] = _limit(grown, "measured", "low",
                                                      f"{ok_24h} clean jobs in 24 h, no throttles (additive increase)")
        conc = row["limits"].get("max_concurrency", {}).get("value") or 0
        window = row["limits"].get("window_jobs", {}).get("value")
        window_left = None if window is None else max(0, int(window) - used_24h)
        row["window_jobs_left"] = window_left
        slots = 0 if blocked else int(conc)
        if live.get("quota_suspected") and int(live.get("consecutive_throttles") or 0) >= SUSPECT_QUOTA_AFTER:
            # After a spent window, probe with one request; parallel probes only burn throttles.
            slots = min(slots, 1)
        if window_left is not None:
            slots = min(slots, window_left)
        row["slots_now"] = slots
        row["available_now"] = slots > 0
        row["reason"] = blocked or ("window spent" if window_left == 0 else "ok")
        lanes[lane] = row
    return {"schema": CAPACITY_SCHEMA, "at": clock, "lanes": lanes,
            "burn_pace": {"at": pace.get("at"), "z0_factory": pace.get("z0_factory")},
            "nous_cap": notice,
            "sources": {"catalog_db": str(catalog_db) if catalog_db else None,
                        "historical": str(historical) if historical else None,
                        "free_tier": str(free_tier) if free_tier else None,
                        "notice_globs": list(notice_globs), "kitchen_runs": list(kitchen_runs)}}


def default_sources() -> dict[str, Any]:
    """Source paths from env, so the CLI and the dispatcher read the same evidence."""
    rl = os.environ.get("KERDOIOS_RATE_LIMIT_DIR")
    globs = lambda name: [g for g in os.environ.get(name, "").split(os.pathsep) if g]
    return {
        "catalog_db": Path(rl) / "catalog.sqlite" if rl else None,
        "historical": Path(rl) / "kerdoios-historical.json" if rl else None,
        "free_tier": Path(os.environ["KERDOIOS_FREE_TIER_CATALOG"]) if os.environ.get("KERDOIOS_FREE_TIER_CATALOG") else None,
        "notice_globs": globs("KERDOIOS_NOTICE_GLOBS"),
        "kitchen_runs": globs("KERDOIOS_KITCHEN_RUNS"),
        "burn_pace_history": Path(os.environ.get(
            "KERDOIOS_BURN_PACE_HISTORY", Path.home() / ".z0int/state/posture/burn-pace.jsonl")),
    }


def current_capacity(*, live_model_check: bool = False, allow_metered: bool = False,
                     env_file: Path | None = None, now: float | None = None) -> dict[str, Any]:
    """Capacity right now from every configured source; caches burn-pace in the state file."""
    path = state_path()
    with _locked_state(path) as state:
        cache = state.setdefault("burn_pace_cache", {})
        pace = burn_pace(cache, now=now)
        snapshot = json.loads(json.dumps(state))
    models = None
    if live_model_check:
        env = {**load_env_file(env_file or Path.home() / ".hermes" / ".env"), **os.environ}
        models = {lane: live_models(spec, env) for lane, spec in LANE_SPECS.items() if spec.get("models_url")}
    return build_capacity(pace=pace, state=snapshot, models=models, allow_metered=allow_metered,
                          now=now, **default_sources())
