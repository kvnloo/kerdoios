"""Subscription plan windows -> posture-aware ResourceOffers.

Flat-rate plans (Claude, Codex, Cursor, Grok, ...) are sunk cost. What matters
is time: unused window capacity perishes at reset (BURN it on real work), and a
window the observed pace empties before reset should shed bounded work
(OFFLOAD). This adapter reads that picture from local files only:

* ``z0int posture --json`` when z0int is installed (it applies host overrides
  such as a corrected reset time), else
* the usage-island CodexBar cache ``~/.cache/codexbar-waybar/last.json``,
  evaluated here with the same rules as z0int resource posture v0.

Offline and fail-open: a missing, unreadable, or stale snapshot yields no
offers and a reason. Credentials are never read. Account identity fields
(e-mail, organization, login method) are never copied.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..types import (
    CapabilityProfile,
    Capacity,
    Economics,
    ResourceOffer,
    SubscriptionWindow,
    Telemetry,
)

SOURCE_ENV = "KERDOIOS_POSTURE_SOURCE"  # auto | z0int | codexbar | off
CODEXBAR_ENV = "KERDOIOS_CODEXBAR"
Z0INT_TIMEOUT_S = 10.0

# Same defaults as z0int resource posture v0 (docs/resource-posture.md).
OFFLOAD_RATIO = 1.0
RESERVE_RATIO = 0.85
BURN_HORIZON_HOURS = 24.0
BURN_MIN_SURPLUS_FRAC = 0.20
MIN_OBSERVATION_HOURS = 0.5
MAX_OBSERVATION_AGE_HOURS = 6.0
_GROUP_PRECEDENCE = {"OFFLOAD": 3, "RESERVE": 2, "BURN": 1, "BALANCED": 0}

# Plan groups that become offers. OpenRouter credits are reported but not
# offered: they are metered money, already covered by the OpenRouter adapter.
_PROFILES: dict[str, tuple[CapabilityProfile, int]] = {
    "claude": (CapabilityProfile(reasoning=0.93, coding=0.95, tool_use=0.97, provenance="provider_claim"), 200_000),
    "codex": (CapabilityProfile(reasoning=0.92, coding=0.94, tool_use=0.95, provenance="provider_claim"), 272_000),
    "cursor": (CapabilityProfile(reasoning=0.90, coding=0.92, tool_use=0.95, provenance="provider_claim"), 200_000),
    "grok": (CapabilityProfile(reasoning=0.88, coding=0.85, tool_use=0.90, provenance="provider_claim"), 256_000),
    "gemini": (CapabilityProfile(reasoning=0.90, coding=0.88, tool_use=0.90, provenance="provider_claim"), 1_000_000),
    "copilot": (CapabilityProfile(reasoning=0.85, coding=0.88, tool_use=0.90, provenance="provider_claim"), 128_000),
}
# A plan window is a harness session, not an API pool: keep it narrow so one
# BURN verdict cannot absorb an entire wide portfolio.
_CONCURRENCY = 2

# Fitness multipliers applied in score(). BURN scales with what would perish.
OFFLOAD_FACTOR = 0.5
RESERVE_FACTOR = 0.85
BURN_MAX_BOOST = 0.5


@dataclass
class WindowReport:
    status: str  # ok | disabled | unavailable | stale | empty
    source: str  # z0int | codexbar | none
    reason: str
    windows: list[SubscriptionWindow] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    snapshot_at: str | None = None
    factory_posture: str | None = None

    @property
    def usable(self) -> bool:
        return self.status == "ok" and bool(self.windows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "reason": self.reason,
            "snapshot_at": self.snapshot_at,
            "factory_posture": self.factory_posture,
            "windows": [w.to_dict() for w in self.windows],
            "skipped": list(self.skipped),
        }


# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------


def _parse_time(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _hours(h: float | None) -> str:
    if h is None:
        return "?"
    return f"{h / 24:.1f}d" if h >= 48 else f"{h:.1f}h"


def _window_label(minutes: Any, fallback: str) -> str:
    if not isinstance(minutes, (int, float)) or minutes <= 0:
        return fallback
    if minutes % 10080 == 0:
        return "weekly" if minutes == 10080 else f"{int(minutes // 10080)}w"
    if minutes % 1440 == 0:
        return f"{int(minutes // 1440)}d"
    return f"{int(minutes // 60)}h" if minutes % 60 == 0 else f"{int(minutes)}m"


# ---------------------------------------------------------------------------
# pure evaluation (CodexBar path)
# ---------------------------------------------------------------------------


def evaluate_window(
    *,
    group: str,
    window: str,
    used_percent: float,
    resets_at: str,
    window_minutes: float | None,
    observed_at: str | None,
    now: datetime,
) -> SubscriptionWindow | str:
    """One window -> SubscriptionWindow, or a skip reason string."""
    reset = _parse_time(resets_at)
    if reset is None:
        return f"{group}:{window}: no reset time"
    hours_left = (reset - now).total_seconds() / 3600
    if hours_left <= 0:
        return f"{group}:{window}: snapshot predates reset at {resets_at}; remaining unknown"
    observed = _parse_time(observed_at)
    age_h = (now - observed).total_seconds() / 3600 if observed else None
    if age_h is None or age_h > MAX_OBSERVATION_AGE_HOURS:
        age = "unknown" if age_h is None else _hours(age_h)
        return f"{group}:{window}: stale snapshot (age {age} > {MAX_OBSERVATION_AGE_HOURS:.0f}h)"
    remaining_pct = max(0.0, 100.0 - float(used_percent))
    base = dict(
        group=group,
        window=window,
        remaining_fraction=round(remaining_pct / 100.0, 4),
        resets_at=resets_at,
        seconds_until_reset=round(hours_left * 3600, 1),
        window_minutes=window_minutes,
        observed_at=observed_at,
        source="codexbar",
    )
    if remaining_pct <= 0:
        return SubscriptionWindow(
            **base, posture="OFFLOAD", reason="exhausted",
            arithmetic=f"remaining 0.0% = exhausted for {_hours(hours_left)} until reset",
        )
    rate = None
    elapsed_h = None
    if isinstance(window_minutes, (int, float)) and window_minutes > 0:
        # Window average since the window started: the only rate one snapshot supports.
        elapsed_h = window_minutes / 60 - (reset - (observed or now)).total_seconds() / 3600
        if elapsed_h > 0:
            rate = float(used_percent) / elapsed_h
    if rate is None:
        return SubscriptionWindow(
            **base, posture="BALANCED", reason="no_rate_observation",
            arithmetic=f"remaining {remaining_pct:.1f}%; no burn-rate observation",
        )
    projected = rate * hours_left
    surplus = remaining_pct - projected
    ratio = projected / remaining_pct
    arithmetic = (
        f"projected {rate:.3g}%/h x {_hours(hours_left)} = {projected:.1f}% vs remaining "
        f"{remaining_pct:.1f}% (ratio {ratio:.2f})"
    )
    if ratio >= OFFLOAD_RATIO:
        posture, reason = "OFFLOAD", "runs_out_before_reset"
        arithmetic += f"; runs out in {_hours(remaining_pct / rate)}, before reset"
    elif ratio >= RESERVE_RATIO:
        posture, reason = "RESERVE", "tight_until_reset"
    elif hours_left <= BURN_HORIZON_HOURS and surplus / 100.0 >= BURN_MIN_SURPLUS_FRAC:
        posture, reason = "BURN", "surplus_perishes_at_reset"
        arithmetic += f"; {surplus:.1f}% perishes at reset in {_hours(hours_left)}"
    else:
        posture, reason = "BALANCED", "on_pace"
    if posture in ("BURN", "OFFLOAD") and elapsed_h is not None and elapsed_h < MIN_OBSERVATION_HOURS:
        arithmetic += f"; {posture} not asserted (rate observed < {MIN_OBSERVATION_HOURS}h)"
        return SubscriptionWindow(
            **base, posture="BALANCED", reason=reason + "_unconfirmed", arithmetic=arithmetic,
            surplus_fraction_at_reset=round(surplus / 100.0, 4), confidence="low",
        )
    return SubscriptionWindow(
        **base, posture=posture, reason=reason, arithmetic=arithmetic,
        surplus_fraction_at_reset=round(surplus / 100.0, 4),
    )


def windows_from_codexbar(rows: Any, now: datetime) -> tuple[list[SubscriptionWindow], list[str]]:
    """CodexBar provider rows -> windows. Only ``usage`` rate-window fields are read;
    identity keys (accountEmail, accountOrganization, identity, loginMethod) never are."""
    windows: list[SubscriptionWindow] = []
    skipped: list[str] = []
    if not isinstance(rows, list):
        return windows, ["codexbar: cache is not a provider list"]
    for entry in rows:
        if not isinstance(entry, dict) or not isinstance(entry.get("usage"), dict):
            continue
        group = str(entry.get("provider") or "")
        if group not in _PROFILES:
            continue
        usage = entry["usage"]
        updated = usage.get("updatedAt") if isinstance(usage.get("updatedAt"), str) else None
        slots: list[tuple[str, dict[str, Any], bool]] = []
        for slot in ("primary", "secondary", "tertiary"):
            w = usage.get(slot)
            if isinstance(w, dict):
                slots.append((slot, w, True))
        for extra in usage.get("extraRateWindows") or []:
            if isinstance(extra, dict) and isinstance(extra.get("window"), dict):
                slots.append((str(extra.get("id") or "extra").removeprefix(group + "-"), extra["window"], False))
        seen: set[tuple[Any, Any, Any]] = set()
        labels: set[str] = set()
        for slot, w, standard in slots:
            used = w.get("usedPercent")
            if not isinstance(used, (int, float)) or isinstance(used, bool):
                continue
            reset, minutes = w.get("resetsAt"), w.get("windowMinutes")
            if standard:
                if (reset, minutes, used) in seen:
                    continue  # cursor repeats one window as primary/secondary/tertiary
                seen.add((reset, minutes, used))
            label = _window_label(minutes, slot) if standard else slot
            if label in labels:
                label = f"{label}-{slot}"
            labels.add(label)
            if not reset:
                skipped.append(f"{group}:{label}: no reset time")
                continue
            out = evaluate_window(
                group=group, window=label, used_percent=float(used), resets_at=str(reset),
                window_minutes=float(minutes) if isinstance(minutes, (int, float)) else None,
                observed_at=updated, now=now,
            )
            if isinstance(out, str):
                skipped.append(out)
            else:
                windows.append(out)
    return windows, skipped


# ---------------------------------------------------------------------------
# z0int path
# ---------------------------------------------------------------------------


def _z0int_command() -> list[str] | None:
    exe = shutil.which("z0int")
    if exe:
        return [exe]
    py = Path.home() / ".z0int" / "bin" / "python"
    if py.is_file() and os.access(py, os.X_OK):
        return [str(py), "-m", "z0int"]
    return None


def windows_from_z0int(doc: Any, now: datetime) -> tuple[list[SubscriptionWindow], list[str], str | None]:
    """``z0int posture --json`` pools -> windows (whitelisted fields only)."""
    if not isinstance(doc, dict) or not isinstance(doc.get("pools"), list):
        raise ValueError("not a z0int.resource_posture document")
    windows: list[SubscriptionWindow] = []
    skipped: list[str] = []
    for row in doc["pools"]:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "")
        group = str(row.get("group") or pid.split(":", 1)[0])
        if group not in _PROFILES or row.get("remaining") is None:
            continue
        window = pid.split(":", 1)[1] if ":" in pid else pid
        reset = _parse_time(row.get("resets_at"))
        if reset is None:
            skipped.append(f"{pid}: no reset time")
            continue
        if row.get("confidence") == "stale" or (reset - now).total_seconds() <= 0:
            skipped.append(f"{pid}: stale ({row.get('arithmetic') or row.get('reason')})")
            continue
        remaining = float(row["remaining"])
        surplus = row.get("projected_surplus_at_reset")
        wh = row.get("window_hours")
        windows.append(
            SubscriptionWindow(
                group=group,
                window=window,
                remaining_fraction=round(max(0.0, remaining) / 100.0, 4),
                resets_at=str(row.get("resets_at")),
                seconds_until_reset=round((reset - now).total_seconds(), 1),
                window_minutes=float(wh) * 60 if isinstance(wh, (int, float)) else None,
                posture=row.get("posture") if row.get("posture") in _GROUP_PRECEDENCE else "BALANCED",
                reason=str(row.get("reason") or ""),
                arithmetic=str(row.get("arithmetic") or ""),
                surplus_fraction_at_reset=round(float(surplus) / 100.0, 4) if isinstance(surplus, (int, float)) else None,
                confidence=str(row.get("confidence") or "ok"),
                observed_at=row.get("observed_at") if isinstance(row.get("observed_at"), str) else None,
                source="z0int",
            )
        )
    factory = doc.get("factory") if isinstance(doc.get("factory"), dict) else {}
    return windows, skipped, factory.get("posture") if isinstance(factory.get("posture"), str) else None


def _run_z0int(codexbar: Path | None) -> dict[str, Any]:
    cmd = _z0int_command()
    if cmd is None:
        raise FileNotFoundError("z0int not installed")
    # --no-kerdoios: z0int would otherwise read this plugin's own cache back.
    argv = cmd + ["posture", "--json", "--no-kerdoios"]
    if codexbar is not None:
        argv += ["--codexbar", str(codexbar)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=Z0INT_TIMEOUT_S, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"z0int posture exit {proc.returncode}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# collection (fail-open)
# ---------------------------------------------------------------------------


def codexbar_path() -> Path:
    raw = os.environ.get(CODEXBAR_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "codexbar-waybar" / "last.json"


def _finish(source: str, windows: list[SubscriptionWindow], skipped: list[str],
            factory: str | None = None) -> WindowReport:
    snap = max((w.observed_at for w in windows if w.observed_at), default=None)
    if windows:
        return WindowReport("ok", source, f"{len(windows)} live window(s)", windows, skipped, snap, factory)
    if skipped and all("stale" in s or "predates reset" in s for s in skipped):
        return WindowReport("stale", source, "every window is stale; plan unchanged", [], skipped, None, factory)
    return WindowReport("empty", source, "no subscription windows with a reset; plan unchanged", [], skipped, None, factory)


def collect(now: datetime | None = None, *, source: str | None = None,
            codexbar: Path | None = None) -> WindowReport:
    """Read subscription windows. Never raises; failure is a report with a reason."""
    now = now or datetime.now(timezone.utc)
    mode = (source or os.environ.get(SOURCE_ENV) or "auto").strip().lower()
    if mode in ("off", "0", "false", "no", "none"):
        return WindowReport("disabled", "none", f"{SOURCE_ENV}={mode}")
    path = codexbar or (codexbar_path() if os.environ.get(CODEXBAR_ENV) else None)
    z0int_note = ""
    if mode in ("auto", "z0int"):
        try:
            windows, skipped, factory = windows_from_z0int(_run_z0int(path), now)
            return _finish("z0int", windows, skipped, factory)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            if mode == "z0int":
                return WindowReport("unavailable", "z0int", f"z0int posture failed: {type(exc).__name__}; plan unchanged")
            z0int_note = f" (z0int: {type(exc).__name__})"
    path = path or codexbar_path()
    try:
        rows = json.loads(path.read_text())
    except OSError as exc:
        return WindowReport("unavailable", "codexbar",
                            f"no CodexBar snapshot at {path} ({type(exc).__name__}){z0int_note}; plan unchanged")
    except ValueError:
        return WindowReport("unavailable", "codexbar", f"unreadable CodexBar snapshot at {path}{z0int_note}; plan unchanged")
    windows, skipped = windows_from_codexbar(rows, now)
    report = _finish("codexbar", windows, skipped)
    if z0int_note:
        report.reason += z0int_note
    return report


# ---------------------------------------------------------------------------
# offers + effect
# ---------------------------------------------------------------------------


def _binding(windows: list[SubscriptionWindow]) -> SubscriptionWindow:
    """Binding window per z0int: OFFLOAD > RESERVE > BURN > BALANCED, then longest window."""
    return max(
        windows,
        key=lambda w: (_GROUP_PRECEDENCE[w.posture], w.window_minutes or 0, -(w.remaining_fraction)),
    )


def offers_from_report(report: WindowReport) -> list[ResourceOffer]:
    if not report.usable:
        return []
    by_group: dict[str, list[SubscriptionWindow]] = {}
    for w in report.windows:
        by_group.setdefault(w.group, []).append(w)
    offers: list[ResourceOffer] = []
    for group in sorted(by_group):
        bind = _binding(by_group[group])
        caps, context = _PROFILES[group]
        offers.append(
            ResourceOffer(
                id=f"subscription/{group}",
                provider=group,
                resource_type="llm",
                model="subscription",
                local=False,
                capabilities=caps,
                capacity=Capacity(concurrency=_CONCURRENCY, context_window=context),
                economics=Economics(
                    seconds_until_quota_reset=bind.seconds_until_reset,
                    subscription=bind,
                ),
                telemetry=Telemetry(latency_p50_ms=1500, latency_p95_ms=6000, failure_rate=0.02, availability=0.99),
                tools=("*",),
                source=f"subscription:{report.source}",
                confidence=0.6 if bind.confidence == "ok" else 0.4,
            )
        )
    return offers


def merge(offers: list[ResourceOffer], report: WindowReport) -> list[ResourceOffer]:
    """Add subscription offers; an existing subscription/<group> row is replaced."""
    extra = offers_from_report(report)
    if not extra:
        return offers
    ids = {o.id for o in extra}
    return [o for o in offers if o.id not in ids] + extra


def attach(offers: list[ResourceOffer], *, enabled: bool = True,
           now: datetime | None = None) -> tuple[list[ResourceOffer], WindowReport]:
    """Offers plus live subscription windows, and the report explain prints."""
    if not enabled:
        return offers, WindowReport("disabled", "none", "subscription windows not requested")
    report = collect(now)
    return merge(offers, report), report


def posture_factor(window: SubscriptionWindow) -> float:
    if window.posture == "BURN":
        surplus = max(0.0, min(1.0, window.surplus_fraction_at_reset or 0.0))
        return 1.0 + BURN_MAX_BOOST * surplus
    if window.posture == "OFFLOAD":
        return OFFLOAD_FACTOR
    if window.posture == "RESERVE":
        return RESERVE_FACTOR
    return 1.0


def posture_reason(window: SubscriptionWindow) -> str:
    factor = posture_factor(window)
    head = f"{window.posture} {window.group}:{window.window}"
    hours = window.seconds_until_reset / 3600 if window.seconds_until_reset is not None else None
    left = f"{window.remaining_fraction * 100:.0f}% left, resets in {_hours(hours)}"
    if window.posture == "BURN":
        perish = (window.surplus_fraction_at_reset or 0.0) * 100
        return f"{head}: {perish:.0f}% perishes at reset in {_hours(hours)} (fitness x{factor:.2f})"
    if window.posture == "OFFLOAD":
        return f"{head}: over pace, {left} (fitness x{factor:.2f})"
    if window.posture == "RESERVE":
        return f"{head}: tight, {left} (fitness x{factor:.2f})"
    return f"{head}: on pace, {left}"

