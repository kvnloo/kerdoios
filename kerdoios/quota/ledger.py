"""Durable, restart-surviving quota ledger.

The ledger is a cache/index over receipts, never the source of truth: it can be
thrown away and rebuilt from Tokenomics receipts (see ``receipts.py``). Writes
are atomic (temp file + fsync + ``os.replace``) so a crash mid-write cannot
truncate prior state. Only typed quota fields are serialized, so credentials
have no path into the file.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model import QuotaState

LEDGER_SCHEMA = "kerdoios.quota_ledger.v1"
DEFAULT_LEDGER_PATH = Path(
    os.environ.get(
        "KERDOIOS_QUOTA_LEDGER",
        str(Path.home() / ".hermes" / "cache" / "kerdoios" / "quota_ledger.json"),
    )
)


def default_ledger_path() -> Path:
    """Resolved lazily so an env override set after import still applies."""
    override = os.environ.get("KERDOIOS_QUOTA_LEDGER")
    if override:
        return Path(override).expanduser()
    return DEFAULT_LEDGER_PATH


def ledger_key(provider: str, model: str) -> str:
    return f"{provider}/{model}"


@dataclass(frozen=True)
class LedgerEntry:
    provider: str
    model: str
    quota: QuotaState
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "updated_at": self.updated_at,
            "quota": self.quota.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "LedgerEntry | None":
        if not isinstance(raw, Mapping):
            return None
        quota_raw = raw.get("quota")
        if not isinstance(quota_raw, Mapping):
            return None
        provider = str(raw.get("provider") or quota_raw.get("provider") or "")
        model = str(raw.get("model") or quota_raw.get("model") or "")
        if not provider and not model:
            return None
        quota = QuotaState.from_dict(quota_raw)
        try:
            updated_at = float(raw.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            updated_at = 0.0
        return cls(provider=provider, model=model, quota=quota, updated_at=updated_at)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class QuotaLedger:
    """In-memory view plus a JSON file. ``load()`` before reading after restart."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_ledger_path()
        self._entries: dict[str, LedgerEntry] = {}

    def load(self) -> "QuotaLedger":
        """Rebuild from disk. A corrupt file loads as empty, never raises."""
        self._entries = {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self
        if not isinstance(payload, Mapping):
            return self
        rows = payload.get("entries")
        if isinstance(rows, list):
            for row in rows:
                entry = LedgerEntry.from_dict(row)
                if entry is not None:
                    self._entries[ledger_key(entry.provider, entry.model)] = entry
            return self
        # Also accept the keyed form so an operator-edited file still loads.
        if isinstance(rows, Mapping):
            for key, row in rows.items():
                entry = LedgerEntry.from_dict(row)
                if entry is not None:
                    self._entries[str(key)] = entry
        return self

    def states(self) -> dict[str, QuotaState]:
        return {key: entry.quota for key, entry in self._entries.items()}

    def entries(self) -> list[LedgerEntry]:
        return sorted(self._entries.values(), key=lambda entry: ledger_key(entry.provider, entry.model))

    def get(self, provider: str, model: str) -> QuotaState | None:
        entry = self._entries.get(ledger_key(provider, model))
        return entry.quota if entry is not None else None

    def record(self, state: QuotaState, *, now: float | None = None) -> QuotaState:
        clock = time.time() if now is None else now
        self._entries[ledger_key(state.provider, state.model)] = LedgerEntry(
            provider=state.provider,
            model=state.model,
            quota=state,
            updated_at=clock,
        )
        self.save()
        return state

    def record_many(self, states: Iterable[QuotaState], *, now: float | None = None) -> list[QuotaState]:
        clock = time.time() if now is None else now
        recorded = [self.record(state, now=clock) for state in states]
        return recorded

    def clear(self) -> None:
        self._entries = {}
        self.save()

    def save(self) -> Path:
        payload = {
            "schema": LEDGER_SCHEMA,
            "saved_at": time.time(),
            "entries": [entry.to_dict() for entry in self.entries()],
        }
        _atomic_write(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return self.path
