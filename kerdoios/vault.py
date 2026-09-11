"""Hermes Bitwarden is the credential plane. Env files are not.

Kerdoios does not grow a second vault. Hermes already pulls Bitwarden Secrets
Manager at process start (``hermes secrets bitwarden`` / ``bws``) and injects
matching secret names into the process environment. This module:

- fails closed unless that vault is configured
- treats process env as Hermes-injected materialization *only after* vault
  setup is proven
- never treats a missing vault as a free overlay
- never returns secret values from doctor/status helpers
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Protocol

REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class ProviderKey:
    provider: str
    secret_name: str


# Declared names Hermes/Bitwarden should hold. Values may be unset; doctor
# still requires the vault itself. NVIDIA/xAI/Nous have no adapter yet —
# they are catalog entries so enable cannot claim a half-keyed host is ready.
PROVIDER_KEY_CATALOG: tuple[ProviderKey, ...] = (
    ProviderKey("groq", "GROQ_API_KEY"),
    ProviderKey("cerebras", "CEREBRAS_API_KEY"),
    ProviderKey("openrouter", "OPENROUTER_API_KEY"),
    ProviderKey("nvidia", "NVIDIA_API_KEY"),
    ProviderKey("xai", "XAI_API_KEY"),
    ProviderKey("nous", "NOUS_API_KEY"),
)

_CATALOG_BY_PROVIDER = {item.provider: item for item in PROVIDER_KEY_CATALOG}


class VaultResolver(Protocol):
    def configured(self) -> bool:
        """True when Hermes Bitwarden (or a test double) is actually set up."""

    def resolve(self, secret_name: str) -> str | None:
        """Return the secret or None. Callers must not log the return value."""


class UnconfiguredResolver:
    """Default when no vault is present. Missing vault is not a free tier."""

    def configured(self) -> bool:
        return False

    def resolve(self, secret_name: str) -> str | None:
        return None

    def __repr__(self) -> str:
        return "UnconfiguredResolver()"


class FakeVaultResolver:
    """Test double. Values stay in-process; ``repr`` lists names only."""

    def __init__(self, secrets: Mapping[str, str] | None = None, *, configured: bool = True) -> None:
        self._secrets = {str(k): str(v) for k, v in (secrets or {}).items() if v}
        self._configured = configured

    def configured(self) -> bool:
        return self._configured

    def resolve(self, secret_name: str) -> str | None:
        if not self._configured:
            return None
        value = self._secrets.get(secret_name)
        return value if value else None

    def __repr__(self) -> str:
        return f"FakeVaultResolver(configured={self._configured}, names={sorted(self._secrets)!r})"


class HermesBitwardenResolver:
    """Read Hermes ``secrets.bitwarden`` from the operator home. Do not shell to ``bws``.

    Hermes owns fetch/injection. Kerdoios only checks that the vault is on and,
    when it is, reads the names Hermes already materialised. Raw ``GROQ_API_KEY``
    in the process env without vault config is ignored.
    """

    def __init__(self, *, home: Path | None = None) -> None:
        self._home = Path(home) if home is not None else operator_hermes_home()

    def configured(self) -> bool:
        section = bitwarden_config(self._home)
        if not _as_bool(section.get("enabled")):
            return False
        project_id = (section.get("project_id") or "").strip().strip("\"'")
        if not project_id:
            return False
        token_env = (section.get("access_token_env") or "BWS_ACCESS_TOKEN").strip() or "BWS_ACCESS_TOKEN"
        return bool(_bootstrap_token_present(self._home, token_env))

    def resolve(self, secret_name: str) -> str | None:
        if not self.configured():
            return None
        # Hermes injects vault secrets into os.environ. Ignore leftover .env
        # provider keys when the vault is off (configured() already returned).
        value = os.environ.get(secret_name)
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    def __repr__(self) -> str:
        return f"HermesBitwardenResolver(home={str(self._home)!r}, configured={self.configured()})"


_override: VaultResolver | None = None


def operator_hermes_home() -> Path:
    """Operator Hermes home, not Plugin Doctor's staging sandbox."""
    override = os.environ.get("KERDOIOS_HERMES_HOME")
    if override:
        return Path(override).expanduser()
    env_home = os.environ.get("HERMES_HOME")
    if env_home and "hermes-plugin-doctor-" not in env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".hermes"


def active_resolver() -> VaultResolver:
    if _override is not None:
        return _override
    return HermesBitwardenResolver()


def set_resolver(resolver: VaultResolver | None) -> VaultResolver | None:
    """Install a resolver. Tests use ``override_resolver``."""
    global _override
    previous = _override
    _override = resolver
    return previous


@contextmanager
def override_resolver(resolver: VaultResolver) -> Iterator[VaultResolver]:
    previous = set_resolver(resolver)
    try:
        yield resolver
    finally:
        set_resolver(previous)


def provider_secret(provider: str, *, explicit: str | None = None) -> str | None:
    """Keyed-adapter lookup. ``explicit`` is test/injection only, never env."""
    if explicit is not None:
        stripped = explicit.strip()
        return stripped if stripped else None
    spec = _CATALOG_BY_PROVIDER.get(provider)
    if spec is None:
        return None
    return active_resolver().resolve(spec.secret_name)


def catalog_presence() -> tuple[tuple[str, str], ...]:
    """Name + set/unset only. Never includes secret values."""
    resolver = active_resolver()
    rows: list[tuple[str, str]] = []
    for spec in PROVIDER_KEY_CATALOG:
        present = bool(resolver.resolve(spec.secret_name)) if resolver.configured() else False
        rows.append((spec.secret_name, "set" if present else "unset"))
    return tuple(rows)


def redact(text: str, secrets: Mapping[str, str] | None = None) -> str:
    """Replace known secret values with ``[REDACTED]``. Empty secrets are a no-op."""
    out = text
    values: list[str] = []
    if secrets is not None:
        values.extend(str(v) for v in secrets.values() if v)
    resolver = _override
    if isinstance(resolver, FakeVaultResolver):
        values.extend(resolver._secrets.values())
    # Longest first so a key that is a prefix of another still redacts fully.
    for value in sorted(set(values), key=len, reverse=True):
        if value:
            out = out.replace(value, REDACTED)
    return out


def bitwarden_config(home: Path) -> dict[str, str]:
    path = home / "config.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    return _parse_bitwarden_section(text)


def _bootstrap_token_present(home: Path, token_env: str) -> bool:
    env_value = os.environ.get(token_env)
    if env_value and env_value.strip():
        return True
    return _env_file_has_key(home / ".env", token_env)


def _env_file_has_key(path: Path, name: str) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    prefix = f"{name}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith(prefix):
            continue
        value = stripped.split("=", 1)[1].strip().strip("\"'")
        return bool(value)
    return False


def _as_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().strip("\"'").lower() in {"1", "true", "yes", "on"}


def _parse_bitwarden_section(text: str) -> dict[str, str]:
    """Pull ``secrets.bitwarden`` scalars. Not a general YAML loader."""
    in_secrets = False
    in_bitwarden = False
    secrets_indent: int | None = None
    bitwarden_indent: int | None = None
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if stripped == "secrets:" or stripped.startswith("secrets:"):
            in_secrets = True
            secrets_indent = indent
            in_bitwarden = False
            continue
        if in_secrets and secrets_indent is not None and indent <= secrets_indent and not stripped.startswith("bitwarden"):
            in_secrets = False
            in_bitwarden = False
        if in_secrets and (stripped == "bitwarden:" or stripped.startswith("bitwarden:")):
            in_bitwarden = True
            bitwarden_indent = indent
            continue
        if in_bitwarden and bitwarden_indent is not None and indent <= bitwarden_indent:
            in_bitwarden = False
        if not in_bitwarden or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        out[key.strip()] = value.strip()
    return out
