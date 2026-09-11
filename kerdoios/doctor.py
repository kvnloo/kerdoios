"""Plugin doctor: vault is a prerequisite, not an optional later nicety.

``hermes plugins doctor`` / enable load ``register()``, which refuses unless
the operator Hermes Bitwarden vault is configured. Public OpenRouter ``:free``
listing can still run from the CLI without a key; that does not make the
plugin healthy on a half-configured host.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vault import active_resolver, catalog_presence


class VaultNotConfigured(RuntimeError):
    """Raised when enable/doctor should fail closed. Message never holds secrets."""


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    vault_configured: bool
    catalog: tuple[tuple[str, str], ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "vault_configured": self.vault_configured,
            "catalog": [{"name": name, "status": status} for name, status in self.catalog],
            "errors": list(self.errors),
            "hint": "hermes secrets bitwarden setup" if not self.ok else None,
        }

    def message(self) -> str:
        if self.ok:
            return "vault configured; provider-key catalog is resolvable"
        if self.errors:
            return self.errors[0]
        return "vault is not configured; run hermes secrets bitwarden setup"


def inspect() -> DoctorReport:
    resolver = active_resolver()
    configured = resolver.configured()
    catalog = catalog_presence()
    errors: list[str] = []
    if not configured:
        errors.append(
            "vault is not configured; Hermes Bitwarden is required "
            "(hermes secrets bitwarden setup). Provider .env keys are not the credential plane."
        )
    return DoctorReport(
        ok=not errors,
        vault_configured=configured,
        catalog=catalog,
        errors=tuple(errors),
    )


def require_vault() -> DoctorReport:
    report = inspect()
    if not report.ok:
        raise VaultNotConfigured(report.message())
    return report
