from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerdoios import register
from kerdoios.doctor import VaultNotConfigured, inspect, require_vault
from kerdoios.inventory import discover_all
from kerdoios.providers.http import JsonResponse
from kerdoios.providers.openai_compat import discover_groq
from kerdoios.vault import (
    PROVIDER_KEY_CATALOG,
    REDACTED,
    FakeVaultResolver,
    HermesBitwardenResolver,
    UnconfiguredResolver,
    catalog_presence,
    override_resolver,
    provider_secret,
    redact,
)

_SECRET = "gsk_live_test_only_never_commit_abc123"
_GROQ_BODY = {
    "data": [
        {"id": "llama-3.3-70b-versatile", "context_window": 131072},
        {"id": "whisper-large-v3", "context_window": 448},
    ]
}


def _write_hermes(home: Path, *, enabled: bool, project_id: str = "proj-1", token: str | None = "bws-token") -> None:
    home.mkdir(parents=True, exist_ok=True)
    enabled_txt = "true" if enabled else "false"
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        f"    enabled: {enabled_txt}\n"
        f"    project_id: {project_id}\n"
        "    server_url: https://vault.bitwarden.com\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n"
    )
    if token:
        (home / ".env").write_text(f"BWS_ACCESS_TOKEN={token}\n")


class CatalogTests(unittest.TestCase):
    def test_catalog_declares_keyed_providers(self) -> None:
        names = {item.provider: item.secret_name for item in PROVIDER_KEY_CATALOG}
        self.assertEqual(names["groq"], "GROQ_API_KEY")
        self.assertEqual(names["cerebras"], "CEREBRAS_API_KEY")
        self.assertEqual(names["openrouter"], "OPENROUTER_API_KEY")
        self.assertIn("nvidia", names)
        self.assertIn("xai", names)
        self.assertIn("nous", names)


class DoctorWithoutVaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        home = Path(self.tmp.name) / "hermes"
        home.mkdir()
        self.env = patch.dict(os.environ, {"KERDOIOS_HERMES_HOME": str(home)}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        set_resolver = override_resolver(UnconfiguredResolver())
        set_resolver.__enter__()
        self.addCleanup(lambda: set_resolver.__exit__(None, None, None))

    def test_inspect_refuses(self) -> None:
        report = inspect()
        self.assertFalse(report.ok)
        self.assertFalse(report.vault_configured)
        self.assertIn("vault is not configured", report.message())
        blob = json.dumps(report.to_dict())
        self.assertNotIn(_SECRET, blob)
        for name, status in report.catalog:
            self.assertEqual(status, "unset")
            self.assertTrue(name.endswith("_API_KEY"))

    def test_require_vault_raises(self) -> None:
        with self.assertRaises(VaultNotConfigured) as caught:
            require_vault()
        self.assertNotIn(_SECRET, str(caught.exception))

    def test_register_fails_closed(self) -> None:
        with self.assertRaises(VaultNotConfigured):
            register(_Ctx())

    def test_cli_doctor_exits_one(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "kerdoios", "doctor"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "KERDOIOS_HERMES_HOME": str(Path(self.tmp.name) / "hermes")},
        )
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn(_SECRET, proc.stdout)
        self.assertNotIn(_SECRET, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertFalse(payload["ok"])


class DoctorWithFakeVaultTests(unittest.TestCase):
    def test_fake_vault_makes_doctor_ok_and_keyed_overlay_appear(self) -> None:
        fake = FakeVaultResolver({"GROQ_API_KEY": _SECRET, "CEREBRAS_API_KEY": "c-test"})
        resp = JsonResponse(body=_GROQ_BODY, headers={})
        with (
            override_resolver(fake),
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"KERDOIOS_CACHE": tmp, "GROQ_API_KEY": ""}, clear=False),
            patch("kerdoios.providers.openai_compat.get_json_response", return_value=resp),
            patch("kerdoios.inventory.openrouter.discover", return_value=[]),
            patch("kerdoios.inventory.discover_cerebras", return_value=[]),
            patch("kerdoios.inventory.local.discover", return_value=[]),
        ):
            report = inspect()
            self.assertTrue(report.ok)
            self.assertTrue(report.vault_configured)
            presence = dict(catalog_presence())
            self.assertEqual(presence["GROQ_API_KEY"], "set")
            self.assertEqual(presence["NVIDIA_API_KEY"], "unset")
            blob = json.dumps(report.to_dict())
            self.assertNotIn(_SECRET, blob)
            self.assertNotIn("c-test", blob)
            rows = discover_all(include_fixture=False, live=True, free_only=True)
            cache = Path(tmp) / "inventory.json"
            if cache.is_file():
                self.assertNotIn(_SECRET, cache.read_text())
        groq = [o for o in rows if o.provider == "groq"]
        self.assertTrue(any(o.model == "llama-3.3-70b-versatile" for o in groq))
        self.assertFalse(any(o.model == "whisper-large-v3" for o in groq))

    def test_register_succeeds_with_fake_vault(self) -> None:
        with override_resolver(FakeVaultResolver({"GROQ_API_KEY": _SECRET})):
            ctx = _Ctx()
            register(ctx)
            self.assertIn("kerdoios_plan", ctx.tools)


class EnvIsNotTheCredentialPlaneTests(unittest.TestCase):
    def test_env_key_without_vault_does_not_overlay(self) -> None:
        resp = JsonResponse(body=_GROQ_BODY, headers={})
        with (
            override_resolver(UnconfiguredResolver()),
            patch.dict(os.environ, {"GROQ_API_KEY": _SECRET}, clear=False),
            patch("kerdoios.providers.openai_compat.get_json_response", return_value=resp),
        ):
            self.assertEqual(discover_groq(), [])
            self.assertIsNone(provider_secret("groq"))

    def test_repr_and_redact_never_emit_secret(self) -> None:
        fake = FakeVaultResolver({"GROQ_API_KEY": _SECRET})
        self.assertNotIn(_SECRET, repr(fake))
        self.assertEqual(redact(f"Authorization: Bearer {_SECRET}", {"GROQ_API_KEY": _SECRET}), f"Authorization: Bearer {REDACTED}")


class HermesBitwardenResolverTests(unittest.TestCase):
    def test_enabled_project_and_token_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "hermes"
            _write_hermes(home, enabled=True)
            resolver = HermesBitwardenResolver(home=home)
            self.assertTrue(resolver.configured())

    def test_disabled_or_missing_project_is_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "hermes"
            _write_hermes(home, enabled=False)
            self.assertFalse(HermesBitwardenResolver(home=home).configured())
            _write_hermes(home, enabled=True, project_id="", token="bws-token")
            self.assertFalse(HermesBitwardenResolver(home=home).configured())

    def test_env_provider_key_ignored_until_vault_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "hermes"
            _write_hermes(home, enabled=False)
            with patch.dict(os.environ, {"GROQ_API_KEY": _SECRET}, clear=False):
                resolver = HermesBitwardenResolver(home=home)
                self.assertIsNone(resolver.resolve("GROQ_API_KEY"))
            _write_hermes(home, enabled=True)
            with patch.dict(os.environ, {"GROQ_API_KEY": _SECRET}, clear=False):
                resolver = HermesBitwardenResolver(home=home)
                self.assertEqual(resolver.resolve("GROQ_API_KEY"), _SECRET)

    def test_doctor_staging_hermes_home_is_not_the_operator_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "hermes-plugin-doctor-xyz"
            staging.mkdir()
            env = {k: v for k, v in os.environ.items() if k not in {"KERDOIOS_HERMES_HOME", "HERMES_HOME"}}
            env["HERMES_HOME"] = str(staging)
            with patch.dict(os.environ, env, clear=True):
                from kerdoios.vault import operator_hermes_home

                home = operator_hermes_home()
                self.assertNotEqual(home, staging)
                self.assertNotIn("hermes-plugin-doctor-", str(home))


class _Ctx:
    def __init__(self) -> None:
        self.tools: dict = {}

    def register_tool(self, *, name: str, toolset: str, schema: dict, handler) -> None:
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}


if __name__ == "__main__":
    unittest.main()
