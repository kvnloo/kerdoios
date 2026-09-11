from __future__ import annotations

import importlib.util
import subprocess
import sys
import types as stdlib_types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PackagingTests(unittest.TestCase):
    def test_repo_root_does_not_shadow_stdlib_types(self) -> None:
        self.assertFalse(
            (ROOT / "types.py").is_file(),
            "repo-root types.py shadows stdlib types when cwd is the plugin",
        )
        self.assertTrue(hasattr(stdlib_types, "ModuleType"))
        self.assertTrue(hasattr(stdlib_types, "SimpleNamespace"))
        self.assertFalse(hasattr(stdlib_types, "WorkRequirement"))

    def test_plugin_cwd_import_types_is_stdlib(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import types; assert hasattr(types, 'ModuleType'); "
                "assert not hasattr(types, 'WorkRequirement')",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_kerdoios_imports_from_any_checkout_name(self) -> None:
        import kerdoios
        from kerdoios.types import WorkRequirement

        self.assertTrue(callable(kerdoios.register))
        self.assertTrue(callable(WorkRequirement))

    def test_hermes_plugin_manifest_stays_at_root(self) -> None:
        self.assertTrue((ROOT / "plugin.yaml").is_file())
        shim = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("register", shim)
        self.assertIn("name: kerdoios", (ROOT / "plugin.yaml").read_text(encoding="utf-8"))

    def test_hermes_directory_load_exposes_register(self) -> None:
        ns_name = "_kerdoios_hermes_ns_test"
        module_name = f"{ns_name}.plugin"
        ns = stdlib_types.ModuleType(ns_name)
        ns.__path__ = []  # type: ignore[attr-defined]  # ModuleType has no __path__ until we fake a package
        sys.modules[ns_name] = ns

        def _cleanup() -> None:
            for key in [n for n in sys.modules if n == ns_name or n.startswith(ns_name + ".")]:
                sys.modules.pop(key, None)

        self.addCleanup(_cleanup)
        spec = importlib.util.spec_from_file_location(
            module_name,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(ROOT)]  # type: ignore[attr-defined]  # ModuleType has no __path__ until we fake a package
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.register))

    def test_module_cli_runs_from_checkout(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "kerdoios", "plan", "--workers", "8", "--mode", "cheap"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertIn("placements", proc.stdout)
