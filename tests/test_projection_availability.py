"""Availability and backend classification are observed, never assumed.

Two real defects lived in `kerdoios/projection.py`, both instances of the same
rule the rest of the stack already enforces for residency: *do not assert a
state you have not observed*.

1. `k8s_rows` stamped every node `Status.RUNNABLE` unconditionally, so a
   `NotReady` or cordoned node advertised itself as available capacity.
2. `_backend_of` matched only on `offer.provider`, so rows whose transport was
   recorded in `offer.source` (e.g. ``openrouter:/api/v1/models``) fell through
   to a catch-all that returned ``pi-ai``. An unrecognised remote transport was
   reported as a specific, identifiable runtime.

These tests pin the observed-only behaviour so neither regresses.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from kerdoios.projection import ExecutionBackend, Status, _backend_of, k8s_rows


class _Offer:
    """Minimal stand-in: `_backend_of` reads only provider/source/local."""

    def __init__(self, provider: str, source: str = "", local: bool = False) -> None:
        self.provider = provider
        self.source = source
        self.local = local


def _node(name: str, *, ready: bool | None, unschedulable: bool = False, gpu: bool = False) -> dict:
    conditions = [] if ready is None else [{"type": "Ready", "status": "True" if ready else "False",
                                            "reason": "KubeletReady" if ready else "KubeletNotReady"}]
    alloc = {"cpu": "4", "memory": "8Gi", "pods": "110"}
    if gpu:
        alloc["nvidia.com/gpu"] = "1"
    return {
        "metadata": {"name": name},
        "spec": {"unschedulable": unschedulable},
        "status": {
            "allocatable": alloc,
            "conditions": conditions,
            "nodeInfo": {"kubeletVersion": "v1.37.0"},
        },
    }


class K8sNodeAvailabilityTests(unittest.TestCase):
    def _rows(self, nodes: list[dict]):
        # The cluster is discovered by providers.kubernetes; k8s_rows only
        # shapes those offers. Mocking the provider seam (not subprocess) is
        # what proves there is a single kubectl path.
        from kerdoios.providers import kubernetes as k8s_provider

        with mock.patch.object(k8s_provider, "_kubectl_json", return_value={"items": nodes}):
            return k8s_rows()

    def test_ready_node_is_runnable(self) -> None:
        rows = self._rows([_node("n1", ready=True)])
        self.assertEqual(rows[0].status, Status.RUNNABLE.value)
        self.assertTrue(rows[0].constraints["ready"])

    def test_not_ready_node_is_not_advertised_as_capacity(self) -> None:
        rows = self._rows([_node("n2", ready=False)])
        self.assertEqual(
            rows[0].status,
            Status.UNAVAILABLE.value,
            "a NotReady node must not be reported RUNNABLE",
        )
        self.assertIn("KubeletNotReady", rows[0].notes)

    def test_node_without_ready_condition_is_unavailable(self) -> None:
        rows = self._rows([_node("n3", ready=None)])
        self.assertEqual(rows[0].status, Status.UNAVAILABLE.value)

    def test_cordoned_node_is_unavailable_even_when_ready(self) -> None:
        rows = self._rows([_node("n4", ready=True, unschedulable=True)])
        self.assertEqual(rows[0].status, Status.UNAVAILABLE.value)
        self.assertIn("cordoned", rows[0].notes)

    def test_gpu_is_reported_only_when_allocatable(self) -> None:
        self.assertTrue(self._rows([_node("n5", ready=True, gpu=True)])[0].capabilities["gpu"])
        self.assertFalse(self._rows([_node("n6", ready=True)])[0].capabilities["gpu"])


class BackendClassificationTests(unittest.TestCase):
    def test_source_identified_openrouter_row_is_not_pi_ai(self) -> None:
        """The 441-row mislabel: provider says a vendor, source says openrouter."""
        offer = _Offer(provider="some-vendor", source="openrouter:/api/v1/models")
        self.assertEqual(_backend_of(offer), ExecutionBackend.OPENROUTER.value)

    def test_source_identified_groq_and_cerebras_rows(self) -> None:
        self.assertEqual(_backend_of(_Offer("x", "groq:/openai/v1")), ExecutionBackend.GROQ.value)
        self.assertEqual(_backend_of(_Offer("x", "cerebras:/v1")), ExecutionBackend.CEREBRAS.value)

    def test_provider_named_rows_still_resolve(self) -> None:
        self.assertEqual(_backend_of(_Offer("openrouter")), ExecutionBackend.OPENROUTER.value)
        self.assertEqual(_backend_of(_Offer("groq")), ExecutionBackend.GROQ.value)

    def test_unknown_remote_transport_is_unknown_not_pi_ai(self) -> None:
        offer = _Offer(provider="mystery", source="mystery://endpoint", local=False)
        self.assertEqual(
            _backend_of(offer),
            ExecutionBackend.UNKNOWN.value,
            "an unidentified remote transport must not be named a specific runtime",
        )

    def test_local_offer_defaults_to_llama_cpp(self) -> None:
        self.assertEqual(_backend_of(_Offer("local", local=True)), ExecutionBackend.LLAMA_CPP.value)


if __name__ == "__main__":
    unittest.main()
