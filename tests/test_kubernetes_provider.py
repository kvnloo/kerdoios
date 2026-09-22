"""Kubernetes is a provider, not a side-channel.

Kubernetes used to be reached by an inline `kubectl` call in
`projection.k8s_rows()`, behind a `--k8s` flag and absent from `discover_all`.
This module pins the properties that make it an ordinary provider:

* it emits `ResourceOffer` on the same `discover()` contract as every other
  provider, so placement needs no `if kubernetes`;
* availability is OBSERVED from the node's Ready condition and
  `spec.unschedulable`, never inferred from the node merely being listed;
* it is strictly optional -- discovery returns nothing unless asked;
* a node is not a model: `model` stays None and `resource_type` is `gpu` only
  when a GPU is genuinely allocatable.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from kerdoios.providers import kubernetes as k8s


def _node(name: str, *, ready: bool | None, unschedulable: bool = False,
          gpu: str | None = None, cpu: str = "4", memory: str = "8Gi") -> dict:
    conditions = [] if ready is None else [
        {"type": "Ready", "status": "True" if ready else "False",
         "reason": "KubeletReady" if ready else "KubeletNotReady"}
    ]
    alloc = {"cpu": cpu, "memory": memory, "pods": "110"}
    if gpu is not None:
        alloc[k8s.GPU_RESOURCE] = gpu
    return {
        "metadata": {"name": name},
        "spec": {"unschedulable": unschedulable},
        "status": {"allocatable": alloc, "conditions": conditions,
                   "nodeInfo": {"kubeletVersion": "v1.37.0"}},
    }


class QuantityTests(unittest.TestCase):
    def test_parses_kubernetes_quantities(self) -> None:
        self.assertEqual(k8s._parse_quantity("4"), 4.0)
        self.assertEqual(k8s._parse_quantity("1500m"), 1.5)
        self.assertEqual(k8s._parse_quantity("8Gi"), 8 * 1024**3)
        self.assertEqual(k8s._parse_quantity("1Ti"), 1024**4)
        self.assertEqual(k8s._parse_quantity(3), 3.0)

    def test_unknown_quantity_is_none_not_zero(self) -> None:
        """An unparseable capacity must not silently become zero."""
        self.assertIsNone(k8s._parse_quantity("nonsense"))
        self.assertIsNone(k8s._parse_quantity(None))
        self.assertIsNone(k8s._parse_quantity(""))


class ReadinessTests(unittest.TestCase):
    def test_ready_node_is_usable(self) -> None:
        offers = k8s.node_offers(nodes=[_node("n1", ready=True)])
        self.assertEqual(len(offers), 1)
        self.assertGreater(offers[0].telemetry.availability, 0.0)
        self.assertIn("node Ready", offers[0].source)

    def test_not_ready_node_is_offered_at_zero_availability(self) -> None:
        offers = k8s.node_offers(nodes=[_node("n2", ready=False)])
        self.assertEqual(offers[0].telemetry.availability, 0.0)
        self.assertEqual(offers[0].telemetry.failure_rate, 1.0)
        self.assertIn("KubeletNotReady", offers[0].source)

    def test_missing_ready_condition_is_not_usable(self) -> None:
        offers = k8s.node_offers(nodes=[_node("n3", ready=None)])
        self.assertEqual(offers[0].telemetry.availability, 0.0)

    def test_cordoned_node_is_not_usable_even_when_ready(self) -> None:
        offers = k8s.node_offers(nodes=[_node("n4", ready=True, unschedulable=True)])
        self.assertEqual(offers[0].telemetry.availability, 0.0)
        self.assertIn("cordoned", offers[0].source)

    def test_not_ready_node_is_still_offered_not_hidden(self) -> None:
        """Placement should be able to SEE an unusable node and decline it."""
        offers = k8s.node_offers(nodes=[_node("n5", ready=False)])
        self.assertEqual(len(offers), 1)


class ShapeTests(unittest.TestCase):
    def test_a_node_is_not_a_model(self) -> None:
        offer = k8s.node_offers(nodes=[_node("n6", ready=True)])[0]
        self.assertIsNone(offer.model, "a node name must not be offered as a model id")
        self.assertEqual(offer.provider, "kubernetes")
        self.assertEqual(offer.id, "k8s/node/n6")

    def test_resource_type_is_gpu_only_when_allocatable(self) -> None:
        self.assertEqual(
            k8s.node_offers(nodes=[_node("n7", ready=True, gpu="1")])[0].resource_type, "gpu"
        )
        self.assertEqual(
            k8s.node_offers(nodes=[_node("n8", ready=True)])[0].resource_type, "cpu"
        )

    def test_capacity_comes_from_allocatable(self) -> None:
        offer = k8s.node_offers(nodes=[_node("n9", ready=True, cpu="1500m", memory="16Gi")])[0]
        self.assertEqual(offer.capacity.cpu_cores, 1.5)
        self.assertEqual(offer.capacity.ram_gb, 16.0)
        self.assertEqual(offer.capacity.concurrency, 110)

    def test_vram_is_reported_for_a_gpu_node(self) -> None:
        offer = k8s.node_offers(nodes=[_node("n10", ready=True, gpu="2")])[0]
        self.assertEqual(offer.capacity.vram_gb, 2.0)


class OptionalityTests(unittest.TestCase):
    def test_discovery_is_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(k8s.enabled())
            self.assertEqual(k8s.discover(), [])

    def test_discovery_is_on_when_asked(self) -> None:
        with mock.patch.dict(os.environ, {"KERDOIOS_K8S": "1"}, clear=True):
            self.assertTrue(k8s.enabled())
            with mock.patch.object(k8s, "_kubectl_json", return_value={"items": [_node("n11", ready=True)]}):
                self.assertEqual(len(k8s.discover()), 1)

    def test_explicit_force_bypasses_the_env_gate(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(k8s, "_kubectl_json", return_value={"items": [_node("n12", ready=True)]}):
                self.assertEqual(len(k8s.discover(force=True)), 1)

    def test_provider_class_satisfies_the_abc(self) -> None:
        from kerdoios.providers.base import ResourceProvider

        provider = k8s.KubernetesProvider(force=True)
        self.assertIsInstance(provider, ResourceProvider)
        self.assertEqual(provider.name, "kubernetes")
        with mock.patch.object(k8s, "_kubectl_json", return_value={"items": [_node("n13", ready=True)]}):
            self.assertEqual(len(provider.discover()), 1)

    def test_a_cluster_that_is_unreachable_yields_no_offers(self) -> None:
        with mock.patch.object(k8s, "_kubectl_json", return_value=None):
            self.assertEqual(k8s.discover(force=True), [])

    def test_discover_all_includes_kubernetes_only_when_enabled(self) -> None:
        """It must be reachable through the same path as every other provider."""
        from kerdoios import inventory

        nodes = {"items": [_node("n14", ready=True)]}
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(k8s, "_kubectl_json", return_value=nodes):
                self.assertEqual([o for o in inventory.collect_live() if o.provider == "kubernetes"], [])
        with mock.patch.dict(os.environ, {"KERDOIOS_K8S": "1"}, clear=True):
            with mock.patch.object(k8s, "_kubectl_json", return_value=nodes):
                found = [o for o in inventory.collect_live() if o.provider == "kubernetes"]
                self.assertEqual(len(found), 1)


if __name__ == "__main__":
    unittest.main()
