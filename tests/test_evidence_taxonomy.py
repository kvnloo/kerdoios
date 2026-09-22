"""Evidence class governs trust; a class the code does not know must be loud.

Two defects lived here:

1. `EvidenceClass` was missing `PAIRED_REPLAY`, which the canonical taxonomy in
   `kvnloo/z0 registry/maturity.yaml` requires. A paired-replay claim could not
   be expressed at all, even though it is the class that makes a comparative
   claim attributable.
2. The trust ladder had no final `else`. An unrecognised class skipped every
   branch, so the row was still stamped `Status.TESTED` while receiving no trust
   entry -- it looked evaluated and carried no verdict.

The taxonomy also constrains what each class may influence. `PAIRED_REPLAY`
establishes a comparison, not a capability, so it must never confer
`TRUSTED_BOUNDED` however good its numbers look.
"""

from __future__ import annotations

import unittest

from kerdoios.projection import (
    EvidenceClass,
    Status,
    Trust,
    apply_evidence,
)


def _doc(**over) -> dict:
    doc = {
        "model": "some-model",
        "provider": "groq",
        "role": "bounded_choice",
        "n": 84,
        "success": 0.95,
        "unsafe": 0,
        "evidence_class": EvidenceClass.EXPLORATORY_BETA.value,
    }
    doc.update(over)
    return doc


class EvidenceTaxonomyTests(unittest.TestCase):
    def test_paired_replay_is_expressed(self) -> None:
        """The class the registry requires must exist locally."""
        self.assertEqual(EvidenceClass.PAIRED_REPLAY.value, "PAIRED_REPLAY")

    def test_local_classes_match_the_canonical_taxonomy(self) -> None:
        """Mirror of z0 REQUIRED_EVIDENCE_CLASSES, in the same strength order."""
        canonical = (
            "SMOKE", "EXPLORATORY_BETA", "SHADOW", "PAIRED_REPLAY",
            "CONFIRM", "OOD", "PROMOTION",
        )
        self.assertEqual(tuple(c.value for c in EvidenceClass), canonical)

    def test_paired_replay_never_confers_bounded_trust(self) -> None:
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class=EvidenceClass.PAIRED_REPLAY.value)])
        trust = entries[0].trust["bounded_choice"]
        self.assertNotEqual(
            trust, Trust.TRUSTED_BOUNDED.value,
            "paired replay establishes a comparison, not a capability",
        )
        self.assertEqual(trust, Trust.TESTED_EXPERIMENTAL.value)

    def test_confirm_does_confer_bounded_trust(self) -> None:
        """Control: the ladder still promotes the classes allowed to promote."""
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class=EvidenceClass.CONFIRM.value)])
        self.assertEqual(entries[0].trust["bounded_choice"], Trust.TRUSTED_BOUNDED.value)

    def test_paired_replay_that_is_unsafe_is_quarantined(self) -> None:
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class=EvidenceClass.PAIRED_REPLAY.value, unsafe=2)])
        self.assertEqual(entries[0].trust["bounded_choice"], Trust.QUARANTINED.value)

    def test_unknown_evidence_class_raises_instead_of_falling_through(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            apply_evidence([], [_doc(evidence_class="TOTALLY_MADE_UP")])
        self.assertIn("TOTALLY_MADE_UP", str(ctx.exception))
        self.assertIn("maturity.yaml", str(ctx.exception))

    def test_every_row_that_is_marked_tested_carries_a_trust_verdict(self) -> None:
        """The silent-fallthrough property: tested rows must have a verdict."""
        for cls in EvidenceClass:
            entries: list = []
            apply_evidence(entries, [_doc(evidence_class=cls.value)])
            self.assertEqual(entries[0].status, Status.TESTED.value)
            self.assertIn(
                "bounded_choice", entries[0].trust,
                f"{cls.value} produced a TESTED row with no trust verdict",
            )


if __name__ == "__main__":
    unittest.main()
