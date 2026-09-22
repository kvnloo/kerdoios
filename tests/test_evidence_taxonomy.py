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

    def test_trust_is_granted_only_by_classes_the_taxonomy_allows(self) -> None:
        """The tier is derived from declared effects, not asserted per class.

        This test has now been wrong twice in the same direction, which is why it
        is written as an explicit table. It first asserted `tested` for every
        class including SMOKE, locking in an over-claim. It then asserted a
        verdict for every class, which locked in PROMOTION granting
        TRUSTED_BOUNDED -- a class whose declared influence is `default_routing`
        and `install_profile_membership`, and which the taxonomy says "consumes
        CONFIRM and OOD evidence; it does not substitute for it".

        A class grants a verdict only where it may influence a trust record:

          SMOKE            may_influence: []                  -> no verdict, not tested
          EXPLORATORY_BETA tested_observation                 -> TESTED_EXPERIMENTAL
          SHADOW           shadow_trust_record                -> TRUSTED_SHADOW
          PAIRED_REPLAY    tested_observation                 -> TESTED_EXPERIMENTAL
          CONFIRM          capability_trust_record            -> TRUSTED_BOUNDED
          OOD              general_trust_record               -> TRUSTED_GENERAL
          PROMOTION        default_routing only               -> no verdict, not tested
        """
        expected = {
            EvidenceClass.SMOKE.value: (Status.DISCOVERED.value, Trust.UNTESTED.value),
            EvidenceClass.EXPLORATORY_BETA.value: (Status.TESTED.value, Trust.TESTED_EXPERIMENTAL.value),
            EvidenceClass.SHADOW.value: (Status.TESTED.value, Trust.TRUSTED_SHADOW.value),
            EvidenceClass.PAIRED_REPLAY.value: (Status.TESTED.value, Trust.TESTED_EXPERIMENTAL.value),
            EvidenceClass.CONFIRM.value: (Status.TESTED.value, Trust.TRUSTED_BOUNDED.value),
            EvidenceClass.OOD.value: (Status.TESTED.value, Trust.TRUSTED_GENERAL.value),
            EvidenceClass.PROMOTION.value: (Status.DISCOVERED.value, None),
        }
        self.assertEqual(set(expected), {c.value for c in EvidenceClass},
                         "every class must be listed, so a new one cannot slip through")
        for cls, (want_status, want_trust) in expected.items():
            entries: list = []
            apply_evidence(entries, [_doc(evidence_class=cls)])
            self.assertEqual(entries[0].status, want_status, f"{cls}: status")
            self.assertEqual(entries[0].trust.get("bounded_choice"), want_trust, f"{cls}: trust")

    def test_ood_establishes_general_trust_not_bounded(self) -> None:
        """OOD is the generality class, so it is the general tier it establishes.

        `TRUSTED_GENERAL` was declared in this repo and unreachable: the ladder
        granted `TRUSTED_BOUNDED` to OOD, so nothing could ever produce the
        general tier.
        """
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class="OOD")])
        self.assertEqual(entries[0].trust["bounded_choice"], Trust.TRUSTED_GENERAL.value)
        self.assertNotEqual(entries[0].trust["bounded_choice"], Trust.TRUSTED_BOUNDED.value)

    def test_promotion_alone_grants_no_trust(self) -> None:
        """A promotion consumes CONFIRM and OOD evidence; it does not substitute."""
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class="PROMOTION")])
        self.assertNotIn("bounded_choice", entries[0].trust)
        self.assertEqual(entries[0].status, Status.DISCOVERED.value)
        # ...but it is still recorded as evidence, and does not raise.
        self.assertIn("bounded_choice", entries[0].evidence)

    def test_smoke_may_restrict_but_may_not_claim(self) -> None:
        """A restriction is allowed on the weakest evidence; a grant is not.

        Quarantining because a smoke test produced an unsafe result is the safe
        direction and must survive. What must not survive is SMOKE raising a
        capability to anything that reads as evaluated-and-fine.
        """
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class="SMOKE", unsafe=3)])
        self.assertEqual(entries[0].trust["bounded_choice"], Trust.QUARANTINED.value)

    def test_smoke_cannot_reach_a_production_trust_status(self) -> None:
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class="SMOKE")])
        self.assertNotIn(
            entries[0].trust["bounded_choice"],
            {Trust.TRUSTED_BOUNDED.value, Trust.TRUSTED_SHADOW.value, Trust.TRUSTED_GENERAL.value},
        )

    def test_lowercase_serialized_class_is_accepted(self) -> None:
        """The defect this normalization exists for.

        Every artifact in this ecosystem serializes the class lowercased --
        `"evidence_class": "exploratory_beta"` -- and the consumer matched uppercase
        literals exactly, so it rejected all of them::

            ValueError: unknown evidence_class 'exploratory_beta'

        The class set is unchanged; only the case is normalized. An artifact that
        genuinely names an unknown class still raises.
        """
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class="exploratory_beta")])
        self.assertEqual(entries[0].status, Status.TESTED.value)
        self.assertEqual(entries[0].trust["bounded_choice"], Trust.TESTED_EXPERIMENTAL.value)

    def test_lowercase_confirm_also_promotes(self) -> None:
        """Normalization must not be a one-way downgrade.

        Accepting the lowercase spelling only for weak classes would be worse
        than rejecting it: a lowercase CONFIRM artifact would look ingested and
        silently never promote.
        """
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class="confirm")])
        self.assertEqual(entries[0].trust["bounded_choice"], Trust.TRUSTED_BOUNDED.value)

    def test_every_class_is_accepted_in_both_spellings(self) -> None:
        """Accepted, not necessarily granted: PROMOTION is accepted and grants nothing."""
        for cls in EvidenceClass:
            for spelling in (cls.value, cls.value.lower()):
                entries: list = []
                apply_evidence(entries, [_doc(evidence_class=spelling)])
                self.assertIn(
                    "bounded_choice", entries[0].evidence,
                    f"{spelling!r} was not recorded as evidence",
                )

    def test_whitespace_is_not_a_third_spelling(self) -> None:
        entries: list = []
        apply_evidence(entries, [_doc(evidence_class="  SHADOW  ")])
        self.assertEqual(entries[0].trust["bounded_choice"], Trust.TRUSTED_SHADOW.value)

    def test_corpus_is_not_an_evidence_class(self) -> None:
        """A corpus is a dataset, not a claim about a capability.

        evolution-lab stamped a corpus manifest with
        `"evidence_class": "corpus"`. It must not be waved through as a weak
        class, because the ladder would then rank a dataset against CONFIRM.
        """
        with self.assertRaises(ValueError) as ctx:
            apply_evidence([], [_doc(evidence_class="corpus")])
        self.assertIn("corpus", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
