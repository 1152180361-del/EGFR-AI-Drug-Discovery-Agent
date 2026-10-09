import unittest

from egfr_agent.decision_engine import EGFRDecisionEngine
from examples.mini_demo import (
    build_demo_evidence,
    generate_demo_report,
)


class TestMiniDemo(unittest.TestCase):

    def test_expected_ranking(self):
        evidence = build_demo_evidence()
        ranking = EGFRDecisionEngine().analyze(evidence)

        actual = list(
            zip(
                ranking["candidate"],
                ranking["score"].astype(int),
            )
        )

        expected = [
            ("DEMO_SYNTH_A", 11),
            ("DEMO_SYNTH_B", 10),
            ("DEMO_SYNTH_C", 3),
        ]

        self.assertEqual(actual, expected)

    def test_every_candidate_has_reason(self):
        ranking = EGFRDecisionEngine().analyze(
            build_demo_evidence()
        )

        self.assertTrue(
            ranking["reason"].str.strip().ne("").all()
        )

        reason = ranking.loc[
            ranking["candidate"] == "DEMO_SYNTH_C",
            "reason",
        ].iloc[0]

        self.assertIn("Consensus rank #3", reason)

    def test_scientific_boundaries(self):
        report = generate_demo_report()

        self.assertIn(
            "SYNTHETIC DEMONSTRATION ONLY",
            report,
        )
        self.assertIn(
            "No QSAR model inference performed.",
            report,
        )
        self.assertIn(
            "No molecular docking performed.",
            report,
        )
        self.assertIn(
            "No experimental EGFR inhibition validation.",
            report,
        )
        self.assertIn(
            "H-bond unverified",
            report,
        )


if __name__ == "__main__":
    unittest.main()
