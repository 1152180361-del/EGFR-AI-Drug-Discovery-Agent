import unittest

import pandas as pd

from egfr_agent.decision_engine import (
    EGFRDecisionEngine
)


class TestDecisionEvidenceSafety(unittest.TestCase):

    def setUp(self):
        self.engine = EGFRDecisionEngine()

        self.candidates = pd.DataFrame([
            {
                "phase4C_candidate_id":
                    "EGFR_CAND_003",
                "candidate_track": "SUPPORTED_A",
                "consensus_rank": 1,
                "qsar_predicted_pIC50": 6.856,
                "key_ATP_contact_coverage": 0.714,
                "MET793_distance": 5.608,
            },
            {
                "phase4C_candidate_id":
                    "EGFR_CAND_020",
                "candidate_track": "EXPLORATORY_OOD",
                "consensus_rank": 3,
                "qsar_predicted_pIC50": 7.108,
                "key_ATP_contact_coverage": 0.762,
                "MET793_distance": 3.397,
            }
        ])

    def test_existing_scores_preserved(self):
        result = self.engine.analyze(
            self.candidates
        ).set_index("candidate")

        self.assertEqual(
            int(result.loc["EGFR_CAND_003", "score"]),
            11
        )
        self.assertEqual(
            int(result.loc["EGFR_CAND_020", "score"]),
            9
        )

    def test_met793_does_not_claim_hbond(self):
        result = self.engine.analyze(
            self.candidates
        ).set_index("candidate")

        reason = result.loc[
            "EGFR_CAND_020", "reason"
        ]

        self.assertIn(
            "distance descriptor", reason
        )
        self.assertIn(
            "H-bond unverified", reason
        )
        self.assertNotIn(
            "Favorable MET793 geometry", reason
        )

    def test_no_met793_bonus_when_far(self):
        result = self.engine.analyze(
            self.candidates
        ).set_index("candidate")

        reason = result.loc[
            "EGFR_CAND_003", "reason"
        ]

        self.assertNotIn(
            "Short MET793 proximity", reason
        )

    def test_review_field_does_not_silently_score(self):
        baseline = self.engine.analyze(
            self.candidates
        )

        modified = self.candidates.copy()
        modified["hinge_hbond_review"] = [
            "not_checked",
            "geometry_plausible"
        ]

        checked = self.engine.analyze(modified)

        self.assertEqual(
            baseline["score"].tolist(),
            checked["score"].tolist()
        )


if __name__ == "__main__":
    unittest.main()
