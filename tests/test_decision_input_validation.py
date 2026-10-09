import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from egfr_agent.decision_engine import EGFRDecisionEngine
from egfr_agent.tools.structure_tool import StructureTool

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_FIXTURES = [
    ROOT / "results/zinc15_screening/phase5_final_candidate_decision_matrix.csv",
    ROOT / "results/zinc15_screening/phase4E_final_candidate_evidence_matrix.csv",
]

MISSING_EVIDENCE_FIXTURES = [
    str(path.relative_to(ROOT))
    for path in EVIDENCE_FIXTURES
    if not path.is_file()
]


BASE = {
    "phase4C_candidate_id": "EGFR_TEST_001",
    "candidate_track": "SUPPORTED_A",
    "consensus_rank": 1,
    "qsar_predicted_pIC50": 7.2,
    "key_ATP_contact_coverage": 0.8,
    "MET793_distance": 3.0,
}


class TestDecisionInputValidation(unittest.TestCase):

    def setUp(self):
        self.engine = EGFRDecisionEngine()

    def test_valid_score_preserved(self):
        result = self.engine.analyze(pd.DataFrame([BASE]))
        self.assertEqual(int(result.iloc[0]["score"]), 13)

    def test_invalid_values_rejected(self):
        cases = {
            "missing_qsar_value":
                {"qsar_predicted_pIC50": None},
            "nan_qsar":
                {"qsar_predicted_pIC50": np.nan},
            "missing_met_value":
                {"MET793_distance": None},
            "unknown_track":
                {"candidate_track": "UNKNOWN_TRACK"},
            "invalid_rank":
                {"consensus_rank": "invalid"},
            "infinite_qsar":
                {"qsar_predicted_pIC50": np.inf},
            "invalid_coverage":
                {"key_ATP_contact_coverage": 1.5},
            "negative_distance":
                {"MET793_distance": -2},
        }

        for name, changes in cases.items():
            with self.subTest(name=name):
                row = {**BASE, **changes}
                with self.assertRaises(ValueError):
                    self.engine.analyze(pd.DataFrame([row]))

    def test_missing_column_rejected(self):
        data = pd.DataFrame([BASE]).drop(
            columns=["qsar_predicted_pIC50"]
        )
        with self.assertRaises(ValueError):
            self.engine.analyze(data)

    def test_duplicate_candidate_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.analyze(pd.DataFrame([BASE, BASE]))

    @unittest.skipIf(
        bool(MISSING_EVIDENCE_FIXTURES),
        "Research evidence missing: "
        + ", ".join(MISSING_EVIDENCE_FIXTURES),
    )
    def test_real_candidate_evidence(self):
        data = StructureTool().run(top_n=100000)
        result = self.engine.analyze(data)

        self.assertEqual(len(result), len(data))
        self.assertEqual(
            str(result.iloc[0]["candidate"]),
            "EGFR_CAND_003"
        )


if __name__ == "__main__":
    unittest.main()
