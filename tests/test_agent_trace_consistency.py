import json
import unittest
from pathlib import Path

import pandas as pd

from egfr_agent.decision_engine import EGFRDecisionEngine
from egfr_agent.pose_review_evidence import render_pose_review
from egfr_agent.tools.structure_tool import StructureTool


ROOT = Path(__file__).resolve().parents[1]


RESEARCH_FIXTURES = [
    ROOT / "reports/EGFR_AI_Agent_Tool_Trace.json",
    ROOT / "reports/EGFR_AI_Agent_Report.md",
    ROOT / "results/zinc15_screening/phase5_final_candidate_decision_matrix.csv",
    ROOT / "results/zinc15_screening/phase4E_final_candidate_evidence_matrix.csv",
]

MISSING_RESEARCH_FIXTURES = [
    str(path.relative_to(ROOT))
    for path in RESEARCH_FIXTURES
    if not path.is_file()
]


@unittest.skipIf(
    bool(MISSING_RESEARCH_FIXTURES),
    "Research integration fixtures missing: "
    + ", ".join(MISSING_RESEARCH_FIXTURES),
)
class TestAgentTraceConsistency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        trace_path = ROOT / (
            "reports/EGFR_AI_Agent_Tool_Trace.json"
        )
        report_path = ROOT / (
            "reports/EGFR_AI_Agent_Report.md"
        )

        cls.trace = json.loads(
            trace_path.read_text(encoding="utf-8")
        )
        cls.report = report_path.read_text(
            encoding="utf-8"
        )

        cls.structure = StructureTool().run(top_n=10)
        cls.ranking = EGFRDecisionEngine().analyze(
            cls.structure
        )
        cls.best = cls.ranking.iloc[0]

        cid = str(cls.best["candidate"])
        matched = cls.structure.loc[
            cls.structure["phase4C_candidate_id"] == cid
        ]
        if len(matched) != 1:
            raise AssertionError(
                "Primary candidate evidence not unique"
            )

        cls.candidate = matched.iloc[0]

    def test_01_tool_authorization(self):
        t = self.trace
        allowed = set(t["scope"]["allowed_tools"])
        prohibited = set(t["scope"]["prohibited_tools"])
        required = t["required_tools"]
        executed = t["executed_tools"]
        requested = t["model_requested_tools"]

        self.assertEqual(
            set(executed),
            {"qsar_prediction", "structure_analysis"}
        )
        self.assertEqual(set(required), set(executed))
        self.assertEqual(set(requested), set(executed))
        self.assertEqual(len(executed), len(set(executed)))
        self.assertTrue(set(executed) <= allowed)
        self.assertFalse(set(executed) & prohibited)

    def test_02_call_log_records(self):
        t = self.trace
        events = t["call_log"]

        self.assertIsInstance(events, list)
        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(
            events[0]["event"],
            "scope_resolved"
        )

        encoded = json.dumps(
            events, ensure_ascii=False
        )

        for tool in t["executed_tools"]:
            self.assertIn(
                tool, encoded,
                f"Tool missing from call log: {tool}"
            )

    def test_03_deterministic_decision(self):
        decision = self.trace["trusted_decision"]

        self.assertEqual(
            decision["primary_candidate"],
            self.best["candidate"]
        )
        self.assertEqual(
            decision["primary_candidate"],
            "EGFR_CAND_003"
        )
        self.assertAlmostEqual(
            float(decision["heuristic_decision_score"]),
            float(self.best["score"]),
            places=6
        )
        self.assertEqual(
            decision["candidates_considered"],
            len(self.structure)
        )
        self.assertEqual(
            decision["decision_source"],
            "deterministic_decision_engine"
        )

    def test_04_candidate_evidence_values(self):
        decision = self.trace["trusted_decision"]
        row = self.candidate

        self.assertEqual(
            decision["candidate_track"],
            row["candidate_track"]
        )

        pairs = [
            ("qsar_predicted_pIC50",
             "qsar_predicted_pIC50"),
            ("max_training_tanimoto",
             "max_training_tanimoto"),
            ("rf_tree_sd_pIC50",
             "rf_tree_sd_pIC50"),
            ("MET793_distance_A",
             "MET793_distance"),
            ("key_ATP_contact_coverage",
             "key_ATP_contact_coverage"),
        ]

        for trace_key, data_key in pairs:
            with self.subTest(field=trace_key):
                self.assertAlmostEqual(
                    float(decision[trace_key]),
                    float(row[data_key]),
                    places=5
                )

    def test_05_rf_flag_definitions(self):
        audit = self.trace[
            "rf_flag_definition_audit"
        ]
        df = self.structure
        threshold = float(audit["rf_sd_threshold"])

        flags = (
            df["RF_spread_warning"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.assertTrue(
            flags.isin(["true", "false"]).all()
        )

        expected_category = (
            df["candidate_track"]
            == "SUPPORTED_B_HIGH_SPREAD"
        )

        self.assertEqual(
            (flags == "true").tolist(),
            expected_category.tolist()
        )
        self.assertEqual(
            audit["original_category_flag_true"],
            int(expected_category.sum())
        )
        self.assertEqual(
            audit["rf_sd_above_threshold"],
            int((
                pd.to_numeric(
                    df["rf_tree_sd_pIC50"]
                ) > threshold
            ).sum())
        )
        self.assertTrue(audit["source_definition_valid"])

    def test_06_report_and_review_consistency(self):
        candidate_id = str(self.best["candidate"])
        self.assertIn(
            f"**Candidate:** {candidate_id}",
            self.report
        )

        ids = (
            self.ranking["candidate"]
            .head(10)
            .astype(str)
            .tolist()
        )

        expected_review = render_pose_review(ids)
        expected_review = expected_review.replace(
            "## Manual pose review evidence",
            "### 6.1 Manual pose review evidence",
            1
        )

        self.assertIn(
            expected_review.strip(),
            self.report
        )
        self.assertIn(
            "## 6. Evidence provenance",
            self.report
        )
        self.assertIn(
            "## 8. Native function-calling audit",
            self.report
        )

    def test_07_scientific_claim_boundaries(self):
        t = self.trace

        self.assertFalse(
            t["narrative_audit"][
                "publish_as_scientific_conclusion"
            ]
        )
        self.assertTrue(
            t["llm_summary_status"].startswith(
                "unverified"
            )
        )
        self.assertEqual(
            t["trusted_decision"]["experimental_status"],
            "not_validated"
        )
        self.assertEqual(
            t["trusted_decision"]["evidence_source"],
            "precomputed_candidate_evidence"
        )
        self.assertIn(
            "unassessed (not finalized)",
            self.report
        )


if __name__ == "__main__":
    unittest.main()
