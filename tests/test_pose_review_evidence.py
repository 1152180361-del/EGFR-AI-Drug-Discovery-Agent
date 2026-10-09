import unittest
from unittest.mock import patch

import pandas as pd

from egfr_agent.pose_review_evidence import (
    render_pose_review
)


class TestPoseReviewEvidence(unittest.TestCase):

    def setUp(self):
        self.frame = pd.DataFrame([
            {
                "phase4C_candidate_id": "EGFR_CAND_003",
                "manual_review_status": "not_started",
                "hinge_hbond_review": "not_checked",
                "structural_review_outcome": "unassessed",
            },
            {
                "phase4C_candidate_id": "EGFR_CAND_020",
                "manual_review_status": "in_progress",
                "hinge_hbond_review": "geometry_plausible",
                "structural_review_outcome": "unassessed",
            },
        ])

    def test_cand020_local_geometry_only(self):
        text = render_pose_review(
            ["EGFR_CAND_020"], self.frame
        )
        self.assertIn("geometry_plausible", text)
        self.assertIn(
            "unassessed (not finalized)", text
        )

    def test_cand003_not_started(self):
        text = render_pose_review(
            ["EGFR_CAND_003"], self.frame
        )
        self.assertIn("not_started", text)
        self.assertIn("not_checked", text)

    def test_missing_candidate_is_not_reviewed(self):
        text = render_pose_review(
            ["EGFR_CAND_999"], self.frame
        )
        self.assertIn("not_reviewed", text)
        self.assertIn("unassessed", text)

    def test_incomplete_review_cannot_approve_pose(self):
        frame = self.frame.copy()
        frame.loc[
            frame["phase4C_candidate_id"]
            == "EGFR_CAND_020",
            "structural_review_outcome"
        ] = "plausible"

        text = render_pose_review(
            ["EGFR_CAND_020"], frame
        )
        self.assertIn(
            "unassessed (not finalized)", text
        )
        self.assertNotIn(
            "| plausible |", text
        )

    def test_duplicate_id_rejected(self):
        frame = pd.concat(
            [self.frame, self.frame.iloc[[0]]],
            ignore_index=True
        )
        with self.assertRaises(ValueError):
            render_pose_review(
                ["EGFR_CAND_003"], frame
            )

    def test_missing_review_file_is_safe(self):
        with patch(
            "egfr_agent.pose_review_evidence.read_review",
            side_effect=FileNotFoundError
        ):
            text = render_pose_review(
                ["EGFR_CAND_003"]
            )
        self.assertIn(
            "Review evidence unavailable", text
        )


if __name__ == "__main__":
    unittest.main()
