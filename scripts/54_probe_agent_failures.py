from unittest.mock import patch

import numpy as np
import pandas as pd

from egfr_agent.decision_engine import EGFRDecisionEngine
from egfr_agent.pose_review_evidence import render_pose_review
from egfr_agent.report_generator import EGFRReportGenerator


BASE = {
    "phase4C_candidate_id": "EGFR_TEST_001",
    "candidate_track": "SUPPORTED_A",
    "consensus_rank": 1,
    "qsar_predicted_pIC50": 7.2,
    "key_ATP_contact_coverage": 0.8,
    "MET793_distance": 3.0,
}


def decision_probe(label, change):
    row = BASE.copy()
    change(row)

    try:
        result = EGFRDecisionEngine().analyze(
            pd.DataFrame([row])
        )
        score = result.iloc[0]["score"]
        print(f"{label}: ACCEPTED, score={score}")
    except Exception as exc:
        print(
            f"{label}: REJECTED, "
            f"{type(exc).__name__}: {exc}"
        )


def other_probe(label, function):
    try:
        result = function()
        if isinstance(result, str):
            result = result.replace("\n", " ")[:140]
        print(f"{label}: RETURNED {result}")
    except Exception as exc:
        print(
            f"{label}: REJECTED, "
            f"{type(exc).__name__}: {exc}"
        )


def invalid_review():
    return pd.DataFrame([{
        "phase4C_candidate_id": "EGFR_TEST_001",
        "manual_review_status": "completed",
        "hinge_hbond_review": "confirmed",
        "structural_review_outcome": "plausible",
    }])


def duplicate_review():
    frame = pd.DataFrame([{
        "phase4C_candidate_id": "EGFR_TEST_001",
        "manual_review_status": "not_started",
        "hinge_hbond_review": "not_checked",
        "structural_review_outcome": "unassessed",
    }])
    return pd.concat([frame, frame], ignore_index=True)


def missing_review_file():
    with patch(
        "egfr_agent.pose_review_evidence.read_review",
        side_effect=FileNotFoundError
    ):
        return render_pose_review(["EGFR_TEST_001"])


def main():
    print("=" * 60)
    print("PHASE 7.4.1 - AGENT FAILURE INJECTION BASELINE")
    print("=" * 60)

    print("\n[1] DECISION INPUT VALIDATION")

    decision_probe(
        "Missing QSAR prediction",
        lambda r: r.pop("qsar_predicted_pIC50")
    )

    decision_probe(
        "NaN QSAR prediction",
        lambda r: r.update(
            qsar_predicted_pIC50=np.nan
        )
    )

    decision_probe(
        "Missing MET793 descriptor",
        lambda r: r.pop("MET793_distance")
    )

    decision_probe(
        "Invalid candidate track",
        lambda r: r.update(
            candidate_track="UNKNOWN_TRACK"
        )
    )

    decision_probe(
        "Invalid consensus rank type",
        lambda r: r.update(
            consensus_rank="invalid"
        )
    )

    print("\n[2] POSE REVIEW INPUT VALIDATION")

    other_probe(
        "Invalid hinge status",
        lambda: render_pose_review(
            ["EGFR_TEST_001"], invalid_review()
        )
    )

    other_probe(
        "Duplicate review ID",
        lambda: render_pose_review(
            ["EGFR_TEST_001"], duplicate_review()
        )
    )

    other_probe(
        "Missing review file",
        missing_review_file
    )

    print("\n[3] REPORT INPUT VALIDATION")

    other_probe(
        "Empty candidate ranking",
        lambda: EGFRReportGenerator().generate(
            pd.DataFrame(),
            pd.DataFrame([BASE])
        )
    )

    other_probe(
        "Missing candidate evidence",
        lambda: EGFRReportGenerator.find_candidate(
            pd.DataFrame([BASE]),
            "EGFR_TEST_999"
        )
    )

    print("\nBASELINE PROBE COMPLETE")
    print("No production files were modified.")
    print(
        "ACCEPTED does not necessarily mean safe: "
        "inspect missing or invalid required fields."
    )


if __name__ == "__main__":
    main()
