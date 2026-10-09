from pathlib import Path

import pandas as pd

from egfr_agent.decision_engine import EGFRDecisionEngine


def build_demo_evidence():
    """Synthetic examples; not real drug candidates."""

    return pd.DataFrame([
        {
            "phase4C_candidate_id": "DEMO_SYNTH_A",
            "candidate_track": "SUPPORTED_A",
            "consensus_rank": 1,
            "qsar_predicted_pIC50": 6.9,
            "key_ATP_contact_coverage": 0.75,
            "MET793_distance": 5.2,
        },
        {
            "phase4C_candidate_id": "DEMO_SYNTH_B",
            "candidate_track": "EXPLORATORY_OOD",
            "consensus_rank": 2,
            "qsar_predicted_pIC50": 7.2,
            "key_ATP_contact_coverage": 0.85,
            "MET793_distance": 3.3,
        },
        {
            "phase4C_candidate_id": "DEMO_SYNTH_C",
            "candidate_track": "SUPPORTED_B_HIGH_SPREAD",
            "consensus_rank": 3,
            "qsar_predicted_pIC50": 6.6,
            "key_ATP_contact_coverage": 0.30,
            "MET793_distance": 6.0,
        },
    ])


def generate_demo_report():
    evidence = build_demo_evidence()

    ranking = EGFRDecisionEngine().analyze(
        evidence
    )

    tracks = evidence.set_index(
        "phase4C_candidate_id"
    )["candidate_track"]

    lines = [
        "# EGFR Decision Engine - Synthetic Mini Demo",
        "",
        "**Status: SYNTHETIC DEMONSTRATION ONLY**",
        "",
        "The values and candidate IDs are artificial.",
        "They do not represent measured or predicted",
        "activity for real molecules.",
        "",
        "## Heuristic Candidate Ranking",
        "",
        "| Candidate | Track | Score |",
        "|---|---|---:|",
    ]

    for _, row in ranking.iterrows():
        candidate = str(row["candidate"])

        lines.append(
            f"| {candidate} | "
            f"{tracks[candidate]} | "
            f"{int(row['score'])} |"
        )

    lines.extend([
        "",
        "## Decision Evidence",
        "",
    ])

    for _, row in ranking.iterrows():
        lines.append(
            f"- **{row['candidate']}**: "
            f"{row['reason']}"
        )

    lines.extend([
        "",
        "## Scientific Limitations",
        "",
        "- Synthetic numerical inputs only.",
        "- No QSAR model inference performed.",
        "- No molecular docking performed.",
        "- MET793 distance is not proof of hydrogen bonding.",
        "- Heuristic scores are not binding affinities.",
        "- No experimental EGFR inhibition validation.",
        "- No candidate is approved for experimental use.",
        "",
    ])

    return "\n".join(lines)


def main():
    report = generate_demo_report()

    output = Path(
        "reports/EGFR_Mini_Demo_Report.md"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        report,
        encoding="utf-8",
    )

    print(report)
    print(f"\nDEMO REPORT SAVED: {output}")


if __name__ == "__main__":
    main()
