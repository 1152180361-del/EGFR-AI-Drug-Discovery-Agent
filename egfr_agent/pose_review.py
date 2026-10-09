import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .tools.structure_tool import StructureTool
from .risk_engine import EGFRAgentRiskEngine


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "results/zinc15_screening"

SHORTLIST_FILE = (
    DATA_DIR / "phase4E_visual_review_shortlist.csv"
)

EVIDENCE_FILE = (
    DATA_DIR / "phase4E_final_candidate_evidence_matrix.csv"
)

REVIEW_FILE = (
    ROOT / "reports/EGFR_Pose_Review_Matrix.csv"
)

REPORT_FILE = (
    ROOT / "reports/EGFR_Pose_Review_Summary.md"
)


REVIEW_DEFAULTS = {
    "manual_review_status": "not_started",
    "pose_orientation_review": "not_checked",
    "hinge_hbond_review": "not_checked",
    "steric_review": "not_checked",
    "ligand_chemistry_review": "not_checked",
    "protonation_stereo_review": "not_checked",
    "covalent_mechanism_review": "not_checked",
    "structural_review_outcome": "unassessed",
    "reviewer": "",
    "review_date": "",
    "pose_file": "",
    "review_image": "",
    "review_notes": "",
}


ALLOWED = {
    "manual_review_status": {
        "not_started",
        "in_progress",
        "completed",
    },
    "pose_orientation_review": {
        "not_checked",
        "plausible",
        "concern",
        "uncertain",
    },
    "hinge_hbond_review": {
        "not_checked",
        "geometry_plausible",
        "not_supported",
        "uncertain",
    },
    "steric_review": {
        "not_checked",
        "acceptable",
        "concern",
        "uncertain",
    },
    "ligand_chemistry_review": {
        "not_checked",
        "acceptable",
        "concern",
        "uncertain",
    },
    "protonation_stereo_review": {
        "not_checked",
        "acceptable",
        "concern",
        "uncertain",
    },
    "covalent_mechanism_review": {
        "not_checked",
        "likely_reversible",
        "possible_covalent",
        "uncertain",
    },
    "structural_review_outcome": {
        "unassessed",
        "plausible",
        "concern",
        "indeterminate",
    },
}


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_identifier_column(frame):

    possible = [
        "phase4C_candidate_id",
        "candidate_id",
        "candidate",
    ]

    for column in possible:

        if column not in frame.columns:
            continue

        values = (
            frame[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        if (
            len(values) == len(frame)
            and values.str.fullmatch(
                r"EGFR_CAND_\d+"
            ).all()
        ):
            return column

    raise ValueError(
        "Cannot identify candidate IDs in "
        "the Phase 4E shortlist. Columns: "
        + str(frame.columns.tolist())
    )


def create_matrix():

    if REVIEW_FILE.exists():
        raise FileExistsError(
            "Review matrix already exists. "
            "It was NOT overwritten: "
            + str(REVIEW_FILE)
        )

    if not SHORTLIST_FILE.is_file():
        raise FileNotFoundError(
            str(SHORTLIST_FILE)
        )

    shortlist = pd.read_csv(SHORTLIST_FILE)

    key = find_identifier_column(shortlist)

    candidate_ids = (
        shortlist[key]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(candidate_ids) != 9:
        raise ValueError(
            "Expected 9 Phase 4E review candidates, "
            f"found {len(candidate_ids)}."
        )

    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(
            "Duplicate IDs in visual shortlist."
        )

    # StructureTool combines the Phase 5 decision
    # table with the Phase 4E evidence matrix.
    evidence = StructureTool().run(
        top_n=100000
    )

    key = "phase4C_candidate_id"

    if evidence[key].duplicated().any():
        raise ValueError(
            "Duplicate IDs in candidate evidence."
        )

    evidence = evidence.set_index(
        key,
        drop=False,
    )

    missing = [
        cid
        for cid in candidate_ids
        if cid not in evidence.index
    ]

    if missing:
        raise ValueError(
            "Shortlist candidates missing from "
            "evidence table: "
            + str(missing)
        )

    matrix = evidence.loc[
        candidate_ids
    ].copy()

    engine = EGFRAgentRiskEngine()

    categories = []
    above_threshold = []

    for _, row in matrix.iterrows():

        assessment = engine.evaluate(row)

        categories.append(
            assessment["confidence"]
        )

        above_threshold.append(
            assessment["rf_sd_above_threshold"]
        )

    matrix["computational_evidence_category"] = (
        categories
    )

    matrix["rf_sd_above_threshold"] = (
        above_threshold
    )

    # These fields must be entered by a reviewer.
    # No automated structural approval is assumed.
    for column, value in REVIEW_DEFAULTS.items():
        matrix[column] = value

    preferred_columns = [
        "phase4C_candidate_id",
        "candidate_track",
        "candidate_role",
        "consensus_rank",
        "rank_v2",
        "rank_v3",
        "computational_evidence_category",
        "qsar_predicted_pIC50",
        "max_training_tanimoto",
        "rf_tree_sd_pIC50",
        "RF_spread_warning",
        "rf_sd_above_threshold",
        "top1_vina_score",
        "reference_contact_jaccard",
        "key_ATP_contact_coverage",
        "MET793_distance",
        "severe_clash_pairs_lt1p5A",
        "no_severe_clash",
    ]

    ordered_columns = [
        col
        for col in preferred_columns
        if col in matrix.columns
    ] + list(REVIEW_DEFAULTS)

    matrix = matrix[ordered_columns]

    REVIEW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 60)
    print("POSE REVIEW MATRIX CREATED")
    print("=" * 60)

    print("Candidates:", len(matrix))
    print("Output:", REVIEW_FILE)

    print(
        "\nCandidate IDs:",
        candidate_ids,
    )

    print(
        "\nAll manual reviews start as "
        "'not_started'."
    )

    return matrix


def read_review():

    if not REVIEW_FILE.is_file():
        raise FileNotFoundError(
            "Review matrix not found. "
            "Run --init first."
        )

    frame = pd.read_csv(
        REVIEW_FILE,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    key = "phase4C_candidate_id"

    if key not in frame.columns:
        raise ValueError(
            "Candidate identifier column missing."
        )

    if frame[key].duplicated().any():
        raise ValueError(
            "Duplicate candidate IDs in review."
        )

    for column, allowed_values in ALLOWED.items():

        if column not in frame.columns:
            raise ValueError(
                f"Missing manual review column: {column}"
            )

        values = set(
            frame[column]
            .astype(str)
            .str.strip()
        )

        invalid = values - allowed_values

        if invalid:
            raise ValueError(
                f"Invalid values for {column}: "
                + str(sorted(invalid))
            )

    return frame


def markdown_value(value):

    if pd.isna(value) or str(value) == "":
        return "—"

    return str(value).replace("|", "/")


def generate_report():

    frame = read_review()

    status_counts = (
        frame["manual_review_status"]
        .value_counts()
        .to_dict()
    )

    completed = frame.loc[
        frame["manual_review_status"]
        == "completed"
    ]

    review_checks = [
        "pose_orientation_review",
        "hinge_hbond_review",
        "steric_review",
        "ligand_chemistry_review",
        "protonation_stereo_review",
        "covalent_mechanism_review",
    ]

    incomplete_documentation = []

    for _, row in completed.iterrows():

        missing_checks = [
            column
            for column in review_checks
            if row[column] == "not_checked"
        ]

        if not str(row["reviewer"]).strip():
            missing_checks.append("reviewer")

        if not str(row["review_image"]).strip():
            missing_checks.append("review_image")

        if (
            row["structural_review_outcome"]
            == "unassessed"
        ):
            missing_checks.append(
                "structural_review_outcome"
            )

        if missing_checks:
            incomplete_documentation.append({
                "candidate": (
                    row["phase4C_candidate_id"]
                ),
                "missing": missing_checks,
            })

    lines = [
        "# EGFR Candidate Pose Review",
        "",
        "## Scope",
        "",
        f"- Candidates in review matrix: {len(frame)}",
        "- Data source: Phase 4E visual review shortlist",
        "- Structural evidence: existing docking outputs",
        "- This review does not run new docking.",
        "",
        "## Manual review progress",
        "",
        f"- Not started: {status_counts.get('not_started', 0)}",
        f"- In progress: {status_counts.get('in_progress', 0)}",
        f"- Marked completed: {status_counts.get('completed', 0)}",
        "- Completed with missing documentation: "
        f"{len(incomplete_documentation)}",
        "",
        "## Candidate overview",
        "",
        "| Candidate | Track | Consensus rank | "
        "MET793 proximity (A) | ATP coverage | "
        "Review status | Manual outcome |",
        "|---|---|---:|---:|---:|---|---|",
    ]

    for _, row in frame.iterrows():

        lines.append(
            "| "
            + markdown_value(
                row["phase4C_candidate_id"]
            )
            + " | "
            + markdown_value(
                row.get("candidate_track", "")
            )
            + " | "
            + markdown_value(
                row.get("consensus_rank", "")
            )
            + " | "
            + markdown_value(
                row.get("MET793_distance", "")
            )
            + " | "
            + markdown_value(
                row.get(
                    "key_ATP_contact_coverage", ""
                )
            )
            + " | "
            + row["manual_review_status"]
            + " | "
            + row["structural_review_outcome"]
            + " |"
        )

    lines.extend([
        "",
        "## Structural interpretation rules",
        "",
        "- MET793 distance is a proximity descriptor, "
        "not evidence of a validated hydrogen bond.",
        "- Hydrogen-bond assessment requires suitable "
        "donor/acceptor chemistry and geometry.",
        "- No atom pairs below 1.5 A does not guarantee "
        "the pose is chemically plausible.",
        "- ATP-pocket contact coverage is not "
        "an experimental potency measurement.",
        "- Vina score is not experimental binding "
        "free energy.",
        "- Computational support categories "
        "are not experimentally calibrated.",
        "- The original Phase 4E RF category flag "
        "and numeric RF SD threshold flag "
        "have different definitions.",
        "- Manual structural review does not "
        "establish EGFR inhibition.",
        "",
        "## Source provenance",
        "",
        f"- Phase 4E shortlist SHA256: "
        f"`{file_hash(SHORTLIST_FILE)}`",
        f"- Phase 4E evidence SHA256: "
        f"`{file_hash(EVIDENCE_FILE)}`",
        "",
    ])

    if incomplete_documentation:

        lines.extend([
            "## Documentation gaps",
            "",
        ])

        for item in incomplete_documentation:
            lines.append(
                "- "
                + item["candidate"]
                + ": "
                + ", ".join(item["missing"])
            )

        lines.append("")

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 60)
    print("POSE REVIEW SUMMARY")
    print("=" * 60)

    print("Candidates:", len(frame))
    print(
        "Not started:",
        status_counts.get("not_started", 0),
    )
    print(
        "Completed:",
        status_counts.get("completed", 0),
    )

    print(
        "Incomplete documented reviews:",
        len(incomplete_documentation),
    )

    print("Report:", REPORT_FILE)


def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--init",
        action="store_true",
    )

    group.add_argument(
        "--report",
        action="store_true",
    )

    args = parser.parse_args()

    if args.init:
        create_matrix()

    generate_report()


if __name__ == "__main__":
    main()
