import pandas as pd

from .pose_review import read_review


REQUIRED = {
    "phase4C_candidate_id",
    "manual_review_status",
    "hinge_hbond_review",
    "structural_review_outcome",
}

VALID_STATUS = {
    "not_started", "in_progress", "completed"
}

VALID_HINGE = {
    "not_checked",
    "geometry_plausible",
    "not_supported",
    "uncertain",
}

VALID_OUTCOME = {
    "unassessed",
    "plausible",
    "concern",
    "indeterminate",
}


def render_pose_review(candidate_ids, review_df=None):
    """
    Produce a non-scoring review evidence section.

    Local H-bond geometry must not imply that the
    entire docking pose has passed manual review.
    """

    if review_df is None:
        try:
            review_df = read_review()
        except FileNotFoundError:
            return (
                "\n## Manual pose review evidence\n\n"
                "**Review evidence unavailable.** "
                "No candidate is considered reviewed.\n"
            )

    missing = REQUIRED - set(review_df.columns)
    if missing:
        raise ValueError(
            f"Missing review columns: {sorted(missing)}"
        )

    frame = review_df.copy()

    key = "phase4C_candidate_id"
    frame[key] = frame[key].astype(str).str.strip()

    if frame[key].duplicated().any():
        raise ValueError("Duplicate pose review candidate ID")

    checks = {
        "manual_review_status": VALID_STATUS,
        "hinge_hbond_review": VALID_HINGE,
        "structural_review_outcome": VALID_OUTCOME,
    }

    for column, allowed in checks.items():
        frame[column] = (
            frame[column].astype(str).str.strip()
        )
        invalid = set(frame[column]) - allowed
        if invalid:
            raise ValueError(
                f"Invalid {column}: {sorted(invalid)}"
            )

    lookup = frame.set_index(key)

    lines = [
        "",
        "## Manual pose review evidence",
        "",
        "This section reports independent review "
        "status and does not modify heuristic scores.",
        "",
        "| Candidate | Manual status | Hinge review | "
        "Overall pose assessment |",
        "|---|---|---|---|",
    ]

    ids = list(dict.fromkeys(
        str(x).strip() for x in candidate_ids
    ))

    for cid in ids:
        if cid not in lookup.index:
            status = "not_reviewed"
            hinge = "not_checked"
            overall = "unassessed"

        else:
            row = lookup.loc[cid]

            status = row["manual_review_status"]
            hinge = row["hinge_hbond_review"]

            if status == "completed":
                overall = row[
                    "structural_review_outcome"
                ]
            else:
                overall = "unassessed (not finalized)"

        lines.append(
            f"| {cid} | {status} | "
            f"{hinge} | {overall} |"
        )

    lines.extend([
        "",
        "- `geometry_plausible` indicates only "
        "preliminary local H-bond geometry.",
        "- It does not establish a stable H-bond, "
        "validated binding affinity or EGFR inhibition.",
        "- An unfinished manual review cannot "
        "constitute an approved overall pose.",
        "- Candidates absent from the matrix "
        "are treated as not reviewed.",
        "",
    ])

    return "\n".join(lines)
