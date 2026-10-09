from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)

POSE_PATH = (
    ROOT
    / "docking/3W32/phase4C/reports/"
      "phase4C_top1_pose_features.csv"
)

CANDIDATE_PATH = (
    ROOT
    / "results/zinc15_screening/"
      "zinc15_phase4C_candidates.csv"
)

OUT_DIR = (
    ROOT
    / "results/zinc15_screening"
)

MATRIX_OUT = (
    OUT_DIR
    / "phase4E_final_candidate_evidence_matrix.csv"
)

SHORTLIST_OUT = (
    OUT_DIR
    / "phase4E_visual_review_shortlist.csv"
)

REPORT_OUT = (
    OUT_DIR
    / "phase4E_final_prioritization_report.json"
)

MD_OUT = (
    ROOT
    / "reports/"
      "phase4E_final_candidate_prioritization.md"
)


N_EXPLORATORY_STRUCTURAL = 5


# ============================================================
# Load
# ============================================================

pose = pd.read_csv(
    POSE_PATH
)

master = pd.read_csv(
    CANDIDATE_PATH
)


print(
    "\n========================================"
)

print(
    "PHASE 4E-2 FINAL PRIORITIZATION"
)

print(
    "========================================"
)


print(
    "\nPose records:",
    len(pose)
)


# ============================================================
# Merge chemical metadata
# ============================================================

extra_columns = [
    "phase4C_candidate_id",
    "standardized_smiles",
    "murcko_scaffold",
    "generic_murcko_scaffold",
]


extra_columns = [
    x
    for x in extra_columns
    if x in master.columns
]


matrix = pose.merge(
    master[
        extra_columns
    ],
    on="phase4C_candidate_id",
    how="left",
)


# ============================================================
# Boolean normalization
# ============================================================

def to_bool_series(series):

    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


matrix[
    "MET793_any_close_3p5A"
] = to_bool_series(
    matrix[
        "MET793_any_close_3p5A"
    ]
)


# ============================================================
# Pareto frontier helper
#
# NO weighted score.
#
# A candidate is dominated if another candidate
# is at least as good in all dimensions and
# strictly better in at least one dimension.
# ============================================================

def pareto_front(
    df,
    higher_better,
    lower_better,
):

    if len(df) == 0:

        return pd.Series(
            dtype=bool,
            index=df.index,
        )


    transformed = pd.DataFrame(
        index=df.index
    )


    for col in higher_better:

        transformed[
            col
        ] = (
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
        )


    for col in lower_better:

        transformed[
            col
        ] = -(
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
        )


    values = (
        transformed
        .to_numpy(
            dtype=float
        )
    )


    n = len(
        values
    )


    is_pareto = np.ones(
        n,
        dtype=bool,
    )


    for i in range(
        n
    ):

        if not np.all(
            np.isfinite(
                values[i]
            )
        ):

            is_pareto[i] = False
            continue


        for j in range(
            n
        ):

            if i == j:

                continue


            if not np.all(
                np.isfinite(
                    values[j]
                )
            ):

                continue


            at_least_as_good = np.all(
                values[j]
                >= values[i]
            )


            strictly_better = np.any(
                values[j]
                > values[i]
            )


            if (
                at_least_as_good
                and strictly_better
            ):

                is_pareto[i] = False
                break


    return pd.Series(
        is_pareto,
        index=df.index,
    )


# ============================================================
# Initial labels
# ============================================================

matrix[
    "final_evidence_category"
] = "UNASSIGNED"


matrix[
    "recommended_visual_review"
] = False


matrix[
    "prioritization_reason"
] = ""


# ============================================================
# 1. SUPPORTED_A
#
# Pareto frontier across independent evidence:
#
# Higher is better:
# - frozen QSAR prediction
# - training-space similarity
# - W32 contact Jaccard
# - key ATP coverage
#
# Lower is better:
# - RF tree spread
#
# Vina score is deliberately excluded.
# ============================================================

supported_A_mask = (
    matrix[
        "candidate_track"
    ]
    == "SUPPORTED_A"
)


supported_A = matrix[
    supported_A_mask
].copy()


pareto_mask = pareto_front(
    supported_A,

    higher_better=[
        "qsar_predicted_pIC50",
        "max_training_tanimoto",
        "reference_contact_jaccard",
        "key_ATP_contact_coverage",
    ],

    lower_better=[
        "rf_tree_sd_pIC50",
    ],
)


pareto_indices = list(
    supported_A.index[
        pareto_mask
    ]
)


reserve_indices = list(
    supported_A.index[
        ~pareto_mask
    ]
)


matrix.loc[
    pareto_indices,
    "final_evidence_category",
] = (
    "SUPPORTED_PARETO_PRIORITY"
)


matrix.loc[
    pareto_indices,
    "recommended_visual_review",
] = True


matrix.loc[
    pareto_indices,
    "prioritization_reason",
] = (
    "Non-dominated SUPPORTED_A candidate across "
    "frozen QSAR prediction, training-space similarity, "
    "RF tree spread, W32-contact Jaccard, and key ATP "
    "contact coverage. Vina score excluded."
)


matrix.loc[
    reserve_indices,
    "final_evidence_category",
] = (
    "SUPPORTED_RESERVE"
)


matrix.loc[
    reserve_indices,
    "prioritization_reason",
] = (
    "Within strict QSAR similarity-support region, "
    "but dominated by another supported candidate "
    "in the predefined multi-evidence Pareto analysis."
)


# ============================================================
# 2. SUPPORTED_B
#
# Similarity-supported but RF tree spread above
# training p95. Keep, but do not elevate.
# ============================================================

supported_B_mask = (
    matrix[
        "candidate_track"
    ]
    == "SUPPORTED_B_HIGH_SPREAD"
)


matrix.loc[
    supported_B_mask,
    "final_evidence_category",
] = (
    "SUPPORTED_HIGH_SPREAD_CAUTION"
)


matrix.loc[
    supported_B_mask,
    "prioritization_reason",
] = (
    "Within similarity-supported chemical space, "
    "but RF tree-to-tree prediction spread exceeded "
    "the predefined training p95 warning threshold."
)


# ============================================================
# 3. EXPLORATORY OOD
#
# These are NOT promoted using quantitative QSAR potency.
#
# Predefined structural eligibility:
# - MET793 heteroatom proximity <= 3.5 A
# - zero severe clashes
#
# Among eligible hypotheses:
# rank descriptively by:
# 1. W32-contact Jaccard
# 2. key ATP coverage
#
# No Vina score is used.
# ============================================================

ood_mask = (
    matrix[
        "candidate_track"
    ]
    == "EXPLORATORY_OOD"
)


ood = matrix[
    ood_mask
].copy()


structural_eligible = ood[
    (
        ood[
            "MET793_any_close_3p5A"
        ]
    )
    &
    (
        ood[
            "severe_clash_pairs_lt1p5A"
        ] == 0
    )
].copy()


structural_eligible = (
    structural_eligible
    .sort_values(
        [
            "reference_contact_jaccard",
            "key_ATP_contact_coverage",
            "phase4C_candidate_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )
)


exploratory_selected = (
    structural_eligible
    .head(
        N_EXPLORATORY_STRUCTURAL
    )
    .copy()
)


exploratory_selected_indices = list(
    exploratory_selected.index
)


matrix.loc[
    exploratory_selected_indices,
    "final_evidence_category",
] = (
    "EXPLORATORY_STRUCTURAL_HYPOTHESIS"
)


matrix.loc[
    exploratory_selected_indices,
    "recommended_visual_review",
] = True


matrix.loc[
    exploratory_selected_indices,
    "prioritization_reason",
] = (
    "Outside strict QSAR similarity-support region; "
    "retained only as a structure-based hypothesis "
    "because the Top-1 pose satisfies the predefined "
    "MET793 <=3.5 A geometric-proximity criterion, "
    "has no severe clashes, and ranks highly by "
    "W32-contact similarity. Not a potency-supported hit."
)


# Other OOD candidates
ood_remaining = (
    ood.index.difference(
        exploratory_selected_indices
    )
)


matrix.loc[
    ood_remaining,
    "final_evidence_category",
] = (
    "EXPLORATORY_RESERVE"
)


matrix.loc[
    ood_remaining,
    "prioritization_reason",
] = (
    "Outside strict QSAR similarity-support region; "
    "retained as exploratory reserve without elevation "
    "to the structural-hypothesis shortlist."
)


# ============================================================
# Evidence flags
# ============================================================

matrix[
    "strict_QSAR_support"
] = matrix[
    "candidate_track"
].isin(
    [
        "SUPPORTED_A",
        "SUPPORTED_B_HIGH_SPREAD",
    ]
)


matrix[
    "RF_spread_warning"
] = (
    matrix[
        "candidate_track"
    ]
    == "SUPPORTED_B_HIGH_SPREAD"
)


matrix[
    "W32_like_hinge_geometry"
] = matrix[
    "MET793_any_close_3p5A"
]


matrix[
    "no_severe_clash"
] = (
    matrix[
        "severe_clash_pairs_lt1p5A"
    ] == 0
)


# ============================================================
# Stable category order
# ============================================================

category_order = {
    "SUPPORTED_PARETO_PRIORITY":
        0,

    "SUPPORTED_RESERVE":
        1,

    "SUPPORTED_HIGH_SPREAD_CAUTION":
        2,

    "EXPLORATORY_STRUCTURAL_HYPOTHESIS":
        3,

    "EXPLORATORY_RESERVE":
        4,
}


matrix[
    "_category_order"
] = matrix[
    "final_evidence_category"
].map(
    category_order
)


matrix = (
    matrix
    .sort_values(
        [
            "_category_order",
            "qsar_predicted_pIC50",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


matrix[
    "final_matrix_order"
] = np.arange(
    1,
    len(matrix) + 1,
)


matrix = matrix.drop(
    columns=[
        "_category_order"
    ]
)


# ============================================================
# Shortlist
# ============================================================

shortlist = matrix[
    matrix[
        "recommended_visual_review"
    ]
].copy()


# Supported candidates first,
# exploratory hypotheses second.
shortlist_order = {
    "SUPPORTED_PARETO_PRIORITY":
        0,

    "EXPLORATORY_STRUCTURAL_HYPOTHESIS":
        1,
}


shortlist[
    "_order"
] = shortlist[
    "final_evidence_category"
].map(
    shortlist_order
)


shortlist = (
    shortlist
    .sort_values(
        [
            "_order",
            "reference_contact_jaccard",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .drop(
        columns=[
            "_order"
        ]
    )
    .reset_index(
        drop=True
    )
)


shortlist[
    "visual_review_order"
] = np.arange(
    1,
    len(shortlist) + 1,
)


# ============================================================
# Save
# ============================================================

matrix.to_csv(
    MATRIX_OUT,
    index=False,
)


shortlist.to_csv(
    SHORTLIST_OUT,
    index=False,
)


# ============================================================
# Summary counts
# ============================================================

category_counts = (
    matrix[
        "final_evidence_category"
    ]
    .value_counts()
)


report = {
    "input_docked_candidates":
        int(
            len(matrix)
        ),

    "selection_principles": {
        "vina_score_used":
            False,

        "weighted_combined_score_used":
            False,

        "supported_candidates":
            (
                "Pareto non-dominance across frozen QSAR "
                "prediction, training similarity, RF spread, "
                "W32-contact Jaccard and key ATP coverage."
            ),

        "exploratory_candidates":
            (
                "Outside strict QSAR support; structural "
                "hypotheses selected only using the pre-existing "
                "MET793 <=3.5 A geometric criterion, absence "
                "of severe clashes, and W32-contact similarity."
            ),
    },

    "category_counts": {
        str(k):
            int(v)
        for k, v
        in category_counts.items()
    },

    "visual_review_shortlist_n":
        int(
            len(shortlist)
        ),

    "scientific_boundary":
        (
            "The shortlist is a computational prioritization "
            "set, not a set of experimentally validated EGFR "
            "inhibitors. OOD structural hypotheses must not "
            "be described as quantitatively predicted-potent "
            "compounds."
        ),

    "outputs": {
        "evidence_matrix":
            str(
                MATRIX_OUT
            ),

        "visual_review_shortlist":
            str(
                SHORTLIST_OUT
            ),
    },
}


REPORT_OUT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


# ============================================================
# Markdown report
# ============================================================

supported_priority = shortlist[
    shortlist[
        "final_evidence_category"
    ]
    == "SUPPORTED_PARETO_PRIORITY"
]


exploratory_priority = shortlist[
    shortlist[
        "final_evidence_category"
    ]
    == "EXPLORATORY_STRUCTURAL_HYPOTHESIS"
]


lines = []

lines.append(
    "# Phase 4E – Final EGFR Candidate Prioritization"
)

lines.append("")

lines.append(
    "No weighted combined score was fitted."
)

lines.append(
    "Vina score was excluded from final prioritization because "
    "the retrospective benchmark did not support its use as an "
    "EGFR activity-ranking metric."
)

lines.append("")

lines.append(
    "## Supported Pareto-priority candidates"
)

lines.append("")


for _, row in supported_priority.iterrows():

    lines.append(
        f"- {row['phase4C_candidate_id']} / "
        f"{row['zinc_id']}: "
        f"QSAR={row['qsar_predicted_pIC50']:.3f}, "
        f"training similarity="
        f"{row['max_training_tanimoto']:.3f}, "
        f"RF SD={row['rf_tree_sd_pIC50']:.3f}, "
        f"W32 Jaccard="
        f"{row['reference_contact_jaccard']:.3f}, "
        f"key ATP coverage="
        f"{row['key_ATP_contact_coverage']:.3f}."
    )


lines.append("")

lines.append(
    "## Exploratory structural hypotheses"
)

lines.append("")


for _, row in exploratory_priority.iterrows():

    hinge_distance = min(
        float(
            row[
                "MET793_N_hetero_distance_A"
            ]
        ),
        float(
            row[
                "MET793_O_hetero_distance_A"
            ]
        ),
    )


    lines.append(
        f"- {row['phase4C_candidate_id']} / "
        f"{row['zinc_id']}: "
        f"outside strict QSAR support; "
        f"W32 Jaccard="
        f"{row['reference_contact_jaccard']:.3f}, "
        f"key ATP coverage="
        f"{row['key_ATP_contact_coverage']:.3f}, "
        f"nearest MET793 backbone heteroatom="
        f"{hinge_distance:.3f} Å."
    )


lines.append("")

lines.append(
    "These exploratory compounds are hypothesis-generating "
    "candidates only and are not quantitative potency predictions."
)


MD_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


MD_OUT.write_text(
    "\n".join(
        lines
    )
)


# ============================================================
# Terminal output
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL EVIDENCE CATEGORIES"
)

print(
    "========================================"
)


for category, count in (
    category_counts.items()
):

    print(
        f"{category}: {count}"
    )


print(
    "\n========================================"
)

print(
    "SUPPORTED PARETO PRIORITY"
)

print(
    "========================================"
)


display_cols = [
    "phase4C_candidate_id",
    "zinc_id",
    "qsar_predicted_pIC50",
    "max_training_tanimoto",
    "rf_tree_sd_pIC50",
    "reference_contact_jaccard",
    "key_ATP_contact_coverage",
    "MET793_N_hetero_distance_A",
    "MET793_O_hetero_distance_A",
]


print(
    supported_priority[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


print(
    "\n========================================"
)

print(
    "EXPLORATORY STRUCTURAL HYPOTHESES"
)

print(
    "========================================"
)


print(
    exploratory_priority[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


print(
    "\n========================================"
)

print(
    "FINAL VISUAL-REVIEW SHORTLIST:"
)

print(
    len(
        shortlist
    ),
    "candidates"
)

print(
    "========================================"
)


print(
    "\nSaved:"
)

print(
    MATRIX_OUT
)

print(
    SHORTLIST_OUT
)

print(
    REPORT_OUT
)

print(
    MD_OUT
)


print(
    "\n========================================"
)

print(
    "PHASE 4E-2 PRIORITIZATION: PASS"
)

print(
    "========================================"
)
