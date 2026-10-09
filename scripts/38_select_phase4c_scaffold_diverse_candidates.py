from pathlib import Path
import json

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)

INPUT = (
    ROOT
    / "results/zinc15_screening/"
      "zinc15_top20000_qsar_AD.csv"
)

OUT_DIR = (
    ROOT
    / "results/zinc15_screening"
)

CANDIDATES_OUT = (
    OUT_DIR
    / "zinc15_phase4C_candidates.csv"
)

SUPPORTED_OUT = (
    OUT_DIR
    / "zinc15_phase4C_supported_candidates.csv"
)

EXPLORATORY_OUT = (
    OUT_DIR
    / "zinc15_phase4C_exploratory_OOD_candidates.csv"
)

REPORT_OUT = (
    OUT_DIR
    / "zinc15_phase4C_candidate_selection_report.json"
)


# ============================================================
# Settings
# ============================================================

N_EXPLORATORY = 40


# ============================================================
# Scaffold helpers
# ============================================================

def get_scaffolds(smiles):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:

        return (
            None,
            None,
        )


    scaffold = (
        MurckoScaffold
        .GetScaffoldForMol(
            mol
        )
    )


    if scaffold is None:

        return (
            None,
            None,
        )


    exact = Chem.MolToSmiles(
        scaffold,
        canonical=True,
        isomericSmiles=True,
    )


    # Acyclic molecules have empty Murcko scaffold.
    # Keep their whole canonical molecule as a
    # transparent fallback so unrelated acyclic
    # molecules are not collapsed into one scaffold.
    if not exact:

        canonical = (
            Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
            )
        )

        exact = (
            "ACYCLIC::"
            + canonical
        )

        generic = exact

        return (
            exact,
            generic,
        )


    try:

        generic_mol = (
            MurckoScaffold
            .MakeScaffoldGeneric(
                scaffold
            )
        )

        generic = (
            Chem.MolToSmiles(
                generic_mol,
                canonical=True,
                isomericSmiles=False,
            )
        )

    except Exception:

        generic = ""


    return (
        exact,
        generic,
    )


# ============================================================
# Load
# ============================================================

df = pd.read_csv(
    INPUT
)


print(
    "\n========================================"
)

print(
    "PHASE 4C CANDIDATE SELECTION"
)

print(
    "========================================"
)


print(
    "\nInput annotated candidates:",
    len(df)
)


required = [
    "zinc_id",
    "standardized_smiles",
    "qsar_predicted_pIC50",
    "max_training_tanimoto",
    "rf_tree_sd_pIC50",
    "priority_tier",
]


missing = [
    col
    for col in required
    if col not in df.columns
]


if missing:

    raise RuntimeError(
        f"Missing required columns: "
        f"{missing}"
    )


# ============================================================
# Calculate Murcko scaffolds for Top20k
# ============================================================

print(
    "\nCalculating Bemis-Murcko scaffolds..."
)


exact_scaffolds = []
generic_scaffolds = []


for smiles in df[
    "standardized_smiles"
]:

    exact, generic = (
        get_scaffolds(
            smiles
        )
    )

    exact_scaffolds.append(
        exact
    )

    generic_scaffolds.append(
        generic
    )


df[
    "murcko_scaffold"
] = exact_scaffolds


df[
    "generic_murcko_scaffold"
] = generic_scaffolds


invalid_scaffold = int(
    df[
        "murcko_scaffold"
    ].isna().sum()
)


print(
    "Scaffold failures:",
    invalid_scaffold
)


# ============================================================
# Track A:
# strict similarity-supported compounds
#
# Do NOT discard compounds simply because
# multiple members share a scaffold.
# There are only nine supported molecules
# and docking them all is inexpensive.
# ============================================================

supported = df[
    df[
        "priority_tier"
    ].isin(
        [
            "A_WITHIN_AD_SUPPORTED",
            "B_WITHIN_AD_HIGH_SPREAD",
        ]
    )
].copy()


supported = (
    supported
    .sort_values(
        [
            "priority_tier",
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


def supported_track(row):

    if (
        row[
            "priority_tier"
        ]
        == "A_WITHIN_AD_SUPPORTED"
    ):

        return (
            "SUPPORTED_A"
        )

    return (
        "SUPPORTED_B_HIGH_SPREAD"
    )


supported[
    "candidate_track"
] = supported.apply(
    supported_track,
    axis=1,
)


supported[
    "selection_reason"
] = supported[
    "candidate_track"
].map(
    {
        "SUPPORTED_A":
            (
                "Within strict similarity-support "
                "region and RF tree spread not above "
                "training p95."
            ),

        "SUPPORTED_B_HIGH_SPREAD":
            (
                "Within strict similarity-support "
                "region but RF tree spread exceeds "
                "training p95; retained as a "
                "cautionary supported candidate."
            ),
    }
)


# ============================================================
# Track B:
# scaffold-diverse OOD exploration
#
# Selection uses ONLY:
# 1. frozen QSAR ranking
# 2. unique exact Murcko scaffold
#
# No retrospective docking information is used.
# ============================================================

exploration_pool = df[
    df[
        "priority_tier"
    ]
    == "C_OUTSIDE_AD"
].copy()


exploration_pool = (
    exploration_pool
    .sort_values(
        "qsar_predicted_pIC50",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


# Avoid selecting exploration scaffolds that are
# already represented in the supported set.
supported_scaffolds = set(
    supported[
        "murcko_scaffold"
    ]
    .dropna()
    .astype(str)
)


selected_rows = []

seen_scaffolds = set(
    supported_scaffolds
)


for _, row in (
    exploration_pool.iterrows()
):

    scaffold = row[
        "murcko_scaffold"
    ]


    if pd.isna(
        scaffold
    ):

        continue


    scaffold = str(
        scaffold
    )


    if scaffold in seen_scaffolds:

        continue


    selected_rows.append(
        row.copy()
    )


    seen_scaffolds.add(
        scaffold
    )


    if (
        len(selected_rows)
        >= N_EXPLORATORY
    ):

        break


if len(selected_rows) == 0:

    exploratory = (
        exploration_pool.iloc[
            0:0
        ].copy()
    )

else:

    exploratory = pd.DataFrame(
        selected_rows
    )


exploratory[
    "candidate_track"
] = (
    "EXPLORATORY_OOD"
)


exploratory[
    "selection_reason"
] = (
    "Outside strict similarity-support region; "
    "retained as hypothesis-generating exploration "
    "candidate based on frozen QSAR rank and "
    "unique Bemis-Murcko scaffold."
)


# ============================================================
# Combine
# ============================================================

candidates = pd.concat(
    [
        supported,
        exploratory,
    ],
    ignore_index=True,
)


# Stable track ordering
track_order = {
    "SUPPORTED_A":
        0,

    "SUPPORTED_B_HIGH_SPREAD":
        1,

    "EXPLORATORY_OOD":
        2,
}


candidates[
    "_track_order"
] = candidates[
    "candidate_track"
].map(
    track_order
)


candidates = (
    candidates
    .sort_values(
        [
            "_track_order",
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


candidates[
    "phase4C_candidate_id"
] = [
    f"EGFR_CAND_{i+1:03d}"
    for i in range(
        len(candidates)
    )
]


candidates[
    "phase4C_candidate_rank"
] = np.arange(
    1,
    len(candidates) + 1,
)


candidates = candidates.drop(
    columns=[
        "_track_order"
    ]
)


# ============================================================
# Save
# ============================================================

candidates.to_csv(
    CANDIDATES_OUT,
    index=False,
)


supported.to_csv(
    SUPPORTED_OUT,
    index=False,
)


exploratory.to_csv(
    EXPLORATORY_OUT,
    index=False,
)


# ============================================================
# Diagnostics
# ============================================================

n_supported_A = int(
    (
        candidates[
            "candidate_track"
        ]
        == "SUPPORTED_A"
    ).sum()
)


n_supported_B = int(
    (
        candidates[
            "candidate_track"
        ]
        == "SUPPORTED_B_HIGH_SPREAD"
    ).sum()
)


n_exploratory = int(
    (
        candidates[
            "candidate_track"
        ]
        == "EXPLORATORY_OOD"
    ).sum()
)


n_unique_scaffolds = int(
    candidates[
        "murcko_scaffold"
    ]
    .nunique()
)


n_generic_scaffolds = int(
    candidates[
        "generic_murcko_scaffold"
    ]
    .nunique()
)


supported_unique_scaffolds = int(
    supported[
        "murcko_scaffold"
    ]
    .nunique()
)


exploratory_unique_scaffolds = int(
    exploratory[
        "murcko_scaffold"
    ]
    .nunique()
)


# ============================================================
# Report
# ============================================================

report = {
    "input_top20000":
        int(
            len(df)
        ),

    "selection_design": {
        "supported_track":
            (
                "Retain all strict similarity-supported "
                "Tier A and Tier B compounds."
            ),

        "exploration_track":
            (
                "Select highest-QSAR-ranked compounds "
                "outside strict support while allowing "
                "one compound per exact Bemis-Murcko "
                "scaffold."
            ),

        "exploration_target_n":
            N_EXPLORATORY,

        "docking_information_used_for_selection":
            False,
    },

    "candidate_counts": {
        "SUPPORTED_A":
            n_supported_A,

        "SUPPORTED_B_HIGH_SPREAD":
            n_supported_B,

        "EXPLORATORY_OOD":
            n_exploratory,

        "TOTAL":
            int(
                len(candidates)
            ),
    },

    "diversity": {
        "unique_exact_Murcko_scaffolds_total":
            n_unique_scaffolds,

        "unique_generic_Murcko_scaffolds_total":
            n_generic_scaffolds,

        "supported_unique_scaffolds":
            supported_unique_scaffolds,

        "exploratory_unique_scaffolds":
            exploratory_unique_scaffolds,
    },

    "candidate_prediction_range": {
        "min_pIC50":
            float(
                candidates[
                    "qsar_predicted_pIC50"
                ].min()
            ),

        "max_pIC50":
            float(
                candidates[
                    "qsar_predicted_pIC50"
                ].max()
            ),

        "min_training_similarity":
            float(
                candidates[
                    "max_training_tanimoto"
                ].min()
            ),

        "max_training_similarity":
            float(
                candidates[
                    "max_training_tanimoto"
                ].max()
            ),
    },

    "interpretation": {
        "SUPPORTED_A":
            (
                "QSAR-ranked candidate with comparatively "
                "strong training-space similarity support."
            ),

        "SUPPORTED_B_HIGH_SPREAD":
            (
                "Similarity-supported but with unusually "
                "high RF tree-to-tree dispersion."
            ),

        "EXPLORATORY_OOD":
            (
                "Hypothesis-generating candidate. "
                "QSAR potency should not be treated as "
                "reliable quantitative prediction."
            ),
    },

    "outputs": {
        "all_candidates":
            str(
                CANDIDATES_OUT
            ),

        "supported":
            str(
                SUPPORTED_OUT
            ),

        "exploratory":
            str(
                EXPLORATORY_OUT
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
# Terminal summary
# ============================================================

print(
    "\n========================================"
)

print(
    "PHASE 4C CANDIDATE SET"
)

print(
    "========================================"
)


print(
    "\nSUPPORTED_A:",
    n_supported_A
)

print(
    "SUPPORTED_B_HIGH_SPREAD:",
    n_supported_B
)

print(
    "EXPLORATORY_OOD:",
    n_exploratory
)

print(
    "TOTAL:",
    len(candidates)
)


print(
    "\nUnique exact Murcko scaffolds:",
    n_unique_scaffolds
)

print(
    "Unique generic Murcko scaffolds:",
    n_generic_scaffolds
)

print(
    "Supported unique scaffolds:",
    supported_unique_scaffolds
)

print(
    "Exploratory unique scaffolds:",
    exploratory_unique_scaffolds
)


print(
    "\n----------------------------------------"
)

print(
    "SUPPORTED CANDIDATES"
)

print(
    "----------------------------------------"
)


display_cols = [
    "phase4C_candidate_id",
    "candidate_track",
    "zinc_id",
    "qsar_predicted_pIC50",
    "max_training_tanimoto",
    "rf_tree_sd_pIC50",
    "murcko_scaffold",
]


print(
    candidates[
        candidates[
            "candidate_track"
        ]
        != "EXPLORATORY_OOD"
    ][
        display_cols
    ]
    .to_string(
        index=False
    )
)


print(
    "\n----------------------------------------"
)

print(
    "TOP 20 EXPLORATORY CANDIDATES"
)

print(
    "----------------------------------------"
)


print(
    candidates[
        candidates[
            "candidate_track"
        ]
        == "EXPLORATORY_OOD"
    ][
        display_cols
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    CANDIDATES_OUT
)

print(
    SUPPORTED_OUT
)

print(
    EXPLORATORY_OUT
)

print(
    REPORT_OUT
)


print(
    "\n========================================"
)

print(
    "PHASE 4C CANDIDATE SELECTION: PASS"
)

print(
    "========================================"
)
