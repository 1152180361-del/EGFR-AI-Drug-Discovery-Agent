from pathlib import Path
import json

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

INPUT = (
    ROOT
    / "data/processed/egfr_qsar_modeling.csv"
)

OUT_DIR = (
    ROOT
    / "docking/3W32/benchmark"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = (
    OUT_DIR
    / "egfr_retrospective_benchmark.csv"
)

REPORT_JSON = (
    OUT_DIR
    / "benchmark_selection_report.json"
)


# ============================================================
# Settings
# ============================================================

ACTIVE_THRESHOLD = 8.0
LOW_THRESHOLD = 6.0

N_ACTIVE = 50
N_LOW = 50

SEED = 42

rng = np.random.default_rng(
    SEED
)


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(
    INPUT
)

print(
    "Input rows:",
    len(df)
)

print(
    "Columns:"
)

print(
    list(df.columns)
)


# ============================================================
# Detect columns robustly
# ============================================================

def detect_column(
    dataframe,
    candidates,
):

    for col in candidates:

        if col in dataframe.columns:
            return col

    lower_map = {
        c.lower(): c
        for c in dataframe.columns
    }

    for col in candidates:

        if col.lower() in lower_map:
            return lower_map[
                col.lower()
            ]

    return None


smiles_col = detect_column(
    df,
    [
        "standardized_smiles",
        "canonical_smiles",
        "smiles",
        "standard_smiles",
    ],
)


pic50_col = detect_column(
    df,
    [
        "pIC50",
        "pic50",
        "median_pIC50",
        "median_pic50",
    ],
)


id_col = detect_column(
    df,
    [
        "molecule_chembl_id",
        "chembl_id",
        "compound_id",
        "representative_molecule_chembl_id",
    ],
)


if smiles_col is None:

    raise RuntimeError(
        "Could not identify SMILES column."
    )


if pic50_col is None:

    raise RuntimeError(
        "Could not identify pIC50 column."
    )


print(
    "\nDetected SMILES column:",
    smiles_col
)

print(
    "Detected pIC50 column:",
    pic50_col
)

print(
    "Detected ID column:",
    id_col
)


# ============================================================
# Molecular feature calculation
# ============================================================

def molecular_features(
    smiles
):

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        return None

    canonical = Chem.MolToSmiles(
        mol,
        isomericSmiles=True,
    )

    scaffold = (
        MurckoScaffold
        .MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
    )

    # Avoid putting all acyclic molecules
    # into one artificial scaffold group.
    if not scaffold:

        scaffold = (
            "ACYCLIC::"
            + canonical
        )

    return {
        "canonical_smiles":
            canonical,

        "scaffold":
            scaffold,

        "MolWt":
            float(
                Descriptors.MolWt(
                    mol
                )
            ),

        "LogP":
            float(
                Crippen.MolLogP(
                    mol
                )
            ),

        "TPSA":
            float(
                Descriptors.TPSA(
                    mol
                )
            ),

        "HBD":
            int(
                Lipinski.NumHDonors(
                    mol
                )
            ),

        "HBA":
            int(
                Lipinski.NumHAcceptors(
                    mol
                )
            ),

        "RotB":
            int(
                Lipinski.NumRotatableBonds(
                    mol
                )
            ),

        "HeavyAtoms":
            int(
                mol.GetNumHeavyAtoms()
            ),

        "FormalCharge":
            int(
                Chem.GetFormalCharge(
                    mol
                )
            ),
    }


records = []

for idx, row in df.iterrows():

    smiles = row[
        smiles_col
    ]

    try:

        pic50 = float(
            row[pic50_col]
        )

    except Exception:

        continue

    if pd.isna(smiles):

        continue

    features = molecular_features(
        str(smiles)
    )

    if features is None:

        continue

    compound_id = (
        str(row[id_col])
        if id_col is not None
        else f"EGFR_{idx:05d}"
    )

    record = {
        "compound_id":
            compound_id,

        "pIC50":
            pic50,
    }

    record.update(
        features
    )

    records.append(
        record
    )


work = pd.DataFrame(
    records
)


print(
    "\nValid molecules:",
    len(work)
)


# ============================================================
# Define strong actives / low actives
# ============================================================

active_pool = (
    work[
        work["pIC50"]
        >= ACTIVE_THRESHOLD
    ]
    .copy()
)


low_pool = (
    work[
        work["pIC50"]
        <= LOW_THRESHOLD
    ]
    .copy()
)


print(
    "\nActive pool "
    f"(pIC50 >= {ACTIVE_THRESHOLD}):",
    len(active_pool)
)

print(
    "Low-active pool "
    f"(pIC50 <= {LOW_THRESHOLD}):",
    len(low_pool)
)

print(
    "Unique active scaffolds:",
    active_pool[
        "scaffold"
    ].nunique()
)

print(
    "Unique low-active scaffolds:",
    low_pool[
        "scaffold"
    ].nunique()
)


# ============================================================
# Select scaffold-diverse strong actives
#
# One representative per scaffold first.
# Prefer stronger pIC50 within scaffold.
# ============================================================

active_pool = (
    active_pool
    .sort_values(
        "pIC50",
        ascending=False,
    )
)


active_selected = (
    active_pool
    .drop_duplicates(
        subset=[
            "scaffold"
        ],
        keep="first",
    )
    .head(
        N_ACTIVE
    )
    .copy()
)


if len(active_selected) < N_ACTIVE:

    raise RuntimeError(
        "Not enough unique active scaffolds "
        f"for N_ACTIVE={N_ACTIVE}."
    )


active_scaffolds = set(
    active_selected[
        "scaffold"
    ]
)


# ============================================================
# Property-match low actives
#
# Exclude scaffolds used by active set.
# Enforce unique low-active scaffolds.
#
# Matching variables:
# MolWt, LogP, TPSA, HBD, HBA, RotB
# ============================================================

low_candidates = (
    low_pool[
        ~low_pool[
            "scaffold"
        ].isin(
            active_scaffolds
        )
    ]
    .copy()
)


descriptor_cols = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
]


combined_for_scale = pd.concat(
    [
        active_selected[
            descriptor_cols
        ],
        low_candidates[
            descriptor_cols
        ],
    ],
    axis=0,
)


means = (
    combined_for_scale
    .mean()
)


stds = (
    combined_for_scale
    .std()
    .replace(
        0,
        1.0,
    )
)


active_scaled = (
    (
        active_selected[
            descriptor_cols
        ]
        - means
    )
    / stds
)


low_scaled = (
    (
        low_candidates[
            descriptor_cols
        ]
        - means
    )
    / stds
)


used_low_indices = set()
used_low_scaffolds = set()

matched_low_indices = []


for active_position, (
    active_idx,
    active_row
) in enumerate(
    active_selected.iterrows()
):

    active_vector = (
        active_scaled
        .loc[
            active_idx
        ]
        .to_numpy(
            dtype=float
        )
    )

    best_index = None
    best_distance = None

    for low_idx, low_row in (
        low_candidates
        .iterrows()
    ):

        if (
            low_idx
            in used_low_indices
        ):
            continue

        scaffold = (
            low_row[
                "scaffold"
            ]
        )

        if (
            scaffold
            in used_low_scaffolds
        ):
            continue

        low_vector = (
            low_scaled
            .loc[
                low_idx
            ]
            .to_numpy(
                dtype=float
            )
        )

        distance = float(
            np.linalg.norm(
                active_vector
                - low_vector
            )
        )

        if (
            best_distance is None
            or distance < best_distance
        ):

            best_distance = (
                distance
            )

            best_index = (
                low_idx
            )


    if best_index is None:

        raise RuntimeError(
            "Could not find enough "
            "property-matched low actives."
        )


    matched_low_indices.append(
        best_index
    )

    used_low_indices.add(
        best_index
    )

    used_low_scaffolds.add(
        low_candidates
        .loc[
            best_index,
            "scaffold"
        ]
    )


low_selected = (
    low_candidates
    .loc[
        matched_low_indices
    ]
    .copy()
)


# ============================================================
# Label benchmark
# ============================================================

active_selected[
    "benchmark_label"
] = "ACTIVE"


active_selected[
    "binary_label"
] = 1


low_selected[
    "benchmark_label"
] = "LOW_ACTIVE"


low_selected[
    "binary_label"
] = 0


benchmark = pd.concat(
    [
        active_selected,
        low_selected,
    ],
    ignore_index=True,
)


# Randomize final row order reproducibly
benchmark = (
    benchmark
    .sample(
        frac=1.0,
        random_state=SEED,
    )
    .reset_index(
        drop=True
    )
)


benchmark[
    "benchmark_id"
] = [
    f"EGFR_BENCH_{i+1:03d}"
    for i in range(
        len(benchmark)
    )
]


# ============================================================
# QC
# ============================================================

active_final = (
    benchmark[
        benchmark[
            "binary_label"
        ] == 1
    ]
)


low_final = (
    benchmark[
        benchmark[
            "binary_label"
        ] == 0
    ]
)


scaffold_overlap = (
    set(
        active_final[
            "scaffold"
        ]
    )
    &
    set(
        low_final[
            "scaffold"
        ]
    )
)


smiles_overlap = (
    set(
        active_final[
            "canonical_smiles"
        ]
    )
    &
    set(
        low_final[
            "canonical_smiles"
        ]
    )
)


if scaffold_overlap:

    raise RuntimeError(
        "Active/low-active scaffold overlap detected."
    )


if smiles_overlap:

    raise RuntimeError(
        "Active/low-active molecular overlap detected."
    )


# ============================================================
# Property balance summary
# ============================================================

property_summary = {}

for col in descriptor_cols:

    a_mean = float(
        active_final[
            col
        ].mean()
    )

    l_mean = float(
        low_final[
            col
        ].mean()
    )

    pooled_std = float(
        benchmark[
            col
        ].std()
    )

    if pooled_std == 0:

        smd = 0.0

    else:

        smd = (
            a_mean - l_mean
        ) / pooled_std

    property_summary[
        col
    ] = {
        "active_mean":
            a_mean,

        "low_active_mean":
            l_mean,

        "standardized_mean_difference":
            float(
                smd
            ),
    }


# ============================================================
# Save
# ============================================================

column_order = [
    "benchmark_id",
    "compound_id",
    "benchmark_label",
    "binary_label",
    "pIC50",
    "canonical_smiles",
    "scaffold",
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "HeavyAtoms",
    "FormalCharge",
]


benchmark[
    column_order
].to_csv(
    OUT_CSV,
    index=False,
)


report = {
    "input_dataset":
        str(INPUT),

    "input_rows":
        int(
            len(df)
        ),

    "active_threshold_pIC50":
        ACTIVE_THRESHOLD,

    "low_active_threshold_pIC50":
        LOW_THRESHOLD,

    "active_threshold_interpretation":
        "IC50 <= 10 nM",

    "low_active_threshold_interpretation":
        "IC50 >= 1000 nM",

    "active_pool_size":
        int(
            len(active_pool)
        ),

    "low_active_pool_size":
        int(
            len(low_pool)
        ),

    "selected_actives":
        int(
            len(active_final)
        ),

    "selected_low_actives":
        int(
            len(low_final)
        ),

    "active_unique_scaffolds":
        int(
            active_final[
                "scaffold"
            ].nunique()
        ),

    "low_active_unique_scaffolds":
        int(
            low_final[
                "scaffold"
            ].nunique()
        ),

    "cross_class_scaffold_overlap":
        int(
            len(
                scaffold_overlap
            )
        ),

    "cross_class_smiles_overlap":
        int(
            len(
                smiles_overlap
            )
        ),

    "property_summary":
        property_summary,

    "output":
        str(
            OUT_CSV
        ),
}


REPORT_JSON.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


# ============================================================
# Terminal output
# ============================================================

print(
    "\n========================================"
)

print(
    "EGFR RETROSPECTIVE DOCKING BENCHMARK"
)

print(
    "========================================"
)


print(
    "\nSelected actives:",
    len(active_final)
)

print(
    "Selected low-actives:",
    len(low_final)
)

print(
    "Active unique scaffolds:",
    active_final[
        "scaffold"
    ].nunique()
)

print(
    "Low-active unique scaffolds:",
    low_final[
        "scaffold"
    ].nunique()
)

print(
    "Cross-class scaffold overlap:",
    len(scaffold_overlap)
)

print(
    "Cross-class molecule overlap:",
    len(smiles_overlap)
)


print(
    "\nActivity distributions:"
)

print(
    "Active pIC50 mean:",
    round(
        active_final[
            "pIC50"
        ].mean(),
        3,
    )
)

print(
    "Active pIC50 range:",
    round(
        active_final[
            "pIC50"
        ].min(),
        3,
    ),
    "-",
    round(
        active_final[
            "pIC50"
        ].max(),
        3,
    )
)

print(
    "Low-active pIC50 mean:",
    round(
        low_final[
            "pIC50"
        ].mean(),
        3,
    )
)

print(
    "Low-active pIC50 range:",
    round(
        low_final[
            "pIC50"
        ].min(),
        3,
    ),
    "-",
    round(
        low_final[
            "pIC50"
        ].max(),
        3,
    )
)


print(
    "\nProperty balance:"
)

for col, values in (
    property_summary.items()
):

    print(
        f"{col:6s}  "
        f"ACTIVE={values['active_mean']:.3f}  "
        f"LOW={values['low_active_mean']:.3f}  "
        f"SMD={values['standardized_mean_difference']:+.3f}"
    )


print(
    "\nSaved benchmark:"
)

print(
    OUT_CSV
)

print(
    "\nSaved report:"
)

print(
    REPORT_JSON
)


print(
    "\n========================================"
)

print(
    "BENCHMARK BUILD: PASS"
)

print(
    "========================================"
)
