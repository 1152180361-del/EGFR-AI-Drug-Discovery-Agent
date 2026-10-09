from pathlib import Path
from collections import Counter
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_qsar_scaffolds.csv"
)

TRAIN_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_train.csv"
)

VALID_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_validation.csv"
)

TEST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_test.csv"
)

COMBINED_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_scaffold_split.csv"
)

SUMMARY_FILE = (
    ROOT
    / "reports"
    / "phase2_scaffold_split_summary.csv"
)

SCAFFOLD_ASSIGNMENT_FILE = (
    ROOT
    / "reports"
    / "phase2_scaffold_assignments.csv"
)

FIGURE_FILE = (
    ROOT
    / "figures"
    / "phase2_scaffold_split_pic50.png"
)


# =========================================
# Settings
# =========================================

RANDOM_SEED = 42

TRAIN_FRACTION = 0.80
VALID_FRACTION = 0.10
TEST_FRACTION = 0.10


# =========================================
# Load dataset
# =========================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    "Input compounds:",
    len(df)
)

print(
    "Unique split scaffolds:",
    df["split_scaffold_id"].nunique()
)


# =========================================
# Required columns
# =========================================

required_columns = [
    "standardized_smiles",
    "standardized_inchikey",
    "pIC50",
    "split_scaffold_id",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# =========================================
# Build scaffold groups
#
# No pandas groupby is used.
# =========================================

records = df.to_dict("records")

scaffold_groups = {}

for index, record in enumerate(records):

    scaffold_id = str(
        record["split_scaffold_id"]
    )

    if scaffold_id not in scaffold_groups:

        scaffold_groups[
            scaffold_id
        ] = []

    scaffold_groups[
        scaffold_id
    ].append(
        index
    )


# =========================================
# Prepare scaffold list
#
# Shuffle first, then stable-sort by size.
#
# This means scaffolds of equal size are
# ordered reproducibly but not
# alphabetically.
# =========================================

scaffold_items = list(
    scaffold_groups.items()
)

rng = random.Random(
    RANDOM_SEED
)

rng.shuffle(
    scaffold_items
)

scaffold_items.sort(
    key=lambda item: len(item[1]),
    reverse=True
)


# =========================================
# Target sizes
# =========================================

n_total = len(df)

target_train = round(
    n_total * TRAIN_FRACTION
)

target_valid = round(
    n_total * VALID_FRACTION
)

target_test = (
    n_total
    - target_train
    - target_valid
)


targets = {
    "train": target_train,
    "validation": target_valid,
    "test": target_test,
}


assignments = {
    "train": [],
    "validation": [],
    "test": [],
}


scaffold_assignments = []


# =========================================
# Balanced greedy scaffold allocation
#
# Rule:
#
# 1. Never split a scaffold.
# 2. Prefer the split with the lowest
#    current fractional occupancy.
# 3. Prefer splits where the entire scaffold
#    still fits within the target size.
# 4. Fixed seed gives reproducibility.
# =========================================

for scaffold_id, indices in scaffold_items:

    group_size = len(indices)

    current_sizes = {
        split_name:
            len(assignments[split_name])

        for split_name
        in assignments
    }


    # -------------------------------------
    # Find splits where scaffold fits
    # -------------------------------------

    fitting_splits = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        if (
            current_sizes[split_name]
            + group_size
            <= targets[split_name]
        ):

            fitting_splits.append(
                split_name
            )


    # -------------------------------------
    # If it fits somewhere:
    #
    # choose split with smallest fraction
    # of its target already occupied.
    # -------------------------------------

    if fitting_splits:

        chosen_split = min(
            fitting_splits,
            key=lambda split_name: (
                current_sizes[split_name]
                / targets[split_name]
            )
        )


    # -------------------------------------
    # If scaffold cannot fit perfectly
    # anywhere, choose split producing
    # smallest relative overflow.
    # -------------------------------------

    else:

        def overflow_score(split_name):

            future_size = (
                current_sizes[split_name]
                + group_size
            )

            overflow = max(
                0,
                future_size
                - targets[split_name]
            )

            return (
                overflow
                / targets[split_name]
            )


        chosen_split = min(
            [
                "train",
                "validation",
                "test",
            ],
            key=overflow_score
        )


    # -------------------------------------
    # Assign entire scaffold
    # -------------------------------------

    assignments[
        chosen_split
    ].extend(
        indices
    )


    scaffold_assignments.append(
        {
            "split_scaffold_id":
                scaffold_id,

            "scaffold_size":
                group_size,

            "split":
                chosen_split,
        }
    )


# =========================================
# Build split dataframes
# =========================================

train = df.iloc[
    sorted(
        assignments["train"]
    )
].copy()

valid = df.iloc[
    sorted(
        assignments["validation"]
    )
].copy()

test = df.iloc[
    sorted(
        assignments["test"]
    )
].copy()


train["split"] = "train"
valid["split"] = "validation"
test["split"] = "test"


# =========================================
# Combine dataset
# =========================================

combined = pd.concat(
    [
        train,
        valid,
        test,
    ],
    ignore_index=True
)


# =========================================
# Scaffold leakage checks
# =========================================

train_scaffolds = set(
    train[
        "split_scaffold_id"
    ].astype(str)
)

valid_scaffolds = set(
    valid[
        "split_scaffold_id"
    ].astype(str)
)

test_scaffolds = set(
    test[
        "split_scaffold_id"
    ].astype(str)
)


train_valid_overlap = (
    train_scaffolds
    & valid_scaffolds
)

train_test_overlap = (
    train_scaffolds
    & test_scaffolds
)

valid_test_overlap = (
    valid_scaffolds
    & test_scaffolds
)


# =========================================
# Compound overlap checks
# =========================================

train_molecules = set(
    train[
        "standardized_smiles"
    ]
)

valid_molecules = set(
    valid[
        "standardized_smiles"
    ]
)

test_molecules = set(
    test[
        "standardized_smiles"
    ]
)


compound_overlap = (
    len(
        train_molecules
        & valid_molecules
    )
    +
    len(
        train_molecules
        & test_molecules
    )
    +
    len(
        valid_molecules
        & test_molecules
    )
)


# =========================================
# Hard safety checks
# =========================================

if train_valid_overlap:

    raise RuntimeError(
        "Scaffold leakage between "
        "train and validation."
    )

if train_test_overlap:

    raise RuntimeError(
        "Scaffold leakage between "
        "train and test."
    )

if valid_test_overlap:

    raise RuntimeError(
        "Scaffold leakage between "
        "validation and test."
    )

if compound_overlap != 0:

    raise RuntimeError(
        "Compound leakage detected."
    )

if len(combined) != len(df):

    raise RuntimeError(
        "Compound count changed "
        "during split."
    )

if combined[
    "standardized_smiles"
].duplicated().any():

    raise RuntimeError(
        "Duplicate standardized "
        "SMILES detected."
    )


# =========================================
# Save datasets
# =========================================

train.to_csv(
    TRAIN_FILE,
    index=False
)

valid.to_csv(
    VALID_FILE,
    index=False
)

test.to_csv(
    TEST_FILE,
    index=False
)

combined.to_csv(
    COMBINED_FILE,
    index=False
)


# =========================================
# Save scaffold assignments
# =========================================

assignment_df = pd.DataFrame(
    scaffold_assignments
)

assignment_df.to_csv(
    SCAFFOLD_ASSIGNMENT_FILE,
    index=False
)


# =========================================
# Summary helper
# =========================================

def split_summary(
    name,
    subset
):

    return {
        "split":
            name,

        "n_compounds":
            len(subset),

        "fraction":
            len(subset)
            / n_total,

        "n_scaffolds":
            subset[
                "split_scaffold_id"
            ].nunique(),

        "pIC50_mean":
            subset[
                "pIC50"
            ].mean(),

        "pIC50_std":
            subset[
                "pIC50"
            ].std(),

        "pIC50_median":
            subset[
                "pIC50"
            ].median(),

        "pIC50_min":
            subset[
                "pIC50"
            ].min(),

        "pIC50_max":
            subset[
                "pIC50"
            ].max(),
    }


# =========================================
# Build summary table
# =========================================

summary = pd.DataFrame(
    [
        split_summary(
            "train",
            train
        ),

        split_summary(
            "validation",
            valid
        ),

        split_summary(
            "test",
            test
        ),
    ]
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# =========================================
# pIC50 distribution figure
# =========================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    train["pIC50"],
    bins=30,
    alpha=0.5,
    label="Train"
)

plt.hist(
    valid["pIC50"],
    bins=30,
    alpha=0.5,
    label="Validation"
)

plt.hist(
    test["pIC50"],
    bins=30,
    alpha=0.5,
    label="Test"
)

plt.xlabel(
    "pIC50"
)

plt.ylabel(
    "Number of compounds"
)

plt.title(
    "EGFR Scaffold Split: "
    "pIC50 Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_FILE,
    dpi=300
)

plt.close()


# =========================================
# Final reporting
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 2.2 SCAFFOLD SPLIT"
)

print(
    "================================="
)

print(
    "Random seed:",
    RANDOM_SEED
)

print(
    "\nTarget sizes:"
)

print(
    "Train:",
    target_train
)

print(
    "Validation:",
    target_valid
)

print(
    "Test:",
    target_test
)


print(
    "\nActual split:"
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\n--- Scaffold leakage ---"
)

print(
    "Train vs Validation:",
    len(
        train_valid_overlap
    )
)

print(
    "Train vs Test:",
    len(
        train_test_overlap
    )
)

print(
    "Validation vs Test:",
    len(
        valid_test_overlap
    )
)


print(
    "\nCompound overlap:",
    compound_overlap
)


# =========================================
# Scaffold-size distribution per split
# =========================================

assignment_counts = Counter(
    assignment_df["split"]
)

print(
    "\nScaffold groups per split:"
)

print(
    "Train:",
    assignment_counts[
        "train"
    ]
)

print(
    "Validation:",
    assignment_counts[
        "validation"
    ]
)

print(
    "Test:",
    assignment_counts[
        "test"
    ]
)


# =========================================
# Largest scaffold in each split
# =========================================

for split_name in [
    "train",
    "validation",
    "test",
]:

    subset = assignment_df[
        assignment_df[
            "split"
        ]
        == split_name
    ]

    print(
        f"Largest scaffold "
        f"in {split_name}:",
        int(
            subset[
                "scaffold_size"
            ].max()
        )
    )


# =========================================
# Final sanity checks
# =========================================

print(
    "\n--- Sanity checks ---"
)

print(
    "Total compounds after split:",
    len(combined)
)

print(
    "Expected compounds:",
    len(df)
)

print(
    "Missing split labels:",
    int(
        combined[
            "split"
        ].isna().sum()
    )
)

print(
    "Duplicate molecules:",
    int(
        combined[
            "standardized_smiles"
        ]
        .duplicated()
        .sum()
    )
)


print(
    "\nSaved Train:"
)

print(
    TRAIN_FILE
)

print(
    "\nSaved Validation:"
)

print(
    VALID_FILE
)

print(
    "\nSaved Test:"
)

print(
    TEST_FILE
)

print(
    "\nSaved combined split:"
)

print(
    COMBINED_FILE
)

print(
    "\nSaved split summary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nSaved scaffold assignments:"
)

print(
    SCAFFOLD_ASSIGNMENT_FILE
)

print(
    "\nSaved figure:"
)

print(
    FIGURE_FILE
)
