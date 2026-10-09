from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT / "data" / "raw"
    / "chembl_egfr_ic50_raw.csv"
)

EXACT_FILE = (
    ROOT / "data" / "interim"
    / "egfr_ic50_binding_exact.csv"
)

ASSAY_FILE = (
    ROOT / "data" / "interim"
    / "egfr_assay_qc.csv"
)

ACTIVITY_FILE = (
    ROOT / "data" / "interim"
    / "egfr_wt_biochemical_ic50.csv"
)

STANDARDIZED_FILE = (
    ROOT / "data" / "interim"
    / "egfr_standardized_activities.csv"
)

DEDUP_FILE = (
    ROOT / "data" / "interim"
    / "egfr_activity_deduplicated.csv"
)

AGGREGATED_FILE = (
    ROOT / "data" / "processed"
    / "egfr_compound_activity_aggregated.csv"
)

MODELING_FILE = (
    ROOT / "data" / "processed"
    / "egfr_qsar_modeling.csv"
)

REPORT_DIR = ROOT / "reports"
FIGURE_DIR = ROOT / "figures"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Load datasets
# =========================================

raw = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

exact = pd.read_csv(
    EXACT_FILE,
    low_memory=False
)

assays = pd.read_csv(
    ASSAY_FILE,
    low_memory=False
)

activity = pd.read_csv(
    ACTIVITY_FILE,
    low_memory=False
)

standardized = pd.read_csv(
    STANDARDIZED_FILE,
    low_memory=False
)

dedup = pd.read_csv(
    DEDUP_FILE,
    low_memory=False
)

aggregated = pd.read_csv(
    AGGREGATED_FILE,
    low_memory=False
)

modeling = pd.read_csv(
    MODELING_FILE,
    low_memory=False
)


# =========================================
# Dataset funnel
# =========================================

core_assays = int(
    assays["keep_core_qsar"].sum()
)

high_variance = int(
    aggregated[
        "high_variance_flag"
    ].sum()
)

funnel = pd.DataFrame(
    [
        [
            "ChEMBL raw EGFR binding IC50",
            len(raw)
        ],
        [
            "Exact '=' IC50",
            len(exact)
        ],
        [
            "Strict core assays",
            core_assays
        ],
        [
            "Activity QC passed",
            len(activity)
        ],
        [
            "RDKit standardized activities",
            len(standardized)
        ],
        [
            "After duplicate-citation handling",
            len(dedup)
        ],
        [
            "Unique standardized compounds",
            len(aggregated)
        ],
        [
            "Final QSAR compounds",
            len(modeling)
        ],
    ],
    columns=[
        "stage",
        "count"
    ]
)

funnel.to_csv(
    REPORT_DIR
    / "phase1_dataset_funnel.csv",
    index=False
)


# =========================================
# Modeling dataset QC
# =========================================

summary = {
    "raw_activity_records":
        int(len(raw)),

    "exact_activity_records":
        int(len(exact)),

    "total_assays":
        int(len(assays)),

    "core_assays":
        core_assays,

    "activity_qc_records":
        int(len(activity)),

    "standardized_activity_records":
        int(len(standardized)),

    "deduplicated_activity_records":
        int(len(dedup)),

    "aggregated_compounds":
        int(len(aggregated)),

    "high_variance_compounds":
        high_variance,

    "final_qsar_compounds":
        int(len(modeling)),

    "duplicate_standardized_smiles":
        int(
            modeling[
                "standardized_smiles"
            ].duplicated().sum()
        ),

    "missing_smiles":
        int(
            modeling[
                "standardized_smiles"
            ].isna().sum()
        ),

    "missing_pic50":
        int(
            modeling[
                "pIC50"
            ].isna().sum()
        ),

    "pic50_mean":
        float(
            modeling[
                "pIC50"
            ].mean()
        ),

    "pic50_std":
        float(
            modeling[
                "pIC50"
            ].std()
        ),

    "pic50_min":
        float(
            modeling[
                "pIC50"
            ].min()
        ),

    "pic50_median":
        float(
            modeling[
                "pIC50"
            ].median()
        ),

    "pic50_max":
        float(
            modeling[
                "pIC50"
            ].max()
        ),
}


with open(
    REPORT_DIR
    / "phase1_qc_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# =========================================
# pIC50 distribution figure
# =========================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    modeling["pIC50"],
    bins=30
)

plt.xlabel("pIC50")
plt.ylabel("Number of compounds")

plt.title(
    "Final EGFR QSAR Dataset: "
    "pIC50 Distribution"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "phase1_pic50_distribution.png",
    dpi=300
)

plt.close()


# =========================================
# Dataset funnel figure
# =========================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    funnel["stage"],
    funnel["count"]
)

plt.ylabel("Count")

plt.title(
    "EGFR Dataset Curation Funnel"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "phase1_dataset_funnel.png",
    dpi=300
)

plt.close()


# =========================================
# Measurement-count summary
# =========================================

measurement_summary = (
    aggregated[
        "n_measurements"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "n_measurements"
    )
    .reset_index(
        name="n_compounds"
    )
)

measurement_summary.to_csv(
    REPORT_DIR
    / "phase1_measurement_counts.csv",
    index=False
)


# =========================================
# High-variance compounds
# =========================================

high_variance_df = (
    aggregated[
        aggregated[
            "high_variance_flag"
        ]
    ]
    .sort_values(
        "pIC50_range",
        ascending=False
    )
)

high_variance_df.to_csv(
    REPORT_DIR
    / "phase1_high_variance_compounds.csv",
    index=False
)


# =========================================
# Final print summary
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 1 FINAL QC"
)

print(
    "================================="
)

print(
    funnel.to_string(
        index=False
    )
)

print(
    "\nFinal QSAR compounds:",
    len(modeling)
)

print(
    "High-variance excluded:",
    high_variance
)

print(
    "Duplicate standardized SMILES:",
    summary[
        "duplicate_standardized_smiles"
    ]
)

print(
    "Missing SMILES:",
    summary[
        "missing_smiles"
    ]
)

print(
    "Missing pIC50:",
    summary[
        "missing_pic50"
    ]
)

print(
    "\npIC50:"
)

print(
    modeling[
        "pIC50"
    ].describe()
)

print(
    "\nGenerated reports:"
)

print(
    REPORT_DIR
    / "phase1_dataset_funnel.csv"
)

print(
    REPORT_DIR
    / "phase1_qc_summary.json"
)

print(
    REPORT_DIR
    / "phase1_measurement_counts.csv"
)

print(
    REPORT_DIR
    / "phase1_high_variance_compounds.csv"
)

print(
    "\nGenerated figures:"
)

print(
    FIGURE_DIR
    / "phase1_pic50_distribution.png"
)

print(
    FIGURE_DIR
    / "phase1_dataset_funnel.png"
)
