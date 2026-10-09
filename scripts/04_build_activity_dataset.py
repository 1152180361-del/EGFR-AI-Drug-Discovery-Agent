from pathlib import Path
import numpy as np
import pandas as pd


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

ACTIVITY_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_ic50_binding_exact.csv"
)

ASSAY_QC_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_assay_qc.csv"
)

AUDIT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_activity_qc.csv"
)

CORE_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_wt_biochemical_ic50.csv"
)


# =========================================
# Load data
# =========================================

activities = pd.read_csv(
    ACTIVITY_FILE,
    low_memory=False
)

assays = pd.read_csv(
    ASSAY_QC_FILE,
    low_memory=False
)

print("Activity rows:", len(activities))
print("Assay QC rows:", len(assays))


# =========================================
# Keep required assay QC columns
# =========================================

assay_cols = [
    "assay_chembl_id",
    "confidence_score",
    "bao_label",
    "is_confidence_9",
    "is_variant",
    "is_cellular",
    "is_single_protein",
    "is_biochemical_candidate",
    "is_wt_candidate",
    "keep_core_qsar",
]

assay_cols = [
    col
    for col in assay_cols
    if col in assays.columns
]

assay_qc = assays[
    assay_cols
].copy()


# =========================================
# Merge activity data with assay QC
# =========================================

df = activities.merge(
    assay_qc,
    on="assay_chembl_id",
    how="left",
    validate="many_to_one"
)

print(
    "Rows after assay merge:",
    len(df)
)


# =========================================
# 1. Exact relation QC
# =========================================

df["qc_exact_relation"] = (
    df["standard_relation"]
    .astype("string")
    .eq("=")
)


# =========================================
# 2. Unit QC
# =========================================

df["qc_unit_nm"] = (
    df["standard_units"]
    .astype("string")
    .eq("nM")
)


# =========================================
# 3. Activity value QC
# =========================================

df["standard_value_numeric"] = (
    pd.to_numeric(
        df["standard_value"],
        errors="coerce"
    )
)

df["qc_value_numeric"] = (
    df["standard_value_numeric"]
    .notna()
)

df["qc_value_positive"] = (
    df["standard_value_numeric"]
    > 0
)


# =========================================
# 4. ChEMBL data validity QC
# =========================================

if "data_validity_comment" in df.columns:

    validity = (
        df["data_validity_comment"]
        .astype("string")
    )

    df["qc_data_validity"] = (
        validity.isna()
        | validity.eq(
            "Manually validated"
        )
    )

else:

    print(
        "WARNING: data_validity_comment "
        "column not found."
    )

    df["qc_data_validity"] = True


# =========================================
# 5. Molecular structure QC
# =========================================

if "canonical_smiles" in df.columns:

    smiles = (
        df["canonical_smiles"]
        .astype("string")
    )

    df["qc_has_structure"] = (
        smiles.notna()
        & smiles.str.strip().ne("")
    )

else:

    print(
        "WARNING: canonical_smiles "
        "column not found."
    )

    df["qc_has_structure"] = False


# =========================================
# 6. Potential duplicate field
# =========================================

if "potential_duplicate" in df.columns:

    df["potential_duplicate"] = (
        pd.to_numeric(
            df["potential_duplicate"],
            errors="coerce"
        )
    )

else:

    df["potential_duplicate"] = np.nan


# =========================================
# 7. Assay QC safety
# =========================================

df["keep_core_qsar"] = (
    df["keep_core_qsar"]
    .fillna(False)
    .astype(bool)
)


# =========================================
# 8. Final activity-level QC
# IMPORTANT:
# Molecular structure is now REQUIRED
# =========================================

df["keep_activity_core"] = (
    df["keep_core_qsar"]
    & df["qc_exact_relation"]
    & df["qc_unit_nm"]
    & df["qc_value_numeric"]
    & df["qc_value_positive"]
    & df["qc_data_validity"]
    & df["qc_has_structure"]
)


# =========================================
# 9. Calculate pIC50
#
# pIC50 = 9 - log10(IC50_nM)
# =========================================

df["pIC50"] = np.nan

mask = df["keep_activity_core"]

df.loc[
    mask,
    "pIC50"
] = (
    9
    - np.log10(
        df.loc[
            mask,
            "standard_value_numeric"
        ]
    )
)


# =========================================
# 10. Save complete audit table
# =========================================

df.to_csv(
    AUDIT_FILE,
    index=False
)


# =========================================
# 11. Generate core activity dataset
# =========================================

core = df[
    df["keep_activity_core"]
].copy()

core.to_csv(
    CORE_FILE,
    index=False
)


# =========================================
# 12. Main QC summary
# =========================================

print("\n=================================")
print("EGFR activity QC summary")
print("=================================")

print(
    "Input activity rows:",
    len(df)
)

print(
    "Rows from keep_core_qsar assays:",
    df["keep_core_qsar"].sum()
)

print(
    "Exact relation:",
    df["qc_exact_relation"].sum()
)

print(
    "nM:",
    df["qc_unit_nm"].sum()
)

print(
    "Numeric values:",
    df["qc_value_numeric"].sum()
)

print(
    "Positive values:",
    df["qc_value_positive"].sum()
)

print(
    "Validity OK:",
    df["qc_data_validity"].sum()
)

print(
    "Has canonical SMILES:",
    df["qc_has_structure"].sum()
)

print(
    "\nFINAL activity records:",
    len(core)
)

print(
    "Unique compounds:",
    core[
        "molecule_chembl_id"
    ].nunique()
)

print(
    "Unique assays:",
    core[
        "assay_chembl_id"
    ].nunique()
)


# =========================================
# 13. Confirm no missing structures remain
# =========================================

missing_smiles = (
    core["canonical_smiles"]
    .isna()
    |
    core["canonical_smiles"]
    .astype("string")
    .str.strip()
    .eq("")
)

print(
    "Missing canonical SMILES "
    "in FINAL dataset:",
    missing_smiles.sum()
)


# =========================================
# 14. pIC50 distribution
# =========================================

if len(core) > 0:

    print(
        "\n--- pIC50 distribution ---"
    )

    print(
        core["pIC50"]
        .describe()
    )


# =========================================
# 15. Data validity comments
# =========================================

if "data_validity_comment" in df.columns:

    print(
        "\n--- Data validity comments ---"
    )

    print(
        df[
            "data_validity_comment"
        ]
        .value_counts(
            dropna=False
        )
        .head(20)
    )


# =========================================
# 16. Potential duplicate summary
# =========================================

print(
    "\nPotential duplicate "
    "flagged records:",
    (
        core[
            "potential_duplicate"
        ]
        == 1
    ).sum()
)


# =========================================
# 17. Compare pIC50 with ChEMBL pChEMBL
# =========================================

if "pchembl_value" in core.columns:

    core["pchembl_numeric"] = (
        pd.to_numeric(
            core[
                "pchembl_value"
            ],
            errors="coerce"
        )
    )

    compare = core[
        core[
            "pchembl_numeric"
        ].notna()
    ].copy()

    if len(compare) > 0:

        compare[
            "abs_difference"
        ] = (
            compare[
                "pIC50"
            ]
            -
            compare[
                "pchembl_numeric"
            ]
        ).abs()

        print(
            "\nRecords with "
            "ChEMBL pChEMBL:",
            len(compare)
        )

        print(
            "Mean |pIC50 - pChEMBL|:",
            compare[
                "abs_difference"
            ].mean()
        )

        print(
            "Max |pIC50 - pChEMBL|:",
            compare[
                "abs_difference"
            ].max()
        )


# =========================================
# 18. Output paths
# =========================================

print(
    "\nSaved activity audit table:"
)

print(
    AUDIT_FILE
)

print(
    "\nSaved WT biochemical IC50:"
)

print(
    CORE_FILE
)
