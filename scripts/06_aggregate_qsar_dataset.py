from pathlib import Path
import numpy as np
import pandas as pd


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
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

AGGREGATED_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Load data
# =========================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Input activity records:", len(df))
print(
    "Input standardized structures:",
    df["standardized_smiles"].nunique()
)


# =========================================
# Required columns
# =========================================

required_columns = [
    "standardized_smiles",
    "standardized_inchikey",
    "molecule_chembl_id",
    "assay_chembl_id",
    "pIC50",
    "standard_value_numeric",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# =========================================
# Numeric conversion
# =========================================

df["pIC50"] = pd.to_numeric(
    df["pIC50"],
    errors="coerce"
)

df["standard_value_numeric"] = pd.to_numeric(
    df["standard_value_numeric"],
    errors="coerce"
)


# =========================================
# Potential duplicate flag
# =========================================

if "potential_duplicate" in df.columns:

    df["potential_duplicate_numeric"] = pd.to_numeric(
        df["potential_duplicate"],
        errors="coerce"
    )

else:

    df["potential_duplicate_numeric"] = np.nan


df["is_potential_duplicate"] = (
    df["potential_duplicate_numeric"]
    .fillna(0)
    .eq(1)
)

print(
    "Potential duplicate flagged records:",
    int(df["is_potential_duplicate"].sum())
)


# =========================================
# Convert dataframe to Python records
#
# From this point onward:
# NO pandas groupby is used.
# =========================================

records = df.to_dict("records")


# =========================================
# Track duplicate history by structure
# =========================================

duplicate_history = {}

for record in records:

    smiles = record["standardized_smiles"]

    if smiles not in duplicate_history:
        duplicate_history[smiles] = False

    if bool(record["is_potential_duplicate"]):
        duplicate_history[smiles] = True


# =========================================
# Build activity duplicate groups
#
# Key =
# standardized structure + exact IC50 value
# =========================================

activity_groups = {}

for record in records:

    smiles = record["standardized_smiles"]
    value = record["standard_value_numeric"]

    key = (
        str(smiles),
        float(value)
    )

    if key not in activity_groups:
        activity_groups[key] = []

    activity_groups[key].append(record)


# =========================================
# Remove likely repeated citations
# =========================================

deduplicated_records = []

removed_duplicate_citations = 0
flagged_only_groups = 0


for group_records in activity_groups.values():

    normal_records = [
        r for r in group_records
        if not bool(r["is_potential_duplicate"])
    ]

    flagged_records = [
        r for r in group_records
        if bool(r["is_potential_duplicate"])
    ]

    if len(normal_records) > 0:

        deduplicated_records.extend(
            normal_records
        )

        removed_duplicate_citations += (
            len(flagged_records)
        )

    else:

        # Every record in this group is
        # flagged as potential duplicate.
        # Keep one representative.
        deduplicated_records.append(
            group_records[0]
        )

        flagged_only_groups += 1

        removed_duplicate_citations += (
            len(group_records) - 1
        )


# =========================================
# Save deduplicated activity table
# =========================================

dedup = pd.DataFrame(
    deduplicated_records
).reset_index(drop=True)

dedup.to_csv(
    DEDUP_FILE,
    index=False
)


print(
    "Records removed as repeat citations:",
    removed_duplicate_citations
)

print(
    "Flagged-only groups retaining "
    "one representative:",
    flagged_only_groups
)

print(
    "Activity records after "
    "duplicate handling:",
    len(dedup)
)


# =========================================
# Helper functions
# =========================================

def is_valid(value):

    if value is None:
        return False

    try:
        return not pd.isna(value)
    except Exception:
        return True


def unique_join(values):

    cleaned = set()

    for value in values:

        if is_valid(value):

            text = str(value).strip()

            if text:
                cleaned.add(text)

    return ";".join(
        sorted(cleaned)
    )


# =========================================
# Build compound groups using dictionary
#
# Again:
# NO pandas groupby.
# =========================================

compound_groups = {}

for record in deduplicated_records:

    smiles = str(
        record["standardized_smiles"]
    )

    if smiles not in compound_groups:
        compound_groups[smiles] = []

    compound_groups[smiles].append(
        record
    )


print(
    "\nCompound groups identified:",
    len(compound_groups)
)


# =========================================
# Aggregate activity per structure
# =========================================

compound_records = []


for smiles, group_records in compound_groups.items():

    p_values = []
    assay_ids = []
    chembl_ids = []
    document_ids = []
    inchikeys = []


    for record in group_records:

        value = record.get("pIC50")

        if is_valid(value):
            p_values.append(
                float(value)
            )


        assay_id = record.get(
            "assay_chembl_id"
        )

        if is_valid(assay_id):
            assay_ids.append(
                str(assay_id)
            )


        chembl_id = record.get(
            "molecule_chembl_id"
        )

        if is_valid(chembl_id):
            chembl_ids.append(
                str(chembl_id)
            )


        document_id = record.get(
            "document_chembl_id"
        )

        if is_valid(document_id):
            document_ids.append(
                str(document_id)
            )


        inchikey = record.get(
            "standardized_inchikey"
        )

        if is_valid(inchikey):
            inchikeys.append(
                str(inchikey)
            )


    if len(p_values) == 0:
        continue


    p_array = np.array(
        p_values,
        dtype=float
    )


    p_median = float(
        np.median(p_array)
    )

    p_mean = float(
        np.mean(p_array)
    )

    p_min = float(
        np.min(p_array)
    )

    p_max = float(
        np.max(p_array)
    )

    p_range = (
        p_max - p_min
    )


    if len(p_array) > 1:

        p_std = float(
            np.std(
                p_array,
                ddof=1
            )
        )

    else:

        p_std = np.nan


    if inchikeys:

        representative_inchikey = sorted(
            set(inchikeys)
        )[0]

    else:

        representative_inchikey = None


    compound_record = {

        "standardized_smiles":
            smiles,

        "standardized_inchikey":
            representative_inchikey,

        # Main QSAR target
        "pIC50":
            p_median,

        "pIC50_mean":
            p_mean,

        "pIC50_std":
            p_std,

        "pIC50_min":
            p_min,

        "pIC50_max":
            p_max,

        "pIC50_range":
            p_range,

        "n_measurements":
            len(p_values),

        "n_assays":
            len(set(assay_ids)),

        "n_chembl_ids":
            len(set(chembl_ids)),

        "n_documents":
            len(set(document_ids)),

        "molecule_chembl_ids":
            unique_join(chembl_ids),

        "assay_chembl_ids":
            unique_join(assay_ids),

        "document_chembl_ids":
            unique_join(document_ids),

        "had_potential_duplicate":
            bool(
                duplicate_history.get(
                    smiles,
                    False
                )
            ),
    }


    compound_records.append(
        compound_record
    )


# =========================================
# Build compound dataframe
# =========================================

aggregated = pd.DataFrame(
    compound_records
)


# =========================================
# Experimental consistency
#
# >1 pIC50 unit ≈ >10-fold IC50 spread
# =========================================

aggregated["high_variance_flag"] = (
    aggregated["pIC50_range"] > 1.0
)

aggregated["single_measurement"] = (
    aggregated["n_measurements"] == 1
)

aggregated["multiple_measurements"] = (
    aggregated["n_measurements"] > 1
)


# =========================================
# Sort and save complete compound dataset
# =========================================

aggregated = aggregated.sort_values(
    "pIC50",
    ascending=False
).reset_index(drop=True)

aggregated.to_csv(
    AGGREGATED_FILE,
    index=False
)


# =========================================
# Strict QSAR dataset
#
# Exclude high experimental disagreement
# =========================================

modeling = aggregated[
    ~aggregated[
        "high_variance_flag"
    ]
].copy()

modeling = modeling.reset_index(
    drop=True
)

modeling.to_csv(
    MODELING_FILE,
    index=False
)


# =========================================
# Summary
# =========================================

print("\n=================================")
print("Compound aggregation summary")
print("=================================")

print(
    "Input standardized activities:",
    len(df)
)

print(
    "Potential duplicate flagged:",
    int(
        df[
            "is_potential_duplicate"
        ].sum()
    )
)

print(
    "Removed repeat citations:",
    removed_duplicate_citations
)

print(
    "Flagged-only groups retaining "
    "one representative:",
    flagged_only_groups
)

print(
    "Activities after duplicate handling:",
    len(dedup)
)

print(
    "\nStandardized compounds aggregated:",
    len(aggregated)
)

print(
    "Single-measurement compounds:",
    int(
        aggregated[
            "single_measurement"
        ].sum()
    )
)

print(
    "Multiple-measurement compounds:",
    int(
        aggregated[
            "multiple_measurements"
        ].sum()
    )
)

print(
    "High-variance compounds "
    "(pIC50 range > 1):",
    int(
        aggregated[
            "high_variance_flag"
        ].sum()
    )
)

print(
    "\nFINAL QSAR modeling compounds:",
    len(modeling)
)


# =========================================
# pIC50 distribution
# =========================================

print(
    "\n--- Final pIC50 distribution ---"
)

print(
    modeling[
        "pIC50"
    ].describe()
)


# =========================================
# Measurement-count distribution
# =========================================

print(
    "\n--- Measurements per compound ---"
)

print(
    aggregated[
        "n_measurements"
    ]
    .value_counts()
    .sort_index()
    .head(20)
)


# =========================================
# Highest-variance compounds
# =========================================

high_variance = aggregated[
    aggregated[
        "high_variance_flag"
    ]
].copy()

high_variance = high_variance.sort_values(
    "pIC50_range",
    ascending=False
)


if len(high_variance) > 0:

    print(
        "\n--- Highest-variance compounds ---"
    )

    print(
        high_variance[
            [
                "standardized_inchikey",
                "pIC50",
                "pIC50_min",
                "pIC50_max",
                "pIC50_range",
                "n_measurements",
                "n_assays",
                "molecule_chembl_ids",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


# =========================================
# Sanity checks
# =========================================

print(
    "\n--- Sanity checks ---"
)

print(
    "Duplicate standardized SMILES "
    "in modeling dataset:",
    int(
        modeling[
            "standardized_smiles"
        ].duplicated().sum()
    )
)

print(
    "Missing pIC50:",
    int(
        modeling[
            "pIC50"
        ].isna().sum()
    )
)

print(
    "Missing standardized SMILES:",
    int(
        modeling[
            "standardized_smiles"
        ].isna().sum()
    )
)


# =========================================
# Save paths
# =========================================

print(
    "\nSaved deduplicated activities:"
)
print(DEDUP_FILE)

print(
    "\nSaved complete compound dataset:"
)
print(AGGREGATED_FILE)

print(
    "\nSaved FINAL QSAR modeling dataset:"
)
print(MODELING_FILE)
