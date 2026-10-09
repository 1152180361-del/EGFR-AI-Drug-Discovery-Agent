from pathlib import Path
import re

import pandas as pd


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_assay_metadata.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_assay_qc.csv"
)


# =========================================
# Read assay metadata
# =========================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Input assays:", len(df))


# =========================================
# Helper text column
# =========================================

df["description_text"] = (
    df["description"]
    .fillna("")
    .astype(str)
    .str.lower()
)


# =========================================
# 1. Confidence flag
# =========================================

df["is_confidence_9"] = (
    df["confidence_score"] == 9
)


# =========================================
# 2. Variant flag
# =========================================

variant_patterns = [
    r"\bl858r\b",
    r"\bt790m\b",
    r"\bc797s\b",
    r"\bg719[a-z]*\b",
    r"\bl861q\b",
    r"\bs768i\b",
    r"\bexon\s*19\b",
    r"\bdel19\b",
    r"\bmutant\b",
    r"\bmutation\b",
    r"\bvariant\b",
]

variant_regex = re.compile(
    "|".join(variant_patterns),
    flags=re.IGNORECASE
)

df["variant_by_sequence"] = (
    df["variant_sequence"].notna()
)

df["variant_by_description"] = (
    df["description_text"]
    .str.contains(
        variant_regex,
        regex=True,
        na=False
    )
)

df["is_variant"] = (
    df["variant_by_sequence"]
    | df["variant_by_description"]
)


# =========================================
# 3. Cellular flag
# =========================================

cell_patterns = [
    r"\bcell\b",
    r"\bcells\b",
    r"\bcellular\b",
    r"\ba431\b",
    r"\bh1975\b",
    r"\bhcc827\b",
    r"\bpc9\b",
    r"\bba/f3\b",
    r"\bba-f3\b",
]

cell_regex = re.compile(
    "|".join(cell_patterns),
    flags=re.IGNORECASE
)

df["cell_by_bao"] = (
    df["bao_label"]
    .fillna("")
    .str.lower()
    .isin([
        "cell-based format",
        "cell membrane format",
        "organism-based format"
    ])
)

df["cell_by_metadata"] = (
    df["assay_cell_type"].notna()
)

df["cell_by_description"] = (
    df["description_text"]
    .str.contains(
        cell_regex,
        regex=True,
        na=False
    )
)

df["is_cellular"] = (
    df["cell_by_bao"]
    | df["cell_by_metadata"]
    | df["cell_by_description"]
)


# =========================================
# 4. Single-protein / biochemical candidate
# =========================================

df["is_single_protein"] = (
    df["bao_label"]
    .fillna("")
    .str.lower()
    .eq("single protein format")
)

df["is_biochemical_candidate"] = (
    df["is_single_protein"]
    & (~df["is_cellular"])
)


# =========================================
# 5. WT candidate
# =========================================

df["is_wt_candidate"] = (
    ~df["is_variant"]
)


# =========================================
# 6. Strict Core QSAR assay
# =========================================

df["keep_core_qsar"] = (
    df["is_confidence_9"]
    & df["is_single_protein"]
    & df["is_wt_candidate"]
    & (~df["is_cellular"])
)


# =========================================
# Save
# =========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================
# Summary
# =========================================

print("\n=================================")
print("EGFR assay QC summary")
print("=================================")

print("Total assays:", len(df))

print(
    "Confidence score 9:",
    df["is_confidence_9"].sum()
)

print(
    "Single protein:",
    df["is_single_protein"].sum()
)

print(
    "Variant assays:",
    df["is_variant"].sum()
)

print(
    "Cellular assays:",
    df["is_cellular"].sum()
)

print(
    "WT candidates:",
    df["is_wt_candidate"].sum()
)

print(
    "Biochemical candidates:",
    df["is_biochemical_candidate"].sum()
)

print(
    "\nKEEP for Core QSAR:",
    df["keep_core_qsar"].sum()
)


# =========================================
# Exclusion reasons
# =========================================

print("\n--- Exclusion reasons ---")

print(
    "confidence != 9:",
    (~df["is_confidence_9"]).sum()
)

print(
    "not single protein:",
    (~df["is_single_protein"]).sum()
)

print(
    "variant:",
    df["is_variant"].sum()
)

print(
    "cellular:",
    df["is_cellular"].sum()
)


# =========================================
# Show examples
# =========================================

print("\n--- Example kept assays ---")

print(
    df.loc[
        df["keep_core_qsar"],
        [
            "assay_chembl_id",
            "confidence_score",
            "bao_label",
            "description"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\n--- Example excluded variant assays ---")

print(
    df.loc[
        df["is_variant"],
        [
            "assay_chembl_id",
            "bao_label",
            "description"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_FILE)
