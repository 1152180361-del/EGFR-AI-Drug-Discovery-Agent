from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_qsar_modeling.csv"
)

MODEL_DIR = (
    ROOT
    / "models"
)

REPORT_DIR = (
    ROOT
    / "reports"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_FILE = (
    MODEL_DIR
    / "egfr_qsar_production_rf.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "egfr_qsar_production_rf.json"
)

TRAINING_MANIFEST_FILE = (
    REPORT_DIR
    / "phase2_production_qsar_manifest.csv"
)


# =========================================
# Fixed production configuration
# =========================================

RANDOM_SEED = 42

FP_RADIUS = 2
FP_SIZE = 2048

RF_PARAMS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


# =========================================
# Load final curated dataset
# =========================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    "RDKit version:",
    rdBase.rdkitVersion
)

print(
    "Production training compounds:",
    len(df)
)


# =========================================
# Required columns
# =========================================

required_columns = [
    "standardized_smiles",
    "standardized_inchikey",
    "pIC50",
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
# Morgan generator
# =========================================

generator = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_SIZE
    )
)


# =========================================
# Generate feature matrix
# =========================================

fingerprints = []


for i, smiles in enumerate(
    df["standardized_smiles"],
    start=1
):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:

        raise RuntimeError(
            f"RDKit parse failed "
            f"for row {i}: {smiles}"
        )


    fp = generator.GetFingerprint(
        mol
    )


    array = np.zeros(
        FP_SIZE,
        dtype=np.uint8
    )


    DataStructs.ConvertToNumpyArray(
        fp,
        array
    )


    fingerprints.append(
        array
    )


    if i % 500 == 0:

        print(
            f"Processed "
            f"{i}/{len(df)}"
        )


X = np.asarray(
    fingerprints,
    dtype=np.uint8
)

y = df[
    "pIC50"
].to_numpy(
    dtype=np.float32
)


print(
    "Feature matrix:",
    X.shape
)

print(
    "Label vector:",
    y.shape
)


# =========================================
# Final sanity checks
# =========================================

if X.shape != (
    len(df),
    FP_SIZE
):

    raise RuntimeError(
        "Unexpected feature matrix shape."
    )


if np.isnan(y).any():

    raise RuntimeError(
        "NaN found in pIC50."
    )


if df[
    "standardized_smiles"
].duplicated().any():

    raise RuntimeError(
        "Duplicate structures detected."
    )


# =========================================
# Train production Random Forest
# =========================================

print(
    "\nTraining production RF..."
)


model = RandomForestRegressor(
    **RF_PARAMS
)


model.fit(
    X,
    y
)


print(
    "Training complete."
)


# =========================================
# Save model
# =========================================

joblib.dump(
    model,
    MODEL_FILE
)


# =========================================
# Save training manifest
# =========================================

manifest = df[
    [
        "standardized_inchikey",
        "standardized_smiles",
        "pIC50",
    ]
].copy()


manifest.to_csv(
    TRAINING_MANIFEST_FILE,
    index=False
)


# =========================================
# Metadata
# =========================================

metadata = {

    "project":
        "EGFR_AIDD_Pipeline",

    "target":
        "human EGFR",

    "dataset":
        "curated WT biochemical IC50",

    "n_training_compounds":
        int(
            len(df)
        ),

    "label":
        "median pIC50",

    "representation": {
        "type":
            "Morgan fingerprint",

        "radius":
            FP_RADIUS,

        "fp_size":
            FP_SIZE,
    },

    "model": {
        "type":
            "RandomForestRegressor",

        "parameters":
            RF_PARAMS,
    },

    "selection_basis": {
        "method":
            "repeated scaffold split",

        "validation_seeds":
            [
                11,
                22,
                33,
                42,
                55,
            ],

        "selected_representation":
            "Morgan",

        "selected_model":
            "RF1",
    },

    "important_note":
        (
            "This production model is trained "
            "on the complete curated dataset. "
            "Performance claims must use the "
            "previous held-out scaffold "
            "validation/test experiments, "
            "not training-set performance."
        ),
}


with open(
    METADATA_FILE,
    "w"
) as handle:

    json.dump(
        metadata,
        handle,
        indent=4
    )


# =========================================
# Final report
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 2.7 PRODUCTION QSAR"
)

print(
    "================================="
)

print(
    "Training compounds:",
    len(df)
)

print(
    "Feature dimension:",
    X.shape[1]
)

print(
    "pIC50 mean:",
    round(
        float(
            np.mean(y)
        ),
        4
    )
)

print(
    "pIC50 median:",
    round(
        float(
            np.median(y)
        ),
        4
    )
)

print(
    "pIC50 min:",
    round(
        float(
            np.min(y)
        ),
        4
    )
)

print(
    "pIC50 max:",
    round(
        float(
            np.max(y)
        ),
        4
    )
)

print(
    "\nModel:"
)

print(
    "RandomForestRegressor"
)

print(
    RF_PARAMS
)

print(
    "\nSaved production model:"
)

print(
    MODEL_FILE
)

print(
    "\nSaved metadata:"
)

print(
    METADATA_FILE
)

print(
    "\nSaved training manifest:"
)

print(
    TRAINING_MANIFEST_FILE
)
