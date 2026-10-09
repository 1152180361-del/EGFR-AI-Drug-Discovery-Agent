from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

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

FEATURE_DIR = (
    ROOT
    / "data"
    / "features"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "phase2_morgan_feature_qc.csv"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Fingerprint settings
# =========================================

RADIUS = 2
FP_SIZE = 2048


# =========================================
# RDKit Morgan generator
# =========================================

generator = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE
    )
)


print(
    "RDKit version:",
    rdBase.rdkitVersion
)

print(
    "Morgan radius:",
    RADIUS
)

print(
    "Fingerprint size:",
    FP_SIZE
)


# =========================================
# Fingerprint one molecule
# =========================================

def smiles_to_fp(smiles):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:

        return None

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

    return array


# =========================================
# Process one dataset split
# =========================================

def process_split(
    split_name,
    input_file
):

    print(
        "\n================================="
    )

    print(
        f"Processing {split_name}"
    )

    print(
        "================================="
    )

    df = pd.read_csv(
        input_file,
        low_memory=False
    )


    # -------------------------------------
    # Required columns
    # -------------------------------------

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
            f"{split_name}: missing columns: "
            + ", ".join(
                missing_columns
            )
        )


    # -------------------------------------
    # Generate fingerprints
    # -------------------------------------

    fingerprints = []

    labels = []

    inchikeys = []

    smiles_list = []

    scaffold_ids = []

    failed = []


    for i, record in enumerate(
        df.to_dict("records"),
        start=1
    ):

        smiles = record[
            "standardized_smiles"
        ]

        fp = smiles_to_fp(
            smiles
        )

        if fp is None:

            failed.append(
                {
                    "standardized_smiles":
                        smiles,

                    "standardized_inchikey":
                        record[
                            "standardized_inchikey"
                        ]
                }
            )

            continue


        fingerprints.append(
            fp
        )

        labels.append(
            float(
                record[
                    "pIC50"
                ]
            )
        )

        inchikeys.append(
            str(
                record[
                    "standardized_inchikey"
                ]
            )
        )

        smiles_list.append(
            str(smiles)
        )

        scaffold_ids.append(
            str(
                record[
                    "split_scaffold_id"
                ]
            )
        )


        if i % 500 == 0:

            print(
                f"Processed "
                f"{i}/{len(df)}"
            )


    # -------------------------------------
    # Convert to arrays
    # -------------------------------------

    X = np.asarray(
        fingerprints,
        dtype=np.uint8
    )

    y = np.asarray(
        labels,
        dtype=np.float32
    )

    inchikey_array = np.asarray(
        inchikeys,
        dtype=str
    )

    smiles_array = np.asarray(
        smiles_list,
        dtype=str
    )

    scaffold_array = np.asarray(
        scaffold_ids,
        dtype=str
    )


    # -------------------------------------
    # Fingerprint QC
    # -------------------------------------

    bit_counts = X.sum(
        axis=1
    )

    bit_density = (
        bit_counts
        / FP_SIZE
    )


    zero_fp_count = int(
        np.sum(
            bit_counts == 0
        )
    )


    # -------------------------------------
    # Save compressed feature file
    # -------------------------------------

    output_file = (
        FEATURE_DIR
        / f"egfr_{split_name}_morgan.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        inchikey=inchikey_array,
        smiles=smiles_array,
        scaffold_id=scaffold_array,
    )


    # -------------------------------------
    # Return QC results
    # -------------------------------------

    summary = {
        "split":
            split_name,

        "input_compounds":
            len(df),

        "fingerprints_generated":
            len(X),

        "failed_structures":
            len(failed),

        "feature_dimension":
            FP_SIZE,

        "mean_on_bits":
            float(
                np.mean(
                    bit_counts
                )
            ),

        "median_on_bits":
            float(
                np.median(
                    bit_counts
                )
            ),

        "min_on_bits":
            int(
                np.min(
                    bit_counts
                )
            ),

        "max_on_bits":
            int(
                np.max(
                    bit_counts
                )
            ),

        "mean_bit_density":
            float(
                np.mean(
                    bit_density
                )
            ),

        "zero_fingerprints":
            zero_fp_count,

        "pIC50_mean":
            float(
                np.mean(y)
            ),

        "pIC50_std":
            float(
                np.std(
                    y,
                    ddof=1
                )
            ),
    }


    print(
        "Input compounds:",
        len(df)
    )

    print(
        "Fingerprints generated:",
        len(X)
    )

    print(
        "Failed structures:",
        len(failed)
    )

    print(
        "Feature matrix shape:",
        X.shape
    )

    print(
        "Label shape:",
        y.shape
    )

    print(
        "Mean ON bits:",
        round(
            summary[
                "mean_on_bits"
            ],
            2
        )
    )

    print(
        "Median ON bits:",
        round(
            summary[
                "median_on_bits"
            ],
            2
        )
    )

    print(
        "Mean bit density:",
        round(
            summary[
                "mean_bit_density"
            ],
            4
        )
    )

    print(
        "Zero fingerprints:",
        zero_fp_count
    )

    print(
        "Saved:",
        output_file
    )


    return summary


# =========================================
# Process Train / Validation / Test
# =========================================

summaries = []

summaries.append(
    process_split(
        "train",
        TRAIN_FILE
    )
)

summaries.append(
    process_split(
        "validation",
        VALID_FILE
    )
)

summaries.append(
    process_split(
        "test",
        TEST_FILE
    )
)


# =========================================
# Save QC summary
# =========================================

summary_df = pd.DataFrame(
    summaries
)

summary_df.to_csv(
    REPORT_FILE,
    index=False
)


# =========================================
# Final cross-split sanity checks
# =========================================

train_data = np.load(
    FEATURE_DIR
    / "egfr_train_morgan.npz"
)

valid_data = np.load(
    FEATURE_DIR
    / "egfr_validation_morgan.npz"
)

test_data = np.load(
    FEATURE_DIR
    / "egfr_test_morgan.npz"
)


train_keys = set(
    train_data[
        "inchikey"
    ].tolist()
)

valid_keys = set(
    valid_data[
        "inchikey"
    ].tolist()
)

test_keys = set(
    test_data[
        "inchikey"
    ].tolist()
)


train_valid_overlap = (
    train_keys
    & valid_keys
)

train_test_overlap = (
    train_keys
    & test_keys
)

valid_test_overlap = (
    valid_keys
    & test_keys
)


if train_valid_overlap:

    raise RuntimeError(
        "Molecule leakage: "
        "train vs validation"
    )

if train_test_overlap:

    raise RuntimeError(
        "Molecule leakage: "
        "train vs test"
    )

if valid_test_overlap:

    raise RuntimeError(
        "Molecule leakage: "
        "validation vs test"
    )


# =========================================
# Final summary
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 2.3 MORGAN FEATURES"
)

print(
    "================================="
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nCross-split molecule overlap:"
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
    "\nSaved feature QC report:"
)

print(
    REPORT_FILE
)
