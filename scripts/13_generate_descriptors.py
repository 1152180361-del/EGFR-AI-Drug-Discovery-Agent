from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, rdBase
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)


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

OUTPUT_FILE = (
    ROOT
    / "data"
    / "features"
    / "egfr_physchem_descriptors.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "phase2_descriptor_qc.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Load dataset
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
    "Input compounds:",
    len(df)
)


# =========================================
# Descriptor function
# =========================================

def calculate_descriptors(smiles):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    return {
        "MolWt":
            Descriptors.MolWt(mol),

        "LogP":
            Crippen.MolLogP(mol),

        "TPSA":
            rdMolDescriptors.CalcTPSA(mol),

        "HBD":
            Lipinski.NumHDonors(mol),

        "HBA":
            Lipinski.NumHAcceptors(mol),

        "RotatableBonds":
            Lipinski.NumRotatableBonds(mol),

        "RingCount":
            Lipinski.RingCount(mol),

        "AromaticRingCount":
            rdMolDescriptors.CalcNumAromaticRings(
                mol
            ),

        "FractionCSP3":
            rdMolDescriptors.CalcFractionCSP3(
                mol
            ),

        "HeavyAtomCount":
            mol.GetNumHeavyAtoms(),

        "FormalCharge":
            Chem.GetFormalCharge(mol),
    }


# =========================================
# Calculate descriptors
# =========================================

records = []

failed = []


for i, record in enumerate(
    df.to_dict("records"),
    start=1
):

    smiles = record[
        "standardized_smiles"
    ]

    descriptor_values = (
        calculate_descriptors(
            smiles
        )
    )

    if descriptor_values is None:

        failed.append(
            record[
                "standardized_inchikey"
            ]
        )

        continue


    output_record = {
        "standardized_inchikey":
            record[
                "standardized_inchikey"
            ],

        "standardized_smiles":
            smiles,

        "split_scaffold_id":
            record[
                "split_scaffold_id"
            ],

        "pIC50":
            record[
                "pIC50"
            ],
    }

    output_record.update(
        descriptor_values
    )

    records.append(
        output_record
    )


    if i % 500 == 0:

        print(
            f"Processed "
            f"{i}/{len(df)}"
        )


# =========================================
# Build output dataframe
# =========================================

descriptor_df = pd.DataFrame(
    records
)


# =========================================
# Descriptor columns
# =========================================

descriptor_columns = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "AromaticRingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "FormalCharge",
]


# =========================================
# QC
# =========================================

missing_counts = (
    descriptor_df[
        descriptor_columns
    ]
    .isna()
    .sum()
)

infinite_counts = {}

for column in descriptor_columns:

    values = descriptor_df[
        column
    ].to_numpy(
        dtype=float
    )

    infinite_counts[column] = int(
        np.isinf(values).sum()
    )


duplicate_count = int(
    descriptor_df[
        "standardized_smiles"
    ]
    .duplicated()
    .sum()
)


# =========================================
# Save descriptor dataset
# =========================================

descriptor_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================
# Save descriptor statistics
# =========================================

summary_records = []


for column in descriptor_columns:

    values = pd.to_numeric(
        descriptor_df[column],
        errors="coerce"
    )

    summary_records.append(
        {
            "descriptor":
                column,

            "mean":
                values.mean(),

            "std":
                values.std(),

            "min":
                values.min(),

            "median":
                values.median(),

            "max":
                values.max(),

            "missing":
                int(
                    values.isna().sum()
                ),

            "infinite":
                infinite_counts[
                    column
                ],
        }
    )


summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    REPORT_FILE,
    index=False
)


# =========================================
# Final summary
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 2.6A DESCRIPTOR QC"
)

print(
    "================================="
)

print(
    "Input compounds:",
    len(df)
)

print(
    "Descriptor records:",
    len(descriptor_df)
)

print(
    "Failed structures:",
    len(failed)
)

print(
    "Descriptor dimension:",
    len(descriptor_columns)
)

print(
    "Duplicate standardized SMILES:",
    duplicate_count
)

print(
    "\nMissing descriptor values:"
)

print(
    missing_counts.to_string()
)

print(
    "\nDescriptor statistics:"
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\nSaved descriptor dataset:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSaved QC report:"
)

print(
    REPORT_FILE
)
