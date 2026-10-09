from pathlib import Path
import multiprocessing as mp
import pickle
import tempfile
import os

import pandas as pd

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_wt_biochemical_ic50.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_standardized_activities.csv"
)

FAILURE_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_structure_failures.csv"
)

MAPPING_FILE = (
    ROOT
    / "data"
    / "interim"
    / "egfr_structure_mapping.csv"
)


# =========================================
# Standardize ONE molecule
# =========================================

def standardize_one(smiles):

    result = {
        "canonical_smiles": smiles,
        "standardization_success": False,
        "standardization_error": None,
        "standardized_smiles": None,
        "standardized_inchikey": None,
        "heavy_atom_count": None,
        "has_carbon": False,
    }

    if smiles is None:

        result["standardization_error"] = (
            "missing_smiles"
        )

        return result

    smiles = str(smiles).strip()

    if smiles == "":

        result["standardization_error"] = (
            "empty_smiles"
        )

        return result

    try:

        # ---------------------------------
        # Parse
        # ---------------------------------

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:

            result["standardization_error"] = (
                "rdkit_parse_failed"
            )

            return result


        # ---------------------------------
        # Cleanup
        # ---------------------------------

        mol = rdMolStandardize.Cleanup(
            mol
        )


        # ---------------------------------
        # Keep parent fragment
        # ---------------------------------

        mol = rdMolStandardize.FragmentParent(
            mol
        )


        # ---------------------------------
        # Neutralize
        # ---------------------------------

        uncharger = (
            rdMolStandardize.Uncharger()
        )

        mol = uncharger.uncharge(
            mol
        )


        # ---------------------------------
        # Sanitize
        # ---------------------------------

        Chem.SanitizeMol(
            mol
        )


        # ---------------------------------
        # Carbon check
        # ---------------------------------

        has_carbon = any(
            atom.GetAtomicNum() == 6
            for atom in mol.GetAtoms()
        )

        result["has_carbon"] = has_carbon

        if not has_carbon:

            result["standardization_error"] = (
                "no_carbon_atom"
            )

            return result


        # ---------------------------------
        # Canonical isomeric SMILES
        # ---------------------------------

        standardized_smiles = (
            Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True
            )
        )


        # ---------------------------------
        # Heavy atom count
        # ---------------------------------

        heavy_atom_count = (
            mol.GetNumHeavyAtoms()
        )


        # ---------------------------------
        # InChIKey
        #
        # Keep it inside isolated worker.
        # If native code crashes, parent
        # process remains alive.
        # ---------------------------------

        try:

            standardized_inchikey = (
                Chem.MolToInchiKey(
                    mol
                )
            )

        except Exception:

            standardized_inchikey = None


        result[
            "standardization_success"
        ] = True

        result[
            "standardized_smiles"
        ] = standardized_smiles

        result[
            "standardized_inchikey"
        ] = standardized_inchikey

        result[
            "heavy_atom_count"
        ] = heavy_atom_count

        return result


    except Exception as error:

        result["standardization_error"] = (
            f"{type(error).__name__}: "
            f"{str(error)}"
        )

        return result


# =========================================
# Worker process
#
# A native RDKit crash here kills only
# this child process, not the main pipeline.
# =========================================

def worker_process(
    smiles_list,
    output_path
):

    results = []

    for smiles in smiles_list:

        result = standardize_one(
            smiles
        )

        results.append(
            result
        )

    with open(
        output_path,
        "wb"
    ) as handle:

        pickle.dump(
            results,
            handle
        )


# =========================================
# Run isolated batch
# =========================================

def run_batch(
    smiles_list,
    context
):

    fd, temp_path = tempfile.mkstemp(
        prefix="egfr_rdkit_",
        suffix=".pkl"
    )

    os.close(fd)

    # Remove empty file so success/failure
    # can be determined cleanly.
    if os.path.exists(temp_path):
        os.remove(temp_path)

    process = context.Process(
        target=worker_process,
        args=(
            smiles_list,
            temp_path
        )
    )

    process.start()
    process.join()

    exit_code = process.exitcode

    if (
        exit_code == 0
        and os.path.exists(temp_path)
    ):

        try:

            with open(
                temp_path,
                "rb"
            ) as handle:

                results = pickle.load(
                    handle
                )

        finally:

            if os.path.exists(
                temp_path
            ):

                os.remove(
                    temp_path
                )

        return True, results

    if os.path.exists(
        temp_path
    ):

        os.remove(
            temp_path
        )

    return False, exit_code


# =========================================
# Recursive safe processing
#
# If a batch crashes:
#
# 100 molecules
# ↓
# 50 + 50
# ↓
# ...
# ↓
# isolate exact bad molecule
# =========================================

def process_safely(
    smiles_list,
    context,
    depth=0
):

    success, output = run_batch(
        smiles_list,
        context
    )

    if success:

        return output

    indent = "  " * depth

    print(
        f"{indent}Native crash detected "
        f"in batch of {len(smiles_list)} "
        f"molecules."
    )

    # -------------------------------------
    # Single molecule itself causes crash
    # -------------------------------------

    if len(smiles_list) == 1:

        smiles = smiles_list[0]

        print(
            f"{indent}Isolated crashing "
            f"structure:"
        )

        print(
            f"{indent}{smiles}"
        )

        return [
            {
                "canonical_smiles":
                    smiles,

                "standardization_success":
                    False,

                "standardization_error":
                    "native_crash",

                "standardized_smiles":
                    None,

                "standardized_inchikey":
                    None,

                "heavy_atom_count":
                    None,

                "has_carbon":
                    False,
            }
        ]


    # -------------------------------------
    # Split batch and retry each half
    # -------------------------------------

    midpoint = (
        len(smiles_list)
        // 2
    )

    left = smiles_list[
        :midpoint
    ]

    right = smiles_list[
        midpoint:
    ]

    left_results = (
        process_safely(
            left,
            context,
            depth + 1
        )
    )

    right_results = (
        process_safely(
            right,
            context,
            depth + 1
        )
    )

    return (
        left_results
        + right_results
    )


# =========================================
# Main
# =========================================

def main():

    print(
        "RDKit version:",
        rdBase.rdkitVersion
    )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        "Input activity records:",
        len(df)
    )

    print(
        "Input ChEMBL compounds:",
        df[
            "molecule_chembl_id"
        ].nunique()
    )

    unique_smiles = (
        df["canonical_smiles"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(
        "Unique original SMILES:",
        len(unique_smiles)
    )


    # =====================================
    # Use spawn instead of fork
    #
    # Better isolation for native libraries.
    # =====================================

    context = (
        mp.get_context(
            "spawn"
        )
    )


    # =====================================
    # Batch size
    # =====================================

    BATCH_SIZE = 100

    all_results = []


    print(
        "\nStarting isolated RDKit "
        "standardization..."
    )


    for start in range(
        0,
        len(unique_smiles),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(unique_smiles)
        )

        batch = unique_smiles[
            start:end
        ]

        print(
            f"\nProcessing "
            f"{start + 1}-{end} "
            f"/ {len(unique_smiles)}"
        )

        batch_results = (
            process_safely(
                batch,
                context
            )
        )

        all_results.extend(
            batch_results
        )


    # =====================================
    # Build mapping table
    # =====================================

    mapping = pd.DataFrame(
        all_results
    )

    mapping.to_csv(
        MAPPING_FILE,
        index=False
    )


    # =====================================
    # Merge back to activity records
    # =====================================

    merged = df.merge(
        mapping,
        on="canonical_smiles",
        how="left",
        validate="many_to_one"
    )


    # =====================================
    # Failures
    # =====================================

    failures = merged[
        ~merged[
            "standardization_success"
        ].fillna(False)
    ].copy()

    failures.to_csv(
        FAILURE_FILE,
        index=False
    )


    # =====================================
    # Successful records
    # =====================================

    clean = merged[
        merged[
            "standardization_success"
        ].fillna(False)
    ].copy()


    # =====================================
    # Structure-change flag
    # =====================================

    clean[
        "structure_changed"
    ] = (
        clean[
            "canonical_smiles"
        ]
        !=
        clean[
            "standardized_smiles"
        ]
    )


    # =====================================
    # Save standardized activities
    # =====================================

    clean.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # =====================================
    # Statistics
    # =====================================

    original_structure_count = (
        clean[
            "canonical_smiles"
        ].nunique()
    )

    standardized_structure_count = (
        clean[
            "standardized_smiles"
        ].nunique()
    )

    standardized_inchikey_count = (
        clean[
            "standardized_inchikey"
        ].nunique()
    )


    # -------------------------------------
    # Multiple ChEMBL IDs -> same structure
    # -------------------------------------

    chembl_per_structure = (
        clean[
            [
                "standardized_smiles",
                "molecule_chembl_id"
            ]
        ]
        .drop_duplicates()
        .groupby(
            "standardized_smiles"
        )[
            "molecule_chembl_id"
        ]
        .nunique()
    )

    multi_chembl_structures = (
        chembl_per_structure
        > 1
    ).sum()


    # -------------------------------------
    # Multiple raw representations collapse
    # -------------------------------------

    original_per_standardized = (
        clean[
            [
                "canonical_smiles",
                "standardized_smiles"
            ]
        ]
        .drop_duplicates()
        .groupby(
            "standardized_smiles"
        )[
            "canonical_smiles"
        ]
        .nunique()
    )

    collapsed_structures = (
        original_per_standardized
        > 1
    ).sum()


    # =====================================
    # Final summary
    # =====================================

    print(
        "\n================================="
    )

    print(
        "RDKit standardization summary"
    )

    print(
        "================================="
    )

    print(
        "Input activity records:",
        len(merged)
    )

    print(
        "Successful records:",
        len(clean)
    )

    print(
        "Failed records:",
        len(failures)
    )

    print(
        "\nOriginal ChEMBL compounds:",
        merged[
            "molecule_chembl_id"
        ].nunique()
    )

    print(
        "Unique original SMILES:",
        original_structure_count
    )

    print(
        "Unique standardized SMILES:",
        standardized_structure_count
    )

    print(
        "Unique standardized InChIKeys:",
        standardized_inchikey_count
    )

    print(
        "\nActivity records whose "
        "structure changed:",
        clean[
            "structure_changed"
        ].sum()
    )

    print(
        "Standardized structures with "
        ">1 ChEMBL compound ID:",
        multi_chembl_structures
    )

    print(
        "Standardized structures formed "
        "from >1 original SMILES:",
        collapsed_structures
    )


    # =====================================
    # Failure reasons
    # =====================================

    if len(failures) > 0:

        print(
            "\n--- Failure reasons ---"
        )

        print(
            failures[
                "standardization_error"
            ]
            .value_counts(
                dropna=False
            )
        )


    print(
        "\nSaved standardized activities:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nSaved structure mapping:"
    )

    print(
        MAPPING_FILE
    )

    print(
        "\nSaved failures:"
    )

    print(
        FAILURE_FILE
    )


# =========================================
# Required for multiprocessing spawn
# =========================================

if __name__ == "__main__":
    main()
