from pathlib import Path
import sys
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


RDLogger.DisableLog("rdApp.*")


if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python 35_worker_standardize_zinc15.py "
        "<input.csv> <output.csv>"
    )


INPUT = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


uncharger = (
    rdMolStandardize.Uncharger()
)


def standardize_one(smiles):

    result = {
        "standardization_status":
            "FAIL",

        "standardization_error":
            "",

        "standardized_smiles":
            "",

        "standardized_inchikey":
            "",

        "standardization_changed":
            False,
    }

    try:

        if pd.isna(smiles):

            result[
                "standardization_error"
            ] = "missing_smiles"

            return result


        smiles = str(
            smiles
        ).strip()


        if not smiles:

            result[
                "standardization_error"
            ] = "empty_smiles"

            return result


        # ----------------------------------------
        # Parse
        # ----------------------------------------

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:

            result[
                "standardization_error"
            ] = "MolFromSmiles_failed"

            return result


        original_canonical = (
            Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
            )
        )


        # ----------------------------------------
        # Same standardization logic
        # used in Phase 1
        # ----------------------------------------

        mol = (
            rdMolStandardize
            .Cleanup(
                mol
            )
        )


        mol = (
            rdMolStandardize
            .FragmentParent(
                mol
            )
        )


        mol = (
            uncharger.uncharge(
                mol
            )
        )


        Chem.SanitizeMol(
            mol
        )


        # ----------------------------------------
        # Require at least one carbon
        # ----------------------------------------

        has_carbon = any(
            atom.GetAtomicNum() == 6
            for atom in mol.GetAtoms()
        )


        if not has_carbon:

            result[
                "standardization_error"
            ] = "no_carbon"

            return result


        # ----------------------------------------
        # Canonical standardized representation
        # ----------------------------------------

        standardized_smiles = (
            Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
            )
        )


        if not standardized_smiles:

            result[
                "standardization_error"
            ] = "empty_standardized_smiles"

            return result


        inchikey = (
            Chem.MolToInchiKey(
                mol
            )
        )


        result[
            "standardization_status"
        ] = "PASS"


        result[
            "standardized_smiles"
        ] = standardized_smiles


        result[
            "standardized_inchikey"
        ] = inchikey


        result[
            "standardization_changed"
        ] = (
            standardized_smiles
            != original_canonical
        )


        return result


    except Exception as exc:

        result[
            "standardization_error"
        ] = (
            type(exc).__name__
            + ": "
            + str(exc)[:300]
        )

        return result


df = pd.read_csv(
    INPUT
)


records = []


for _, row in df.iterrows():

    rec = row.to_dict()

    std = standardize_one(
        row["smiles"]
    )

    rec.update(
        std
    )

    records.append(
        rec
    )


out = pd.DataFrame(
    records
)


out.to_csv(
    OUTPUT,
    index=False,
)


print(
    f"WORKER COMPLETE: "
    f"{len(out)} rows | "
    f"PASS="
    f"{(out['standardization_status'] == 'PASS').sum()}"
)
