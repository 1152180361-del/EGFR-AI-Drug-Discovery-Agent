import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docking/3W32/phase4C"
CID = "EGFR_CAND_003"

SDF = BASE / "ligands" / CID / f"{CID}.sdf"
PREP = BASE / "ligands" / CID / f"{CID}.pdbqt"
DOCK = BASE / "docking" / CID / f"{CID}_docked.pdbqt"

SOURCE = (
    ROOT / "results/zinc15_screening"
    / "phase4E_final_candidate_evidence_matrix.csv"
)

OUT = ROOT / "reports/protonation_review"

# Verified mapping from previous analysis:
# PDBQT heavy atom 3 -> SDF atom 9.
SDF_N_INDEX = 8
PDBQT_N_INDEX = 2


def pdbqt_heavy_atoms(path):
    atoms = []
    for line in path.read_text(
        errors="replace"
    ).splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        kind = line.split()[-1].upper()
        if kind in {"H", "HD", "HS"}:
            continue
        atoms.append({
            "name": line[12:16].strip(),
            "type": kind,
            "xyz": np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54])
            ])
        })
    return atoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ph", type=float, default=7.4
    )
    args = parser.parse_args()

    if not 0 <= args.ph <= 14:
        parser.error("--ph must be between 0 and 14")

    for path in (SDF, PREP, DOCK, SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)

    mol = Chem.SDMolSupplier(
        str(SDF), removeHs=False
    )[0]

    if mol is None or not mol.GetNumConformers():
        raise RuntimeError("Invalid source SDF")

    df = pd.read_csv(SOURCE)
    rows = df.loc[
        df["phase4C_candidate_id"] == CID
    ]
    if len(rows) != 1:
        raise RuntimeError("Candidate lookup failed")

    ref = Chem.MolFromSmiles(
        rows.iloc[0]["standardized_smiles"]
    )
    if ref is None:
        raise RuntimeError("Invalid reference SMILES")

    neutral_smiles = Chem.MolToSmiles(
        Chem.RemoveHs(mol), isomericSmiles=True
    )
    ref_smiles = Chem.MolToSmiles(
        ref, isomericSmiles=True
    )
    if neutral_smiles != ref_smiles:
        raise RuntimeError("SDF/SMILES mismatch")

    n = mol.GetAtomWithIdx(SDF_N_INDEX)

    if not (
        n.GetSymbol() == "N"
        and n.GetFormalCharge() == 0
        and n.GetDegree() == 3
        and n.GetTotalNumHs(
            includeNeighbors=True
        ) == 0
        and all(
            x.GetSymbol() == "C"
            for x in n.GetNeighbors()
        )
    ):
        raise RuntimeError(
            "N3 is not the expected neutral "
            "tertiary amine"
        )

    prepared = pdbqt_heavy_atoms(PREP)
    docked = pdbqt_heavy_atoms(DOCK)

    if not len(prepared) == len(docked) == 20:
        raise RuntimeError("Unexpected atom count")

    pre_n = prepared[PDBQT_N_INDEX]
    dock_n = docked[PDBQT_N_INDEX]

    if (
        pre_n["type"] != "NA"
        or dock_n["type"] != "NA"
        or pre_n["name"] != dock_n["name"]
    ):
        raise RuntimeError("PDBQT N3 mapping mismatch")

    sdf_xyz = np.array(
        mol.GetConformer().GetAtomPosition(
            SDF_N_INDEX
        )
    )

    map_error = float(np.linalg.norm(
        sdf_xyz - pre_n["xyz"]
    ))

    if map_error > 0.02:
        raise RuntimeError(
            "SDF/PDBQT N3 coordinate mapping "
            f"failed: {map_error:.4f} A"
        )

    # Construct the charged chemical microstate.
    # This changes chemical identity only.
    # No 3D proton position is generated.
    rw = Chem.RWMol(mol)
    protonated_n = rw.GetAtomWithIdx(SDF_N_INDEX)
    protonated_n.SetFormalCharge(1)
    protonated_n.SetNumExplicitHs(1)
    protonated_n.SetNoImplicit(True)

    charged = rw.GetMol()
    Chem.SanitizeMol(charged)

    if not (
        Chem.GetFormalCharge(charged) == 1
        and charged.GetAtomWithIdx(
            SDF_N_INDEX
        ).GetTotalNumHs(
            includeNeighbors=True
        ) == 1
    ):
        raise RuntimeError(
            "Protonated valence validation failed"
        )

    charged_smiles = Chem.MolToSmiles(
        Chem.RemoveHs(charged),
        isomericSmiles=True
    )

    print("=" * 60)
    print("CAND003 PROTONATION STATE AUDIT")
    print("=" * 60)

    print("SDF/PDBQT N3 mapping error:",
          round(map_error, 5), "A")
    print("Prepared PDBQT N3 type:", pre_n["type"])

    print("\nNEUTRAL STATE")
    print("SMILES:", neutral_smiles)
    print("Charge:", Chem.GetFormalCharge(mol))
    print("Formula:", CalcMolFormula(mol))
    print("N3 attached H: 0")
    print("N3: potential neutral-amine acceptor")

    print("\nPROTONATED STATE (2D ONLY)")
    print("SMILES:", charged_smiles)
    print("Charge:", Chem.GetFormalCharge(charged))
    print("Formula:", CalcMolFormula(charged))
    print("N3 attached H: 1")
    print("N3: protonated ammonium, not acceptor")
    print("No charged 3D pose was generated.")

    records = []
    for pka in (6.0, 7.0, 8.0, 9.0, 10.0):
        fraction = 1 / (
            1 + 10 ** (args.ph - pka)
        )
        records.append({
            "assumed_pH": args.ph,
            "hypothetical_pKa": pka,
            "protonated_fraction_percent": round(
                100 * fraction, 2
            ),
            "neutral_fraction_percent": round(
                100 * (1 - fraction), 2
            )
        })

    scenarios = pd.DataFrame(records)

    print("\nHYPOTHETICAL pKa SENSITIVITY")
    print("Assumed pH:", args.ph)
    print(scenarios.to_string(index=False))
    print(
        "\nThese pKa values are scenarios, "
        "NOT predicted or measured pKa values."
    )

    OUT.mkdir(parents=True, exist_ok=True)

    scenarios.to_csv(
        OUT / "CAND003_pKa_scenarios.csv",
        index=False
    )

    report = {
        "candidate": CID,
        "source_sdf": str(SDF),
        "source_pdbqt": str(PREP),
        "sdf_pdbqt_map_error_A": map_error,
        "neutral_smiles": neutral_smiles,
        "protonated_smiles_2D": charged_smiles,
        "neutral_formal_charge": 0,
        "protonated_formal_charge": 1,
        "candidate_pKa_known": False,
        "actual_assay_pH_known": False,
        "scenario_pH": args.ph,
        "new_docking_performed": False,
        "source_files_modified": False,
        "limitation": (
            "No experimental pKa, no charged "
            "3D pose, and no protonated-state "
            "docking validation."
        )
    }

    path = OUT / "CAND003_protonation_audit.json"
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nPROTONATION AUDIT COMPLETE")
    print("Saved:", path)


if __name__ == "__main__":
    main()
