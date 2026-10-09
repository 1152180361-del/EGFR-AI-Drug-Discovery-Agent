from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

root = Path.home() / "EGFR_AIDD_Pipeline"
base = root / "docking/3W32/phase4C"
sdf = base / "ligands/EGFR_CAND_020/EGFR_CAND_020.sdf"
docked = base / "docking/EGFR_CAND_020/EGFR_CAND_020_docked.pdbqt"
csv = root / "results/zinc15_screening/phase4E_final_candidate_evidence_matrix.csv"

elements = {
    "A": "C", "C": "C", "N": "N", "NA": "N",
    "O": "O", "OA": "O", "S": "S", "SA": "S",
    "F": "F", "CL": "Cl", "BR": "Br", "I": "I",
    "P": "P", "H": "H", "HD": "H", "HS": "H"
}

def read_pdbqt(path):
    atoms = []
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        typ = line.split()[-1].upper()
        if typ not in elements:
            raise ValueError(f"Unknown atom type: {typ}")
        if elements[typ] == "H":
            continue
        atoms.append({
            "serial": line[6:11].strip(),
            "name": line[12:16].strip(),
            "type": typ,
            "element": elements[typ],
            "xyz": np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54])
            ])
        })
    return atoms

def main():
    print("=" * 55)
    print("CAND020 N15 CHEMICAL IDENTITY")
    print("=" * 55)

    matches = sorted(
        p for p in base.rglob("*CAND_020*.pdbqt")
        if p.resolve() != docked.resolve()
        and "docked" not in p.name.lower()
    )

    print("Prepared PDBQT candidates:", len(matches))
    for p in matches:
        print(p)

    if len(matches) != 1:
        raise RuntimeError("Cannot uniquely locate prepared PDBQT.")

    mol = Chem.SDMolSupplier(
        str(sdf), removeHs=False
    )[0]
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError("Invalid SDF or missing coordinates.")

    source = pd.read_csv(csv)
    row = source.loc[
        source["phase4C_candidate_id"] == "EGFR_CAND_020"
    ]
    if len(row) != 1:
        raise RuntimeError("Source candidate ID mismatch.")

    expected = Chem.MolFromSmiles(
        row.iloc[0]["standardized_smiles"]
    )
    original = Chem.RemoveHs(mol)

    same_structure = (
        Chem.MolToSmiles(original, isomericSmiles=True)
        == Chem.MolToSmiles(expected, isomericSmiles=True)
    )
    print("SDF/source SMILES agreement:", same_structure)
    if not same_structure:
        raise RuntimeError("SDF/source chemical identity mismatch.")

    prepared = read_pdbqt(matches[0])
    pose = read_pdbqt(docked)
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

    print("SDF heavy atoms:", len(heavy))
    print("Prepared heavy atoms:", len(prepared))
    print("Docked heavy atoms:", len(pose))

    if not len(heavy) == len(prepared) == len(pose):
        raise RuntimeError("Heavy atom count mismatch.")

    def signature(atoms):
        return [
            (a["serial"], a["name"], a["type"])
            for a in atoms
        ]

    if signature(prepared) != signature(pose):
        raise RuntimeError("Prepared/docked atom order mismatch.")

    xyz = mol.GetConformer().GetPositions()
    sdf_xyz = np.array([xyz[a.GetIdx()] for a in heavy])
    prep_xyz = np.array([a["xyz"] for a in prepared])

    cost = np.linalg.norm(
        sdf_xyz[:, None, :] - prep_xyz[None, :, :], axis=2
    )
    for i, atom in enumerate(heavy):
        for j, p in enumerate(prepared):
            if atom.GetSymbol() != p["element"]:
                cost[i, j] = 1e6

    rows, cols = linear_sum_assignment(cost)
    max_error = float(cost[rows, cols].max())
    print("Maximum SDF/prepared atom distance:", round(max_error, 4), "A")

    if max_error > 0.20:
        raise RuntimeError(
            "Coordinate mapping not established. "
            "Do not assume SDF/PDBQT atom order."
        )

    mapping = {
        int(j): heavy[int(i)].GetIdx()
        for i, j in zip(rows, cols)
    }
    atom = mol.GetAtomWithIdx(mapping[14])

    factory = ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )
    acceptors = set()
    for feature in factory.GetFeaturesForMol(mol):
        if feature.GetFamily() == "Acceptor":
            acceptors.update(feature.GetAtomIds())

    print("\nVERIFIED N15 IDENTITY")
    print("PDBQT heavy atom:", 15)
    print("Mapped SDF atom:", atom.GetIdx() + 1)
    print("Element:", atom.GetSymbol())
    print("Aromatic:", atom.GetIsAromatic())
    print("Formal charge:", atom.GetFormalCharge())
    print("RDKit H-bond acceptor:", atom.GetIdx() in acceptors)
    print("Neighbors:", [
        (n.GetIdx() + 1, n.GetSymbol())
        for n in atom.GetNeighbors()
    ])
    print("ATOM IDENTITY CHECK COMPLETE")

if __name__ == "__main__":
    main()
