from importlib import import_module
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

geo = import_module(
    "scripts.43_audit_egfr_candidate_pose_geometry"
)
reader = import_module(
    "scripts.46_verify_cand020_n15_identity"
)

ROOT = geo.ROOT
BASE = ROOT / "docking/3W32/phase4C"
HOPT = ROOT / "docking/3W32/receptor/3W32_reduce2_Hopt.pdb"
OUT = ROOT / "reports/pose_review_geometry/Polar_Contact_Chemistry_Audit.csv"

factory = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle(d, h, a):
    v = d - h
    w = a - h
    cosine = np.dot(v, w) / (
        np.linalg.norm(v) * np.linalg.norm(w)
    )
    return float(np.degrees(
        np.arccos(np.clip(cosine, -1, 1))
    ))


def protein_hopt_atoms():
    atoms = []
    for line in HOPT.read_text().splitlines():
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        if line[16:17] not in (" ", "A"):
            continue
        name = line[12:16].strip()
        element = line[76:78].strip().upper()
        if not element:
            element = (
                "H" if re.match(r"^\d*H", name.upper())
                else name[:1].upper()
            )
        atoms.append({
            "name": name,
            "element": element,
            "chain": line[21:22].strip(),
            "resi": line[22:26].strip(),
            "resname": line[17:20].strip(),
            "xyz": geo.xyz_from_line(line),
        })
    return atoms


def chemistry(cid, source):
    sdf = BASE / "ligands" / cid / f"{cid}.sdf"
    prep = BASE / "ligands" / cid / f"{cid}.pdbqt"
    pose = Path(source.loc[cid, "docked_pdbqt"])

    mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
    if mol is None:
        raise RuntimeError(f"{cid}: invalid SDF")

    expected = Chem.MolFromSmiles(
        source.loc[cid, "standardized_smiles"]
    )
    if expected is None or (
        Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
        != Chem.MolToSmiles(expected, isomericSmiles=True)
    ):
        raise RuntimeError(f"{cid}: SDF/SMILES mismatch")

    p = reader.read_pdbqt(prep)
    q = reader.read_pdbqt(pose)
    heavy = [a for a in mol.GetAtoms()
             if a.GetAtomicNum() > 1]

    if len(heavy) != len(p) or len(p) != len(q):
        raise RuntimeError(f"{cid}: heavy atom mismatch")

    sig = lambda atoms: [
        (a["serial"], a["name"], a["type"], a["element"])
        for a in atoms
    ]
    if sig(p) != sig(q):
        raise RuntimeError(f"{cid}: PDBQT ordering changed")

    xyz = mol.GetConformer().GetPositions()
    cost = np.linalg.norm(
        np.array([xyz[a.GetIdx()] for a in heavy])[:, None, :]
        - np.array([a["xyz"] for a in p])[None, :, :],
        axis=2
    )
    for i, a in enumerate(heavy):
        for j, b in enumerate(p):
            if a.GetSymbol() != b["element"]:
                cost[i, j] = 1e6

    ii, jj = linear_sum_assignment(cost)
    err = float(cost[ii, jj].max())
    if err > 0.02:
        raise RuntimeError(f"{cid}: mapping error {err:.4f} A")

    mapping = {
        int(j): heavy[int(i)].GetIdx()
        for i, j in zip(ii, jj)
    }

    acceptors, donors = set(), set()
    for f in factory.GetFeaturesForMol(mol):
        if f.GetFamily() == "Acceptor":
            acceptors.update(f.GetAtomIds())
        if f.GetFamily() == "Donor":
            donors.update(f.GetAtomIds())

    return q, mapping, acceptors, donors, err


def main():
    source = pd.read_csv(geo.SOURCE_CSV).set_index(
        "phase4C_candidate_id"
    )
    receptor = geo.read_pdb(geo.RECEPTOR, {"ATOM"})
    protein = protein_hopt_atoms()
    records = []

    for cid in ("EGFR_CAND_003", "EGFR_CAND_020"):
        ligand, mapping, acceptors, donors, err = chemistry(
            cid, source
        )
        print(f"\n{cid} | mapping error: {err:.4f} A")

        for j, la in enumerate(ligand):
            if la["element"] not in ("N", "O"):
                continue

            idx = mapping[j]
            is_acceptor = idx in acceptors
            is_donor = idx in donors

            for ra in receptor:
                if ra["element"] not in ("N", "O"):
                    continue

                d = dist(la["xyz"], ra["xyz"])
                if d > 3.6:
                    continue

                same = lambda a: (
                    a["chain"] == ra["chain"]
                    and a["resi"] == ra["resi"]
                    and a["resname"] == ra["resname"]
                )
                matching_heavy = [
                    a for a in protein
                    if same(a) and a["name"] == ra["name"]
                ]
                if len(matching_heavy) != 1:
                    raise RuntimeError(
                        f"Missing Hopt heavy atom {ra['resname']}"
                        f"{ra['resi']} {ra['name']}"
                    )
                donor_xyz = matching_heavy[0]["xyz"]
                if dist(donor_xyz, ra["xyz"]) > 0.35:
                    raise RuntimeError("Receptor coordinates differ")

                hs = [
                    a for a in protein
                    if same(a) and a["element"] == "H"
                    and dist(a["xyz"], donor_xyz) <= 1.25
                ]

                status = "no_supported_DA_direction"
                dha = None
                ha = None

                if is_acceptor and hs:
                    possibilities = [
                        (
                            angle(donor_xyz, h["xyz"], la["xyz"]),
                            dist(h["xyz"], la["xyz"])
                        )
                        for h in hs
                    ]
                    dha, ha = max(
                        possibilities, key=lambda x: x[0]
                    )
                    status = (
                        "provisional_geometry_pass"
                        if d <= 3.5 and ha <= 2.7 and dha >= 135
                        else "geometry_not_supported"
                    )
                elif is_donor and ra["element"] == "O":
                    status = "ligand_donor_H_not_evaluated"
                elif is_acceptor and ra["name"] == "N":
                    status = "protein_donor_H_not_found"

                record = {
                    "candidate": cid,
                    "ligand_pdbqt_atom": j + 1,
                    "ligand_sdf_atom": idx + 1,
                    "ligand_element": la["element"],
                    "ligand_acceptor": is_acceptor,
                    "ligand_donor": is_donor,
                    "protein_residue": (
                        f"{ra['resname']}{ra['resi']}"
                    ),
                    "protein_atom": ra["name"],
                    "distance_A": round(d, 3),
                    "D_H_A_angle_deg": (
                        round(dha, 2) if dha is not None else None
                    ),
                    "H_A_distance_A": (
                        round(ha, 3) if ha is not None else None
                    ),
                    "status": status,
                }
                records.append(record)

    df = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 65)
    print("POLAR CONTACT CHEMISTRY AUDIT")
    print("=" * 65)
    print(df.to_string(index=False))
    print("\nSaved:", OUT)
    print("No experimental hydrogen bonds are confirmed.")


if __name__ == "__main__":
    main()
