from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docking/3W32/phase4C"
DIR = BASE / "ligands/EGFR_CAND_020"

SDF = DIR / "EGFR_CAND_020.sdf"
PREP = DIR / "EGFR_CAND_020.pdbqt"
DOCK = (
    BASE / "docking/EGFR_CAND_020"
    / "EGFR_CAND_020_docked.pdbqt"
)

OUT = ROOT / "reports/pose_review_geometry"
OUT.mkdir(parents=True, exist_ok=True)

ELEMENTS = {
    "A": "C", "C": "C", "N": "N", "NA": "N",
    "O": "O", "OA": "O", "S": "S", "SA": "S",
    "P": "P", "F": "F", "CL": "Cl",
    "BR": "Br", "I": "I"
}


def read_pose(path):
    atoms = []

    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break

        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue

        typ = line.split()[-1].upper()
        if typ in ("H", "HD", "HS"):
            continue

        if typ not in ELEMENTS:
            raise ValueError(f"Unknown atom type: {typ}")

        xyz = np.array([
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54])
        ])

        atoms.append({
            "name": line[12:16].strip(),
            "serial": line[6:11].strip(),
            "type": typ,
            "element": ELEMENTS[typ],
            "xyz": xyz
        })

    return atoms


mol = Chem.SDMolSupplier(
    str(SDF), removeHs=False
)[0]

if mol is None or not mol.GetNumConformers():
    raise RuntimeError("Cannot load original SDF.")

prepared = read_pose(PREP)
docked = read_pose(DOCK)

heavy = [
    a for a in mol.GetAtoms()
    if a.GetAtomicNum() > 1
]

if not len(heavy) == len(prepared) == len(docked):
    raise RuntimeError("Heavy atom counts disagree.")

def signature(items):
    return [
        (a["serial"], a["name"], a["type"])
        for a in items
    ]

if signature(prepared) != signature(docked):
    raise RuntimeError(
        "Prepared/docked atom order differs."
    )

source_xyz = mol.GetConformer().GetPositions()
original = np.array([
    source_xyz[a.GetIdx()] for a in heavy
])
prep_xyz = np.array([a["xyz"] for a in prepared])

cost = np.linalg.norm(
    original[:, None, :] - prep_xyz[None, :, :],
    axis=2
)

for i, atom in enumerate(heavy):
    for j, item in enumerate(prepared):
        if atom.GetSymbol() != item["element"]:
            cost[i, j] = 1e6

rows, cols = linear_sum_assignment(cost)
max_error = float(cost[rows, cols].max())

print("Mapping error:", round(max_error, 5), "A")

if max_error > 0.02:
    raise RuntimeError(
        "SDF/prepared atom mapping is unreliable."
    )

# Original SDF heavy-atom order is retained.
bonded = Chem.RemoveHs(mol)
bonded.RemoveAllConformers()

conf = Chem.Conformer(len(heavy))
mapped = {}

for i, j in zip(rows, cols):
    xyz = docked[int(j)]["xyz"]
    conf.SetAtomPosition(
        int(i),
        tuple(float(v) for v in xyz)
    )
    mapped[int(j)] = int(i)

bonded.AddConformer(conf, assignId=True)

# PDBQT heavy atom 15 must map to SDF atom 6.
n15 = mapped[14]

if n15 != 5:
    raise RuntimeError(
        f"Unexpected N15 mapping: SDF atom {n15 + 1}"
    )

if bonded.GetAtomWithIdx(n15).GetSymbol() != "N":
    raise RuntimeError("Mapped atom is not nitrogen.")

path = OUT / "EGFR_CAND_020_docked_bonded.sdf"

writer = Chem.SDWriter(str(path))
writer.write(bonded)
writer.close()

print("Docked bonded SDF:", path)
print("Heavy atoms:", bonded.GetNumAtoms())
print("PDBQT N15 = SDF atom:", n15 + 1)
print("Topology and docked coordinates combined.")
