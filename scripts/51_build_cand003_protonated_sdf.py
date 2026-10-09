from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

ROOT = Path(__file__).resolve().parents[1]

SRC = (
    ROOT / "docking/3W32/phase4C/ligands"
    / "EGFR_CAND_003/EGFR_CAND_003.sdf"
)

OUTDIR = (
    ROOT / "docking/3W32/protonation_sensitivity"
    / "EGFR_CAND_003_N3H"
)

OUT = OUTDIR / "EGFR_CAND_003_N3H.sdf"

N_INDEX = 8

mol = Chem.SDMolSupplier(
    str(SRC), removeHs=False
)[0]

if mol is None or not mol.GetNumConformers():
    raise RuntimeError("Original SDF invalid")

n = mol.GetAtomWithIdx(N_INDEX)

if not (
    n.GetSymbol() == "N"
    and n.GetFormalCharge() == 0
    and n.GetDegree() == 3
    and n.GetTotalNumHs(includeNeighbors=True) == 0
):
    raise RuntimeError("N3 identity check failed")

heavy_ids = [
    a.GetIdx() for a in mol.GetAtoms()
    if a.GetAtomicNum() > 1
]

original_xyz = np.array([
    list(mol.GetConformer().GetAtomPosition(i))
    for i in heavy_ids
])

rw = Chem.RWMol(mol)
n = rw.GetAtomWithIdx(N_INDEX)
n.SetFormalCharge(1)
n.SetNumExplicitHs(1)
n.SetNoImplicit(True)

charged = rw.GetMol()
Chem.SanitizeMol(charged)

charged = Chem.AddHs(charged, addCoords=True)

n = charged.GetAtomWithIdx(N_INDEX)
attached_h = [
    a.GetIdx() for a in n.GetNeighbors()
    if a.GetAtomicNum() == 1
]

if len(attached_h) != 1:
    raise RuntimeError("Expected exactly one N3-H")

if Chem.GetFormalCharge(charged) != 1:
    raise RuntimeError("Net charge is not +1")

if not AllChem.MMFFHasAllMoleculeParams(charged):
    raise RuntimeError("MMFF parameters unavailable")

properties = AllChem.MMFFGetMoleculeProperties(
    charged, mmffVariant="MMFF94s"
)

ff = AllChem.MMFFGetMoleculeForceField(
    charged, properties
)

for i in heavy_ids:
    ff.AddFixedPoint(i)

status = ff.Minimize(maxIts=500)

positions = np.array(
    ff.Positions(), dtype=float
).reshape(-1, 3)

conf = charged.GetConformer()

for i, xyz in enumerate(positions):
    conf.SetAtomPosition(
        i, tuple(float(v) for v in xyz)
    )

new_xyz = np.array([
    list(conf.GetAtomPosition(i))
    for i in heavy_ids
])

max_shift = float(np.max(
    np.linalg.norm(
        new_xyz - original_xyz, axis=1
    )
))

nh_distance = float(np.linalg.norm(
    positions[N_INDEX] -
    positions[attached_h[0]]
))

print("=" * 55)
print("CAND003 N3 PROTONATED STRUCTURE")
print("=" * 55)
print("MMFF convergence status:", status)
print("Formal charge:", Chem.GetFormalCharge(charged))
print("Formula:", CalcMolFormula(charged))
print("Heavy atoms:", len(heavy_ids))
print("N3 attached H:", len(attached_h))
print("N3-H distance:", round(nh_distance, 4), "A")
print("Maximum heavy atom shift:",
      round(max_shift, 6), "A")

if status != 0:
    raise RuntimeError("Hydrogen minimization not converged")

if max_shift > 0.001:
    raise RuntimeError("Heavy atom positions changed")

if not 0.85 <= nh_distance <= 1.20:
    raise RuntimeError("Unexpected N3-H bond length")

OUTDIR.mkdir(parents=True, exist_ok=True)

writer = Chem.SDWriter(str(OUT))
writer.write(charged)
writer.close()

print("\nPROTONATED SDF CREATED")
print("Output:", OUT)
print("Original files were not modified.")
print("No docking performed.")
