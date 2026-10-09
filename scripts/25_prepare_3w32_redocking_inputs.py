from pathlib import Path
from collections import defaultdict
import json
import shlex
import numpy as np

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.warning")


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

PDB_IN = ROOT / "docking/3W32/raw/3W32.pdb"
CCD_CIF = ROOT / "docking/3W32/ligand/W32.cif"
IDEAL_SDF = ROOT / "docking/3W32/ligand/W32_ideal.sdf"

RECEPTOR_OUT = (
    ROOT / "docking/3W32/receptor/3W32_clean_receptor.pdb"
)

LIGAND_PDB_OUT = (
    ROOT / "docking/3W32/ligand/W32_3W32_crystal.pdb"
)

LIGAND_SDF_OUT = (
    ROOT / "docking/3W32/ligand/W32_3W32_crystal_H.sdf"
)

REPORT_OUT = (
    ROOT / "docking/3W32/reports/3W32_input_preparation.json"
)

for p in [
    RECEPTOR_OUT.parent,
    LIGAND_PDB_OUT.parent,
    REPORT_OUT.parent,
]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def normalize_element_symbol(value):
    """
    Convert element symbols to RDKit-compatible capitalization.

    Examples:
        C  -> C
        N  -> N
        CL -> Cl
        Cl -> Cl
        BR -> Br
    """
    value = str(value).strip()

    if not value:
        return ""

    if len(value) == 1:
        return value.upper()

    return (
        value[0].upper()
        + value[1:].lower()
    )


def pdb_element(line):
    element = line[76:78].strip()

    if element:
        return normalize_element_symbol(
            element
        )

    atom_name = line[12:16].strip()

    letters = "".join(
        c for c in atom_name
        if c.isalpha()
    )

    return normalize_element_symbol(
        letters[:2]
    )


def pdb_xyz(line):
    return np.array(
        [
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ],
        dtype=float,
    )


def clean_value(x):
    if x in {None, ".", "?"}:
        return None
    return x


# ============================================================
# Generic simple CIF loop reader
# ============================================================

def read_cif_loops(path):
    lines = path.read_text().splitlines()
    loops = []

    i = 0

    while i < len(lines):

        if lines[i].strip() != "loop_":
            i += 1
            continue

        i += 1
        tags = []

        while (
            i < len(lines)
            and lines[i].strip().startswith("_")
        ):
            tags.append(
                lines[i].strip().split()[0]
            )
            i += 1

        if not tags:
            continue

        tokens = []

        while i < len(lines):

            s = lines[i].strip()

            if (
                s == "loop_"
                or s.startswith("data_")
                or s.startswith("_")
            ):
                break

            if s.startswith("#"):
                i += 1
                break

            if not s:
                i += 1
                continue

            tokens.extend(
                shlex.split(s, posix=True)
            )

            i += 1

        ncols = len(tags)

        if not tokens:
            continue

        if len(tokens) % ncols != 0:
            continue

        rows = []

        for j in range(
            0,
            len(tokens),
            ncols,
        ):
            row = dict(
                zip(
                    tags,
                    tokens[j:j+ncols],
                )
            )
            rows.append(row)

        loops.append((tags, rows))

    return loops


def find_loop(loops, required_tag):
    for tags, rows in loops:
        if required_tag in tags:
            return rows

    raise RuntimeError(
        f"Required CIF loop not found: "
        f"{required_tag}"
    )


# ============================================================
# Read raw 3W32
# ============================================================

lines = PDB_IN.read_text().splitlines()


# ============================================================
# 1. Prepare clean protein receptor
#
# Keep only ATOM protein coordinates.
# HETATM including W32/SO4/waters are excluded.
# Resolve alternate locations:
# blank > A > other.
# ============================================================

cryst1 = [
    line
    for line in lines
    if line.startswith("CRYST1")
]

protein_candidates = {}

for line in lines:

    if not line.startswith("ATOM"):
        continue

    chain = line[21].strip()
    resnum = line[22:26]
    icode = line[26]
    resname = line[17:20].strip()
    atom_name = line[12:16].strip()
    altloc = line[16].strip()

    try:
        occupancy = float(line[54:60])
    except Exception:
        occupancy = 0.0

    key = (
        chain,
        resnum,
        icode,
        resname,
        atom_name,
    )

    if altloc == "":
        priority = 2
    elif altloc == "A":
        priority = 1
    else:
        priority = 0

    candidate = (
        priority,
        occupancy,
        line,
    )

    if key not in protein_candidates:
        protein_candidates[key] = candidate

    elif candidate[:2] > protein_candidates[key][:2]:
        protein_candidates[key] = candidate


protein_lines = []

for value in protein_candidates.values():

    line = value[2]

    # clear altloc column
    line = (
        line[:16]
        + " "
        + line[17:]
    )

    protein_lines.append(line)


protein_lines.sort(
    key=lambda x: int(x[6:11])
)


with open(RECEPTOR_OUT, "w") as f:

    for line in cryst1:
        f.write(line + "\n")

    for line in protein_lines:
        f.write(line + "\n")

    f.write("TER\n")
    f.write("END\n")


protein_residues = set()

for line in protein_lines:
    protein_residues.add(
        (
            line[21].strip(),
            int(line[22:26]),
            line[17:20].strip(),
        )
    )


# ============================================================
# 2. Extract experimental W32 coordinates
# ============================================================

w32_groups = defaultdict(list)

for line in lines:

    if not line.startswith("HETATM"):
        continue

    if line[17:20].strip() != "W32":
        continue

    key = (
        line[21].strip(),
        int(line[22:26]),
        line[26].strip(),
    )

    w32_groups[key].append(line)


if not w32_groups:
    raise RuntimeError(
        "No W32 ligand found in 3W32."
    )


ligand_key = max(
    w32_groups,
    key=lambda x: len(w32_groups[x]),
)

raw_ligand = w32_groups[ligand_key]


# Resolve ligand altloc
ligand_candidates = {}

for line in raw_ligand:

    atom_name = line[12:16].strip()
    altloc = line[16].strip()

    try:
        occupancy = float(line[54:60])
    except Exception:
        occupancy = 0.0

    if altloc == "":
        priority = 2
    elif altloc == "A":
        priority = 1
    else:
        priority = 0

    candidate = (
        priority,
        occupancy,
        line,
    )

    if atom_name not in ligand_candidates:
        ligand_candidates[atom_name] = candidate

    elif candidate[:2] > ligand_candidates[atom_name][:2]:
        ligand_candidates[atom_name] = candidate


ligand_lines = [
    x[2]
    for x in ligand_candidates.values()
]

ligand_lines.sort(
    key=lambda x: int(x[6:11])
)


with open(LIGAND_PDB_OUT, "w") as f:

    for line in ligand_lines:

        line = (
            line[:16]
            + " "
            + line[17:]
        )

        f.write(line + "\n")

    f.write("END\n")


# Experimental heavy-atom coordinates by atom name
crystal_atoms = {}

for line in ligand_lines:

    element = pdb_element(line)

    if element == "H":
        continue

    atom_name = line[12:16].strip()

    crystal_atoms[atom_name] = {
        "element": element,
        "xyz": pdb_xyz(line),
    }


# ============================================================
# 3. Parse W32 CCD
# ============================================================

loops = read_cif_loops(CCD_CIF)

atom_rows = find_loop(
    loops,
    "_chem_comp_atom.atom_id",
)

bond_rows = find_loop(
    loops,
    "_chem_comp_bond.atom_id_1",
)


ccd_atoms = []

for row in atom_rows:

    atom_id = clean_value(
        row.get(
            "_chem_comp_atom.atom_id"
        )
    )

    element = clean_value(
        row.get(
            "_chem_comp_atom.type_symbol"
        )
    )

    if atom_id is None or element is None:
        continue

    element = normalize_element_symbol(
        element
    )

    if element in {"H", "D"}:
        continue

    charge_text = clean_value(
        row.get(
            "_chem_comp_atom.charge"
        )
    )

    try:
        charge = int(charge_text)
    except Exception:
        charge = 0

    aromatic_flag = clean_value(
        row.get(
            "_chem_comp_atom.pdbx_aromatic_flag"
        )
    )

    ccd_atoms.append(
        {
            "atom_id": atom_id,
            "element": element,
            "charge": charge,
            "aromatic": aromatic_flag == "Y",
        }
    )


# ============================================================
# 4. Exact atom-name mapping
# ============================================================

ccd_names = {
    x["atom_id"]
    for x in ccd_atoms
}

pdb_names = set(crystal_atoms)

missing_from_pdb = sorted(
    ccd_names - pdb_names
)

extra_in_pdb = sorted(
    pdb_names - ccd_names
)


if missing_from_pdb or extra_in_pdb:

    raise RuntimeError(
        "W32 CCD/PDB atom-name mismatch.\n"
        f"Missing from PDB: {missing_from_pdb}\n"
        f"Extra in PDB: {extra_in_pdb}"
    )


element_mismatches = []

for atom in ccd_atoms:

    atom_id = atom["atom_id"]

    if (
        atom["element"]
        != crystal_atoms[atom_id]["element"]
    ):
        element_mismatches.append(
            (
                atom_id,
                atom["element"],
                crystal_atoms[atom_id]["element"],
            )
        )


if element_mismatches:

    raise RuntimeError(
        "CCD/PDB element mismatch: "
        f"{element_mismatches}"
    )


# ============================================================
# 5. Build RDKit molecule directly from CCD topology
# ============================================================

rw = Chem.RWMol()

atom_index = {}

for item in ccd_atoms:

    atom = Chem.Atom(
        item["element"]
    )

    atom.SetFormalCharge(
        item["charge"]
    )

    atom.SetProp(
        "ccd_atom_id",
        item["atom_id"],
    )

    idx = rw.AddAtom(atom)

    atom_index[
        item["atom_id"]
    ] = idx


bond_type_map = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "AROM": Chem.BondType.AROMATIC,
    "DELO": Chem.BondType.AROMATIC,
}


heavy_bond_count = 0

for row in bond_rows:

    a1 = clean_value(
        row.get(
            "_chem_comp_bond.atom_id_1"
        )
    )

    a2 = clean_value(
        row.get(
            "_chem_comp_bond.atom_id_2"
        )
    )

    if (
        a1 not in atom_index
        or a2 not in atom_index
    ):
        # bond to hydrogen; skip for now
        continue

    order = clean_value(
        row.get(
            "_chem_comp_bond.value_order"
        )
    )

    aromatic_flag = clean_value(
        row.get(
            "_chem_comp_bond.pdbx_aromatic_flag"
        )
    )

    if aromatic_flag == "Y":
        bond_type = Chem.BondType.AROMATIC
    else:
        bond_type = bond_type_map.get(
            order,
            Chem.BondType.SINGLE,
        )

    rw.AddBond(
        atom_index[a1],
        atom_index[a2],
        bond_type,
    )

    if bond_type == Chem.BondType.AROMATIC:

        bond = rw.GetBondBetweenAtoms(
            atom_index[a1],
            atom_index[a2],
        )

        bond.SetIsAromatic(True)

        rw.GetAtomWithIdx(
            atom_index[a1]
        ).SetIsAromatic(True)

        rw.GetAtomWithIdx(
            atom_index[a2]
        ).SetIsAromatic(True)

    heavy_bond_count += 1


mol = rw.GetMol()


# ============================================================
# 6. Attach crystal coordinates
# ============================================================

conf = Chem.Conformer(
    mol.GetNumAtoms()
)

conf.Set3D(True)


for item in ccd_atoms:

    atom_id = item["atom_id"]
    idx = atom_index[atom_id]

    xyz = crystal_atoms[
        atom_id
    ]["xyz"]

    conf.SetAtomPosition(
        idx,
        (
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
        )
    )


mol.AddConformer(
    conf,
    assignId=True,
)


# ============================================================
# 7. Sanitize chemistry
# ============================================================

try:
    Chem.SanitizeMol(mol)

except Exception as exc:
    raise RuntimeError(
        "CCD-built W32 failed RDKit sanitization:\n"
        f"{exc}"
    )


# ============================================================
# 8. Compare chemistry against official ideal SDF
#
# Coordinates are NOT taken from ideal SDF.
# It is used only as a chemistry identity check.
# ============================================================

ideal_supplier = Chem.SDMolSupplier(
    str(IDEAL_SDF),
    removeHs=False,
)

ideal = ideal_supplier[0]

if ideal is None:
    raise RuntimeError(
        "Could not read W32_ideal.sdf"
    )


ideal_noh = Chem.RemoveHs(ideal)
built_noh = Chem.RemoveHs(mol)


try:
    ideal_inchikey = Chem.MolToInchiKey(
        ideal_noh
    )
except Exception:
    ideal_inchikey = None


try:
    built_inchikey = Chem.MolToInchiKey(
        built_noh
    )
except Exception:
    built_inchikey = None


ideal_charge = int(
    Chem.GetFormalCharge(
        ideal_noh
    )
)

built_charge = int(
    Chem.GetFormalCharge(
        built_noh
    )
)


# ============================================================
# 9. Add hydrogens to crystal-coordinate molecule
# ============================================================

mol_H = Chem.AddHs(
    mol,
    addCoords=True,
)


# Heavy atoms remain first after AddHs.
n_heavy = mol.GetNumAtoms()

heavy_displacements = []

conf_before = mol.GetConformer()
conf_after = mol_H.GetConformer()

for i in range(n_heavy):

    p1 = np.array(
        list(
            conf_before.GetAtomPosition(i)
        )
    )

    p2 = np.array(
        list(
            conf_after.GetAtomPosition(i)
        )
    )

    heavy_displacements.append(
        float(
            np.linalg.norm(
                p1 - p2
            )
        )
    )


heavy_rmsd = float(
    np.sqrt(
        np.mean(
            np.square(
                heavy_displacements
            )
        )
    )
)

max_heavy_shift = float(
    max(heavy_displacements)
)


# ============================================================
# 10. Save final ligand
# ============================================================

mol_H.SetProp(
    "_Name",
    "W32_3W32_crystal"
)

mol_H.SetProp(
    "source_coordinates",
    "PDB 3W32"
)

mol_H.SetProp(
    "chemical_topology",
    "RCSB CCD W32.cif"
)


writer = Chem.SDWriter(
    str(LIGAND_SDF_OUT)
)

writer.write(mol_H)
writer.close()


# ============================================================
# Ligand geometry
# ============================================================

xyz_array = np.array(
    [
        crystal_atoms[
            item["atom_id"]
        ]["xyz"]
        for item in ccd_atoms
    ]
)

center = xyz_array.mean(axis=0)

extent = (
    xyz_array.max(axis=0)
    - xyz_array.min(axis=0)
)


# ============================================================
# EGFR residue QC
# ============================================================

important_residues = [
    718, 719, 720, 721, 722,
    723, 724, 725, 726,
    743, 745, 762, 766,
    790, 791, 792, 793, 794,
    795, 796, 797, 798, 799, 800,
    835, 855,
]

observed_numbers = {
    resnum
    for chain, resnum, resname
    in protein_residues
    if chain == "A"
}

missing_important = [
    x
    for x in important_residues
    if x not in observed_numbers
]


# ============================================================
# Final status
# ============================================================

status = "PASS"

reasons = []

if len(crystal_atoms) != 39:
    status = "REVIEW_REQUIRED"
    reasons.append(
        "crystal heavy atom count != 39"
    )

if len(ccd_atoms) != 39:
    status = "REVIEW_REQUIRED"
    reasons.append(
        "CCD heavy atom count != 39"
    )

if missing_important:
    status = "REVIEW_REQUIRED"
    reasons.append(
        "important EGFR residue missing"
    )

if heavy_rmsd > 0.001:
    status = "REVIEW_REQUIRED"
    reasons.append(
        "heavy coordinates moved after AddHs"
    )

if (
    ideal_inchikey
    and built_inchikey
    and ideal_inchikey != built_inchikey
):
    status = "REVIEW_REQUIRED"
    reasons.append(
        "CCD-built ligand InChIKey "
        "differs from ideal SDF"
    )

if ideal_charge != built_charge:
    status = "REVIEW_REQUIRED"
    reasons.append(
        "formal charge differs from ideal SDF"
    )


# ============================================================
# Report
# ============================================================

report = {
    "pdb_id": "3W32",

    "receptor": {
        "protein_atoms":
            len(protein_lines),

        "protein_residues":
            len(protein_residues),

        "missing_important_residues":
            missing_important,

        "output":
            str(RECEPTOR_OUT),
    },

    "ligand": {
        "resname":
            "W32",

        "chain":
            ligand_key[0],

        "resnum":
            ligand_key[1],

        "crystal_heavy_atoms":
            len(crystal_atoms),

        "ccd_heavy_atoms":
            len(ccd_atoms),

        "heavy_bonds":
            heavy_bond_count,

        "atoms_after_H":
            mol_H.GetNumAtoms(),

        "formal_charge":
            built_charge,

        "built_inchikey":
            built_inchikey,

        "ideal_inchikey":
            ideal_inchikey,

        "center_A":
            center.tolist(),

        "extent_A":
            extent.tolist(),

        "heavy_coordinate_RMSD_after_H_A":
            heavy_rmsd,

        "max_heavy_shift_after_H_A":
            max_heavy_shift,

        "output_sdf":
            str(LIGAND_SDF_OUT),
    },

    "status":
        status,

    "review_reasons":
        reasons,
}


REPORT_OUT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


# ============================================================
# Terminal output
# ============================================================

print("\n========================================")
print("3W32 REDOCKING INPUT PREPARATION")
print("========================================")

print("\nRECEPTOR")
print("Protein atoms:", len(protein_lines))
print("Protein residues:", len(protein_residues))
print(
    "Missing important residues:",
    missing_important
)

print("\nW32 — CCD / CRYSTAL MAPPING")
print(
    "Crystal heavy atoms:",
    len(crystal_atoms)
)
print(
    "CCD heavy atoms:",
    len(ccd_atoms)
)
print(
    "CCD heavy-atom bonds:",
    heavy_bond_count
)
print(
    "Missing CCD atoms in crystal:",
    missing_from_pdb
)
print(
    "Extra crystal atoms:",
    extra_in_pdb
)
print(
    "Element mismatches:",
    element_mismatches
)

print("\nCHEMISTRY IDENTITY CHECK")
print(
    "CCD-built InChIKey:",
    built_inchikey
)
print(
    "Ideal-SDF InChIKey:",
    ideal_inchikey
)
print(
    "CCD-built formal charge:",
    built_charge
)
print(
    "Ideal-SDF formal charge:",
    ideal_charge
)

print("\nFINAL W32")
print(
    "Atoms after adding H:",
    mol_H.GetNumAtoms()
)
print(
    "Crystal center:",
    np.round(center, 3)
)
print(
    "Crystal extent:",
    np.round(extent, 3)
)
print(
    "Heavy-atom coordinate RMSD "
    "after adding H:",
    f"{heavy_rmsd:.6f} A"
)
print(
    "Maximum heavy-atom shift:",
    f"{max_heavy_shift:.6f} A"
)

print("\nGENERATED")
print(RECEPTOR_OUT)
print(LIGAND_PDB_OUT)
print(LIGAND_SDF_OUT)
print(REPORT_OUT)

if reasons:
    print("\nReview reasons:")
    for reason in reasons:
        print(" -", reason)

print("\n========================================")
print(
    "INPUT PREPARATION STATUS:",
    status
)
print("========================================")
