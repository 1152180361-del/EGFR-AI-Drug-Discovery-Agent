from pathlib import Path
from collections import defaultdict
import json

import numpy as np

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

RAW_PDB = (
    ROOT
    / "docking"
    / "2ITY"
    / "raw"
    / "2ITY.pdb"
)

RECEPTOR_FILE = (
    ROOT
    / "docking"
    / "2ITY"
    / "receptor"
    / "2ITY_clean_receptor.pdb"
)

LIGAND_PDB = (
    ROOT
    / "docking"
    / "2ITY"
    / "ligand"
    / "gefitinib_2ITY_crystal.pdb"
)

LIGAND_SDF = (
    ROOT
    / "docking"
    / "2ITY"
    / "ligand"
    / "gefitinib_2ITY_crystal_H.sdf"
)

REPORT_FILE = (
    ROOT
    / "docking"
    / "2ITY"
    / "reports"
    / "2ITY_redocking_input_preparation.json"
)

for path in [
    RECEPTOR_FILE.parent,
    LIGAND_PDB.parent,
    REPORT_FILE.parent,
]:
    path.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================
# Gefitinib reference chemistry
#
# RCSB CCD ligand IRE
# =========================================

GEFITINIB_SMILES = (
    "COc1cc2ncnc("
    "Nc3ccc(F)c(Cl)c3"
    ")c2cc1OCCCN4CCOCC4"
)

EXPECTED_INCHIKEY = (
    "XGALLCVXEZPNRQ-UHFFFAOYSA-N"
)


# =========================================
# PDB parser
# =========================================

def parse_atom_line(line):

    record = line[0:6].strip()

    if record not in {
        "ATOM",
        "HETATM",
    }:
        return None

    try:

        occupancy_text = (
            line[54:60].strip()
        )

        occupancy = (
            float(occupancy_text)
            if occupancy_text
            else 0.0
        )

        return {
            "record":
                record,

            "atom_name":
                line[12:16].strip(),

            "altloc":
                line[16].strip(),

            "resname":
                line[17:20].strip(),

            "chain":
                (
                    line[21].strip()
                    or "_"
                ),

            "resseq":
                int(
                    line[22:26]
                ),

            "icode":
                line[26].strip(),

            "occupancy":
                occupancy,

            "element":
                line[76:78].strip(),

            "line":
                line.rstrip("\n"),
        }

    except Exception:

        return None


def clear_altloc(line):

    chars = list(
        line.ljust(80)
    )

    chars[16] = " "

    return "".join(
        chars
    ).rstrip()


# =========================================
# Load structure
# =========================================

if not RAW_PDB.exists():

    raise FileNotFoundError(
        RAW_PDB
    )


with open(
    RAW_PDB,
    "r"
) as handle:

    lines = handle.readlines()


# =========================================
# Extract gefitinib
# =========================================

ligand_lines = []


for line in lines:

    atom = parse_atom_line(
        line
    )

    if atom is None:
        continue

    if (
        atom["record"] == "HETATM"
        and atom["resname"] == "IRE"
    ):

        ligand_lines.append(
            atom["line"]
        )


if not ligand_lines:

    raise RuntimeError(
        "IRE/gefitinib not found."
    )


with open(
    LIGAND_PDB,
    "w"
) as handle:

    for line in ligand_lines:

        handle.write(
            line
            + "\n"
        )

    handle.write(
        "END\n"
    )


# =========================================
# Build clean receptor
#
# Keep only chain A protein ATOM records.
# All waters and ligand are removed.
# =========================================

protein_candidates = []


for line_number, line in enumerate(
    lines
):

    atom = parse_atom_line(
        line
    )

    if atom is None:
        continue


    if atom["record"] != "ATOM":
        continue


    if atom["chain"] != "A":
        continue


    protein_candidates.append(
        {
            "line_number":
                line_number,

            "atom":
                atom,

            "line":
                atom["line"],
        }
    )


# =========================================
# Resolve alternate locations
# =========================================

atom_groups = defaultdict(
    list
)


for item in protein_candidates:

    atom = item["atom"]

    key = (
        atom["chain"],
        atom["resseq"],
        atom["icode"],
        atom["resname"],
        atom["atom_name"],
    )

    atom_groups[
        key
    ].append(
        item
    )


def altloc_rank(atom):

    if atom["altloc"] == "":
        return 3

    if atom["altloc"] == "A":
        return 2

    return 1


selected = []

altloc_removed = 0


for key, group in atom_groups.items():

    best = max(
        group,
        key=lambda item: (
            altloc_rank(
                item["atom"]
            ),
            item[
                "atom"
            ][
                "occupancy"
            ],
        )
    )

    altloc_removed += (
        len(group)
        - 1
    )

    selected.append(
        {
            "line_number":
                best["line_number"],

            "line":
                clear_altloc(
                    best["line"]
                ),
        }
    )


selected.sort(
    key=lambda item:
        item[
            "line_number"
        ]
)


with open(
    RECEPTOR_FILE,
    "w"
) as handle:

    for item in selected:

        handle.write(
            item["line"]
            + "\n"
        )

    handle.write(
        "TER\n"
    )

    handle.write(
        "END\n"
    )


# =========================================
# Read crystal ligand coordinates
#
# PDB has coordinates but bond orders are
# not trusted as the chemical definition.
# =========================================

crystal_mol = (
    Chem.MolFromPDBFile(
        str(
            LIGAND_PDB
        ),
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
)


if crystal_mol is None:

    raise RuntimeError(
        "RDKit failed to read "
        "crystal gefitinib."
    )


# =========================================
# Official chemical template
# =========================================

template = Chem.MolFromSmiles(
    GEFITINIB_SMILES
)


if template is None:

    raise RuntimeError(
        "Failed to parse "
        "gefitinib template."
    )


template_heavy_atoms = (
    template.GetNumHeavyAtoms()
)

crystal_heavy_atoms = (
    crystal_mol.GetNumHeavyAtoms()
)


if (
    template_heavy_atoms
    != crystal_heavy_atoms
):

    raise RuntimeError(
        "Heavy-atom count mismatch: "
        f"template={template_heavy_atoms}, "
        f"crystal={crystal_heavy_atoms}"
    )


# =========================================
# Transfer bond orders from reference
# chemistry to crystal coordinates
# =========================================

try:

    ligand = (
        AllChem
        .AssignBondOrdersFromTemplate(
            template,
            crystal_mol
        )
    )

except Exception as error:

    raise RuntimeError(
        "Bond-order assignment failed: "
        + str(error)
    )


Chem.SanitizeMol(
    ligand
)


# =========================================
# Record original heavy atom coordinates
# =========================================

before_conf = (
    ligand.GetConformer()
)


before_xyz = np.asarray(
    [
        list(
            before_conf.GetAtomPosition(i)
        )
        for i in range(
            ligand.GetNumAtoms()
        )
    ],
    dtype=float
)


# =========================================
# Add explicit hydrogens
#
# Heavy atom coordinates are NOT optimized.
# =========================================

ligand_H = Chem.AddHs(
    ligand,
    addCoords=True
)


after_conf = (
    ligand_H.GetConformer()
)


after_heavy_xyz = np.asarray(
    [
        list(
            after_conf.GetAtomPosition(i)
        )
        for i in range(
            ligand.GetNumAtoms()
        )
    ],
    dtype=float
)


coordinate_difference = (
    after_heavy_xyz
    - before_xyz
)


heavy_atom_coordinate_rmsd = float(
    np.sqrt(
        np.mean(
            np.sum(
                coordinate_difference ** 2,
                axis=1
            )
        )
    )
)


# =========================================
# Chemical identity QC
# =========================================

observed_inchikey = (
    Chem.MolToInchiKey(
        ligand_H
    )
)


formal_charge = int(
    Chem.GetFormalCharge(
        ligand_H
    )
)


# =========================================
# Save crystal-pose SDF
# =========================================

writer = Chem.SDWriter(
    str(
        LIGAND_SDF
    )
)

writer.write(
    ligand_H
)

writer.close()


# =========================================
# Final receptor QC
# =========================================

final_atom_count = 0

final_residues = set()

for item in selected:

    atom = parse_atom_line(
        item["line"]
    )

    if atom is None:
        continue

    final_atom_count += 1

    final_residues.add(
        (
            atom["chain"],
            atom["resseq"],
            atom["resname"],
        )
    )


# =========================================
# Hard QC
# =========================================

if (
    observed_inchikey
    != EXPECTED_INCHIKEY
):

    raise RuntimeError(
        "Gefitinib InChIKey mismatch: "
        f"{observed_inchikey}"
    )


if formal_charge != 0:

    raise RuntimeError(
        "Unexpected Gefitinib "
        f"formal charge: {formal_charge}"
    )


if (
    heavy_atom_coordinate_rmsd
    > 1e-4
):

    raise RuntimeError(
        "Crystal heavy atom coordinates "
        "changed unexpectedly."
    )


# =========================================
# Report
# =========================================

report = {

    "pdb_id":
        "2ITY",

    "rdkit_version":
        rdBase.rdkitVersion,

    "receptor": {

        "chain":
            "A",

        "atom_count":
            final_atom_count,

        "residue_count":
            len(
                final_residues
            ),

        "alternate_location_atoms_removed":
            altloc_removed,

        "hetero_atoms_removed":
            True,

        "waters_removed":
            True,
    },

    "ligand": {

        "name":
            "Gefitinib",

        "pdb_code":
            "IRE",

        "template_smiles":
            GEFITINIB_SMILES,

        "expected_inchikey":
            EXPECTED_INCHIKEY,

        "observed_inchikey":
            observed_inchikey,

        "formal_charge":
            formal_charge,

        "template_heavy_atoms":
            template_heavy_atoms,

        "crystal_heavy_atoms":
            crystal_heavy_atoms,

        "atoms_after_adding_H":
            ligand_H.GetNumAtoms(),

        "heavy_atom_coordinate_rmsd_after_H":
            heavy_atom_coordinate_rmsd,
    },

    "status":
        "PASS",
}


with open(
    REPORT_FILE,
    "w"
) as handle:

    json.dump(
        report,
        handle,
        indent=4
    )


# =========================================
# Terminal output
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 3.4A — REDOCKING INPUT PREP"
)

print(
    "================================="
)


print(
    "RDKit version:",
    rdBase.rdkitVersion
)


print(
    "\n--- Receptor ---"
)

print(
    "Protein atoms:",
    final_atom_count
)

print(
    "Protein residues:",
    len(
        final_residues
    )
)

print(
    "AltLoc atoms removed:",
    altloc_removed
)


print(
    "\n--- Gefitinib ---"
)

print(
    "Crystal heavy atoms:",
    crystal_heavy_atoms
)

print(
    "Template heavy atoms:",
    template_heavy_atoms
)

print(
    "Atoms after adding H:",
    ligand_H.GetNumAtoms()
)

print(
    "Formal charge:",
    formal_charge
)

print(
    "InChIKey:",
    observed_inchikey
)

print(
    "Expected InChIKey:",
    EXPECTED_INCHIKEY
)

print(
    "Heavy-atom coordinate RMSD "
    "after adding H:",
    heavy_atom_coordinate_rmsd
)


print(
    "\nChemical identity QC: PASS"
)

print(
    "Crystal-coordinate preservation: PASS"
)

print(
    "Input preparation: PASS"
)


print(
    "\nSaved clean receptor:"
)

print(
    RECEPTOR_FILE
)


print(
    "\nSaved crystal Gefitinib PDB:"
)

print(
    LIGAND_PDB
)


print(
    "\nSaved protonated Gefitinib SDF:"
)

print(
    LIGAND_SDF
)


print(
    "\nSaved preparation report:"
)

print(
    REPORT_FILE
)
