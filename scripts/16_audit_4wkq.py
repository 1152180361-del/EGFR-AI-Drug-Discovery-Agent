from pathlib import Path
from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

PDB_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "raw"
    / "4WKQ.pdb"
)

LIGAND_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "ligand"
    / "gefitinib_4WKQ_crystal.pdb"
)

CSX_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "CSX797_atoms.txt"
)

NEARBY_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "gefitinib_nearby_residues_5A.csv"
)

SUMMARY_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "4WKQ_structure_audit.json"
)


# =========================================
# Parse one PDB atom record
# =========================================

def parse_atom_line(line):

    record_name = line[0:6].strip()

    if record_name not in {
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
            else None
        )

        return {
            "record":
                record_name,

            "serial":
                int(
                    line[6:11]
                ),

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

            "x":
                float(
                    line[30:38]
                ),

            "y":
                float(
                    line[38:46]
                ),

            "z":
                float(
                    line[46:54]
                ),

            "occupancy":
                occupancy,

            "element":
                line[76:78].strip(),

            "raw_line":
                line.rstrip("\n"),
        }

    except Exception as error:

        print(
            "WARNING: failed to parse:"
        )

        print(
            line.rstrip()
        )

        print(
            error
        )

        return None


# =========================================
# Load PDB
# =========================================

if not PDB_FILE.exists():

    raise FileNotFoundError(
        f"PDB not found: {PDB_FILE}"
    )


with open(
    PDB_FILE,
    "r"
) as handle:

    pdb_lines = handle.readlines()


atoms = []


for line in pdb_lines:

    record = parse_atom_line(
        line
    )

    if record is not None:

        atoms.append(
            record
        )


print(
    "Parsed coordinate atoms:",
    len(atoms)
)


# =========================================
# Basic categories
# =========================================

water_names = {
    "HOH",
    "WAT",
}

protein_atoms = []

ligand_atoms = []

water_atoms = []

other_hetero_atoms = []


for atom in atoms:

    resname = atom[
        "resname"
    ]

    # Standard ATOM residues +
    # modified polymer residue CSX
    if (
        atom["record"] == "ATOM"
        or resname == "CSX"
    ):

        protein_atoms.append(
            atom
        )

    elif resname == "IRE":

        ligand_atoms.append(
            atom
        )

    elif resname in water_names:

        water_atoms.append(
            atom
        )

    else:

        other_hetero_atoms.append(
            atom
        )


# =========================================
# Protein residues
# =========================================

protein_residues = set()


for atom in protein_atoms:

    protein_residues.add(
        (
            atom["chain"],
            atom["resseq"],
            atom["icode"],
            atom["resname"],
        )
    )


# =========================================
# Hetero inventory
# =========================================

hetero_groups = defaultdict(
    list
)


for atom in atoms:

    if atom["record"] == "HETATM":

        key = (
            atom["resname"],
            atom["chain"],
            atom["resseq"],
            atom["icode"],
        )

        hetero_groups[
            key
        ].append(
            atom
        )


# =========================================
# Locate Gefitinib
# =========================================

if len(ligand_atoms) == 0:

    raise RuntimeError(
        "No IRE/gefitinib found."
    )


ligand_heavy_atoms = [
    atom
    for atom in ligand_atoms
    if atom["element"] != "H"
]


ligand_xyz = np.array(
    [
        [
            atom["x"],
            atom["y"],
            atom["z"],
        ]
        for atom
        in ligand_heavy_atoms
    ],
    dtype=float
)


ligand_center = (
    ligand_xyz.mean(
        axis=0
    )
)

ligand_min = (
    ligand_xyz.min(
        axis=0
    )
)

ligand_max = (
    ligand_xyz.max(
        axis=0
    )
)

ligand_extent = (
    ligand_max
    - ligand_min
)


# =========================================
# Extract crystallographic gefitinib pose
# =========================================

with open(
    LIGAND_FILE,
    "w"
) as handle:

    for atom in ligand_atoms:

        handle.write(
            atom[
                "raw_line"
            ]
            + "\n"
        )

    handle.write(
        "END\n"
    )


# =========================================
# Inspect CSX797
# =========================================

csx_atoms = [
    atom
    for atom in atoms
    if (
        atom["resname"] == "CSX"
        and atom["chain"] == "A"
        and atom["resseq"] == 797
    )
]


with open(
    CSX_FILE,
    "w"
) as handle:

    for atom in csx_atoms:

        handle.write(
            atom[
                "raw_line"
            ]
            + "\n"
        )


# =========================================
# Residues within 5 A of gefitinib
# =========================================

residue_atoms = defaultdict(
    list
)


for atom in protein_atoms:

    key = (
        atom["chain"],
        atom["resseq"],
        atom["icode"],
        atom["resname"],
    )

    residue_atoms[
        key
    ].append(
        atom
    )


nearby_records = []


for key, residue in (
    residue_atoms.items()
):

    coordinates = np.array(
        [
            [
                atom["x"],
                atom["y"],
                atom["z"],
            ]
            for atom in residue
        ],
        dtype=float
    )


    # All protein atom vs ligand atom
    # pair distances
    differences = (
        coordinates[:, None, :]
        - ligand_xyz[None, :, :]
    )

    distances = np.sqrt(
        np.sum(
            differences ** 2,
            axis=2
        )
    )


    minimum_distance = float(
        distances.min()
    )


    if minimum_distance <= 5.0:

        (
            chain,
            resseq,
            icode,
            resname,
        ) = key


        nearby_records.append(
            {
                "chain":
                    chain,

                "residue_number":
                    resseq,

                "insertion_code":
                    icode,

                "residue_name":
                    resname,

                "minimum_distance_A":
                    minimum_distance,
            }
        )


nearby_df = pd.DataFrame(
    nearby_records
)


nearby_df = (
    nearby_df
    .sort_values(
        "minimum_distance_A"
    )
    .reset_index(
        drop=True
    )
)


nearby_df.to_csv(
    NEARBY_FILE,
    index=False
)


# =========================================
# Count water molecules
# =========================================

water_residues = set()


for atom in water_atoms:

    water_residues.add(
        (
            atom["chain"],
            atom["resseq"],
            atom["icode"],
        )
    )


# =========================================
# Other hetero groups
# =========================================

other_groups = []


for (
    resname,
    chain,
    resseq,
    icode
), group_atoms in hetero_groups.items():

    if resname in water_names:
        continue

    other_groups.append(
        {
            "resname":
                resname,

            "chain":
                chain,

            "resseq":
                resseq,

            "icode":
                icode,

            "atom_count":
                len(
                    group_atoms
                ),
        }
    )


other_groups = sorted(
    other_groups,
    key=lambda item: (
        item["resseq"],
        item["resname"],
    )
)


# =========================================
# Summary
# =========================================

summary = {

    "pdb_id":
        "4WKQ",

    "coordinate_atoms":
        len(atoms),

    "protein_atoms_including_CSX":
        len(
            protein_atoms
        ),

    "protein_residues_including_CSX":
        len(
            protein_residues
        ),

    "gefitinib_atoms":
        len(
            ligand_atoms
        ),

    "gefitinib_heavy_atoms":
        len(
            ligand_heavy_atoms
        ),

    "gefitinib_center_xyz":
        [
            float(value)
            for value
            in ligand_center
        ],

    "gefitinib_extent_xyz":
        [
            float(value)
            for value
            in ligand_extent
        ],

    "water_molecules":
        len(
            water_residues
        ),

    "CSX797_atom_count":
        len(
            csx_atoms
        ),

    "other_hetero_groups":
        other_groups,

    "residues_within_5A":
        len(
            nearby_df
        ),
}


with open(
    SUMMARY_FILE,
    "w"
) as handle:

    json.dump(
        summary,
        handle,
        indent=4
    )


# =========================================
# Terminal report
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 3.1 — 4WKQ STRUCTURE AUDIT"
)

print(
    "================================="
)


print(
    "Protein residues "
    "(including CSX):",
    len(
        protein_residues
    )
)


print(
    "Gefitinib atoms:",
    len(
        ligand_atoms
    )
)


print(
    "Gefitinib heavy atoms:",
    len(
        ligand_heavy_atoms
    )
)


print(
    "\nGefitinib center:"
)

print(
    "X =",
    round(
        ligand_center[0],
        3
    )
)

print(
    "Y =",
    round(
        ligand_center[1],
        3
    )
)

print(
    "Z =",
    round(
        ligand_center[2],
        3
    )
)


print(
    "\nGefitinib molecular extent:"
)

print(
    "X =",
    round(
        ligand_extent[0],
        3
    )
)

print(
    "Y =",
    round(
        ligand_extent[1],
        3
    )
)

print(
    "Z =",
    round(
        ligand_extent[2],
        3
    )
)


print(
    "\nWater molecules:",
    len(
        water_residues
    )
)


print(
    "\nCSX A797 atoms:",
    len(
        csx_atoms
    )
)


if csx_atoms:

    print(
        "CSX797 atom names:"
    )

    print(
        [
            atom[
                "atom_name"
            ]
            for atom
            in csx_atoms
        ]
    )


print(
    "\nNon-water HET groups:"
)


for item in other_groups:

    print(
        item
    )


print(
    "\nResidues within 5 A "
    "of Gefitinib:"
)


print(
    nearby_df.to_string(
        index=False
    )
)


print(
    "\nSaved crystal ligand:"
)

print(
    LIGAND_FILE
)


print(
    "\nSaved CSX797 atom records:"
)

print(
    CSX_FILE
)


print(
    "\nSaved nearby residues:"
)

print(
    NEARBY_FILE
)


print(
    "\nSaved audit summary:"
)

print(
    SUMMARY_FILE
)
