from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd

from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

PDB_FILE = (
    ROOT
    / "docking"
    / "receptor_screen"
    / "2ITY"
    / "2ITY.pdb"
)

CIF_FILE = (
    ROOT
    / "docking"
    / "receptor_screen"
    / "2ITY"
    / "2ITY.cif"
)

REPORT_DIR = (
    ROOT
    / "docking"
    / "receptor_screen"
    / "2ITY"
    / "reports"
)

LIGAND_DIR = (
    ROOT
    / "docking"
    / "receptor_screen"
    / "2ITY"
    / "ligand"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LIGAND_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LIGAND_FILE = (
    LIGAND_DIR
    / "gefitinib_2ITY_crystal.pdb"
)

POCKET_FILE = (
    REPORT_DIR
    / "2ITY_gefitinib_nearby_residues_5A.csv"
)

MISSING_RESIDUES_FILE = (
    REPORT_DIR
    / "2ITY_missing_residues.csv"
)

MISSING_ATOMS_FILE = (
    REPORT_DIR
    / "2ITY_missing_atoms.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "2ITY_structure_audit.json"
)


# =========================================
# Settings
# =========================================

PROTEIN_CHAIN = "A"
LIGAND_NAME = "IRE"
POCKET_DISTANCE = 5.0


# Critical structural regions
#
# These are used as an additional
# completeness check because completely
# missing residues have no coordinates and
# therefore cannot appear in a distance-
# based pocket search.
# =========================================

CRITICAL_REGIONS = {

    "P_loop_718_726":
        set(
            range(
                718,
                727
            )
        ),

    "beta3_alphaC_742_766":
        set(
            range(
                742,
                767
            )
        ),

    "hinge_Cys797_788_800":
        set(
            range(
                788,
                801
            )
        ),

    "activation_DFG_region_835_858":
        set(
            range(
                835,
                859
            )
        ),
}


# =========================================
# Helpers
# =========================================

def parse_atom_line(line):

    record = line[0:6].strip()

    if record not in {
        "ATOM",
        "HETATM",
    }:
        return None


    try:

        return {

            "record":
                record,

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

            "element":
                line[76:78].strip(),

            "raw_line":
                line.rstrip(
                    "\n"
                ),
        }

    except Exception:

        return None


def as_list(value):

    if value is None:
        return []

    if isinstance(
        value,
        list
    ):
        return value

    return [
        value
    ]


def cif_column(
    cif,
    key
):

    return as_list(
        cif.get(
            key,
            []
        )
    )


def safe_int(value):

    try:

        return int(
            str(
                value
            )
        )

    except Exception:

        return None


# =========================================
# File checks
# =========================================

for path in [
    PDB_FILE,
    CIF_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# =========================================
# Parse PDB coordinates
# =========================================

atoms = []


with open(
    PDB_FILE,
    "r"
) as handle:

    pdb_lines = handle.readlines()


for line in pdb_lines:

    atom = parse_atom_line(
        line
    )

    if atom is not None:

        atoms.append(
            atom
        )


print(
    "Coordinate atoms:",
    len(
        atoms
    )
)


# =========================================
# Protein atoms
# =========================================

protein_atoms = [
    atom
    for atom in atoms
    if (
        atom["record"] == "ATOM"
        and atom["chain"]
        == PROTEIN_CHAIN
    )
]


protein_residues = set(
    (
        atom["chain"],
        atom["resseq"],
        atom["icode"],
        atom["resname"],
    )
    for atom in protein_atoms
)


coordinate_residue_numbers = set(
    atom[
        "resseq"
    ]
    for atom
    in protein_atoms
)


# =========================================
# Locate Gefitinib / IRE
# =========================================

ligand_atoms = [
    atom
    for atom in atoms
    if (
        atom["resname"]
        == LIGAND_NAME
    )
]


if not ligand_atoms:

    raise RuntimeError(
        "IRE/gefitinib not found."
    )


ligand_heavy_atoms = [
    atom
    for atom in ligand_atoms
    if atom[
        "element"
    ] != "H"
]


ligand_xyz = np.asarray(
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


ligand_extent = (
    ligand_xyz.max(
        axis=0
    )
    -
    ligand_xyz.min(
        axis=0
    )
)


# =========================================
# Extract crystal ligand
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
# Actual 5 A ligand pocket
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

    coordinates = np.asarray(
        [
            [
                atom["x"],
                atom["y"],
                atom["z"],
            ]
            for atom
            in residue
        ],
        dtype=float
    )


    differences = (
        coordinates[:, None, :]
        -
        ligand_xyz[None, :, :]
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


    if (
        minimum_distance
        <= POCKET_DISTANCE
    ):

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
    POCKET_FILE,
    index=False
)


actual_pocket_residues = set(
    nearby_df[
        "residue_number"
    ].astype(
        int
    )
)


# =========================================
# Hetero inventory
# =========================================

hetero_groups = defaultdict(
    list
)


for atom in atoms:

    if atom[
        "record"
    ] != "HETATM":

        continue


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


hetero_summary = []


for key, group in (
    hetero_groups.items()
):

    (
        resname,
        chain,
        resseq,
        icode,
    ) = key


    hetero_summary.append(
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
                    group
                ),
        }
    )


hetero_summary = sorted(
    hetero_summary,
    key=lambda item: (
        item[
            "resname"
        ],
        item[
            "resseq"
        ],
    )
)


# =========================================
# Water count
# =========================================

water_residues = set()


for atom in atoms:

    if atom[
        "resname"
    ] in {
        "HOH",
        "WAT",
    }:

        water_residues.add(
            (
                atom["chain"],
                atom["resseq"],
                atom["icode"],
            )
        )


# =========================================
# Inspect Cys797
# =========================================

cys797_atoms = [
    atom
    for atom in protein_atoms
    if (
        atom["resseq"] == 797
    )
]


cys797_names = [
    atom[
        "atom_name"
    ]
    for atom
    in cys797_atoms
]


cys797_resnames = sorted(
    set(
        atom[
            "resname"
        ]
        for atom
        in cys797_atoms
    )
)


# =========================================
# Load mmCIF
# =========================================

cif = MMCIF2Dict(
    str(
        CIF_FILE
    )
)


# =========================================
# Official missing residues
# =========================================

res_chains = cif_column(
    cif,
    "_pdbx_unobs_or_zero_occ_residues.auth_asym_id"
)

res_names = cif_column(
    cif,
    "_pdbx_unobs_or_zero_occ_residues.auth_comp_id"
)

res_numbers = cif_column(
    cif,
    "_pdbx_unobs_or_zero_occ_residues.auth_seq_id"
)


n_residue_entries = max(
    len(
        res_chains
    ),
    len(
        res_names
    ),
    len(
        res_numbers
    ),
)


missing_residue_records = []


for i in range(
    n_residue_entries
):

    chain = (
        res_chains[i]
        if i
        < len(
            res_chains
        )
        else ""
    )

    name = (
        res_names[i]
        if i
        < len(
            res_names
        )
        else ""
    )

    number = safe_int(
        res_numbers[i]
        if i
        < len(
            res_numbers
        )
        else None
    )


    if (
        chain
        != PROTEIN_CHAIN
    ):

        continue


    if number is None:

        continue


    region_hits = [
        region_name
        for (
            region_name,
            region_numbers
        ) in CRITICAL_REGIONS.items()
        if number
        in region_numbers
    ]


    missing_residue_records.append(
        {
            "chain":
                chain,

            "residue_name":
                name,

            "residue_number":
                number,

            "is_actual_5A_pocket":
                (
                    number
                    in actual_pocket_residues
                ),

            "critical_region":
                ";".join(
                    region_hits
                ),
        }
    )


missing_residues_df = pd.DataFrame(
    missing_residue_records
)


missing_residues_df.to_csv(
    MISSING_RESIDUES_FILE,
    index=False
)


# =========================================
# Official missing atoms
# =========================================

atom_chains = cif_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.auth_asym_id"
)

atom_resnames = cif_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.auth_comp_id"
)

atom_resnums = cif_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.auth_seq_id"
)

atom_names = cif_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.auth_atom_id"
)


n_atom_entries = max(
    len(
        atom_chains
    ),
    len(
        atom_resnames
    ),
    len(
        atom_resnums
    ),
    len(
        atom_names
    ),
)


missing_atom_records = []


for i in range(
    n_atom_entries
):

    chain = (
        atom_chains[i]
        if i
        < len(
            atom_chains
        )
        else ""
    )

    residue_name = (
        atom_resnames[i]
        if i
        < len(
            atom_resnames
        )
        else ""
    )

    residue_number = safe_int(
        atom_resnums[i]
        if i
        < len(
            atom_resnums
        )
        else None
    )

    missing_atom_name = (
        atom_names[i]
        if i
        < len(
            atom_names
        )
        else ""
    )


    if (
        chain
        != PROTEIN_CHAIN
    ):

        continue


    if residue_number is None:

        continue


    region_hits = [
        region_name
        for (
            region_name,
            region_numbers
        ) in CRITICAL_REGIONS.items()
        if residue_number
        in region_numbers
    ]


    missing_atom_records.append(
        {
            "chain":
                chain,

            "residue_name":
                residue_name,

            "residue_number":
                residue_number,

            "atom_name":
                missing_atom_name,

            "is_actual_5A_pocket":
                (
                    residue_number
                    in actual_pocket_residues
                ),

            "critical_region":
                ";".join(
                    region_hits
                ),
        }
    )


missing_atoms_df = pd.DataFrame(
    missing_atom_records
)


missing_atoms_df.to_csv(
    MISSING_ATOMS_FILE,
    index=False
)


# =========================================
# Critical-region completeness
# =========================================

critical_missing_residues = []


for record in (
    missing_residue_records
):

    if record[
        "critical_region"
    ]:

        critical_missing_residues.append(
            record
        )


actual_pocket_missing_atoms = []


for record in (
    missing_atom_records
):

    if record[
        "is_actual_5A_pocket"
    ]:

        actual_pocket_missing_atoms.append(
            record
        )


actual_pocket_missing_residues = []


for record in (
    missing_residue_records
):

    if record[
        "is_actual_5A_pocket"
    ]:

        actual_pocket_missing_residues.append(
            record
        )


# =========================================
# Explicit presence checks
# =========================================

important_residues = {
    718,
    719,
    720,
    721,
    722,
    723,
    724,
    725,
    726,
    743,
    745,
    762,
    766,
    790,
    791,
    792,
    793,
    794,
    795,
    796,
    797,
    800,
    835,
    855,
}


important_presence = {
    str(
        residue_number
    ):
        (
            residue_number
            in coordinate_residue_numbers
        )

    for residue_number
    in sorted(
        important_residues
    )
}


# =========================================
# Decision
# =========================================

problems = []


if (
    critical_missing_residues
):

    problems.append(
        "missing residues in critical ATP-site region"
    )


if (
    actual_pocket_missing_atoms
):

    problems.append(
        "missing atoms in actual 5A ligand pocket"
    )


if (
    actual_pocket_missing_residues
):

    problems.append(
        "missing residues in actual 5A ligand pocket"
    )


if (
    cys797_resnames
    != [
        "CYS"
    ]
):

    problems.append(
        "Cys797 is not standard CYS"
    )


if problems:

    status = (
        "REVIEW_REQUIRED"
    )

else:

    status = (
        "PASS"
    )


# =========================================
# Save summary
# =========================================

summary = {

    "pdb_id":
        "2ITY",

    "protein_chain":
        PROTEIN_CHAIN,

    "protein_residue_count":
        len(
            protein_residues
        ),

    "ligand_name":
        LIGAND_NAME,

    "ligand_atom_count":
        len(
            ligand_atoms
        ),

    "ligand_heavy_atom_count":
        len(
            ligand_heavy_atoms
        ),

    "ligand_center_xyz":
        [
            float(x)
            for x
            in ligand_center
        ],

    "ligand_extent_xyz":
        [
            float(x)
            for x
            in ligand_extent
        ],

    "water_molecules":
        len(
            water_residues
        ),

    "Cys797_residue_names":
        cys797_resnames,

    "Cys797_atom_names":
        cys797_names,

    "actual_5A_pocket_residues":
        sorted(
            actual_pocket_residues
        ),

    "critical_missing_residues":
        critical_missing_residues,

    "actual_pocket_missing_residues":
        actual_pocket_missing_residues,

    "actual_pocket_missing_atoms":
        actual_pocket_missing_atoms,

    "important_residue_presence":
        important_presence,

    "hetero_groups":
        hetero_summary,

    "status":
        status,

    "problems":
        problems,
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
    "PHASE 3.3 — 2ITY STRUCTURE AUDIT"
)

print(
    "================================="
)


print(
    "Protein residues:",
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
    [
        round(
            float(value),
            3
        )
        for value
        in ligand_center
    ]
)


print(
    "\nGefitinib extent:"
)

print(
    [
        round(
            float(value),
            3
        )
        for value
        in ligand_extent
    ]
)


print(
    "\nWater molecules:",
    len(
        water_residues
    )
)


print(
    "\nCys797 residue name:"
)

print(
    cys797_resnames
)


print(
    "Cys797 atoms:"
)

print(
    cys797_names
)


print(
    "\nActual residues within 5 A "
    "of gefitinib:"
)

print(
    nearby_df.to_string(
        index=False
    )
)


print(
    "\n---------------------------------"
)

print(
    "COMPLETENESS QC"
)

print(
    "---------------------------------"
)


print(
    "\nMissing residues in critical "
    "ATP-site regions:",
    len(
        critical_missing_residues
    )
)


if critical_missing_residues:

    print(
        critical_missing_residues
    )


print(
    "\nMissing residues in actual "
    "5 A pocket:",
    len(
        actual_pocket_missing_residues
    )
)


if actual_pocket_missing_residues:

    print(
        actual_pocket_missing_residues
    )


print(
    "\nMissing atoms in actual "
    "5 A pocket:",
    len(
        actual_pocket_missing_atoms
    )
)


if actual_pocket_missing_atoms:

    print(
        actual_pocket_missing_atoms
    )


print(
    "\nImportant residue presence:"
)


for residue_number in sorted(
    important_residues
):

    print(
        residue_number,
        important_presence[
            str(
                residue_number
            )
        ]
    )


print(
    "\nNon-water hetero groups:"
)


for group in hetero_summary:

    if group[
        "resname"
    ] not in {
        "HOH",
        "WAT",
    }:

        print(
            group
        )


print(
    "\n================================="
)

print(
    "2ITY RECEPTOR STATUS:",
    status
)

print(
    "================================="
)


if problems:

    print(
        "Problems:"
    )

    for problem in problems:

        print(
            "-",
            problem
        )


print(
    "\nSaved ligand:"
)

print(
    LIGAND_FILE
)


print(
    "\nSaved pocket report:"
)

print(
    POCKET_FILE
)


print(
    "\nSaved missing residue report:"
)

print(
    MISSING_RESIDUES_FILE
)


print(
    "\nSaved missing atom report:"
)

print(
    MISSING_ATOMS_FILE
)


print(
    "\nSaved structure audit:"
)

print(
    SUMMARY_FILE
)
