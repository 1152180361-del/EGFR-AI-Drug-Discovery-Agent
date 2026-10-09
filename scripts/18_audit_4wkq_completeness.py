from pathlib import Path
import json

import pandas as pd

from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

MMCIF_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "raw"
    / "4WKQ.cif"
)

CLEAN_RECEPTOR = (
    ROOT
    / "docking"
    / "4WKQ"
    / "receptor"
    / "4WKQ_nativeCys_clean.pdb"
)

POCKET_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "gefitinib_nearby_residues_5A.csv"
)

MISSING_RESIDUES_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "4WKQ_missing_residues.csv"
)

MISSING_ATOMS_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "4WKQ_missing_atoms.csv"
)

SUMMARY_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "4WKQ_completeness_audit.json"
)


# =========================================
# Basic file checks
# =========================================

for file_path in [
    MMCIF_FILE,
    CLEAN_RECEPTOR,
    POCKET_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing required file: "
            f"{file_path}"
        )


# =========================================
# Helpers
# =========================================

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


def get_column(
    cif,
    key
):

    if key not in cif:
        return []

    return as_list(
        cif[key]
    )


def safe_int(value):

    try:
        return int(
            str(value)
        )

    except Exception:
        return None


# =========================================
# Load mmCIF
# =========================================

print(
    "Loading mmCIF..."
)

cif = MMCIF2Dict(
    str(
        MMCIF_FILE
    )
)


# =========================================
# Binding-site residues from Phase 3.1
# =========================================

pocket_df = pd.read_csv(
    POCKET_FILE
)

pocket_residue_numbers = set(
    pd.to_numeric(
        pocket_df[
            "residue_number"
        ],
        errors="coerce"
    )
    .dropna()
    .astype(int)
    .tolist()
)


print(
    "Gefitinib 5 A pocket residues:",
    len(
        pocket_residue_numbers
    )
)

print(
    sorted(
        pocket_residue_numbers
    )
)


# =========================================
# Unobserved / zero occupancy residues
# =========================================

res_chain = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_residues.auth_asym_id"
)

res_name = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_residues.auth_comp_id"
)

res_number = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_residues.auth_seq_id"
)

res_ins = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_residues.PDB_ins_code"
)

res_model = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_residues.PDB_model_num"
)


missing_residue_records = []


n_res_entries = max(
    [
        len(res_chain),
        len(res_name),
        len(res_number),
        len(res_ins),
        len(res_model),
        0,
    ]
)


for i in range(
    n_res_entries
):

    chain = (
        res_chain[i]
        if i < len(res_chain)
        else ""
    )

    name = (
        res_name[i]
        if i < len(res_name)
        else ""
    )

    number_raw = (
        res_number[i]
        if i < len(res_number)
        else ""
    )

    number = safe_int(
        number_raw
    )

    insertion = (
        res_ins[i]
        if i < len(res_ins)
        else ""
    )

    model = (
        res_model[i]
        if i < len(res_model)
        else ""
    )


    if chain != "A":
        continue


    in_pocket = (
        number
        in pocket_residue_numbers
        if number is not None
        else False
    )


    missing_residue_records.append(
        {
            "chain":
                chain,

            "residue_name":
                name,

            "residue_number":
                number,

            "insertion_code":
                insertion,

            "model":
                model,

            "is_gefitinib_5A_pocket":
                in_pocket,
        }
    )


missing_residues_df = pd.DataFrame(
    missing_residue_records
)


# =========================================
# Unobserved / zero occupancy atoms
# =========================================

atom_chain = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.auth_asym_id"
)

atom_resname = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.auth_comp_id"
)

atom_resseq = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.auth_seq_id"
)

atom_name = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.auth_atom_id"
)

atom_ins = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.PDB_ins_code"
)

atom_model = get_column(
    cif,
    "_pdbx_unobs_or_zero_occ_atoms.PDB_model_num"
)


missing_atom_records = []


n_atom_entries = max(
    [
        len(atom_chain),
        len(atom_resname),
        len(atom_resseq),
        len(atom_name),
        len(atom_ins),
        len(atom_model),
        0,
    ]
)


for i in range(
    n_atom_entries
):

    chain = (
        atom_chain[i]
        if i < len(atom_chain)
        else ""
    )

    residue_name = (
        atom_resname[i]
        if i < len(atom_resname)
        else ""
    )

    number_raw = (
        atom_resseq[i]
        if i < len(atom_resseq)
        else ""
    )

    number = safe_int(
        number_raw
    )

    missing_atom = (
        atom_name[i]
        if i < len(atom_name)
        else ""
    )

    insertion = (
        atom_ins[i]
        if i < len(atom_ins)
        else ""
    )

    model = (
        atom_model[i]
        if i < len(atom_model)
        else ""
    )


    if chain != "A":
        continue


    in_pocket = (
        number
        in pocket_residue_numbers
        if number is not None
        else False
    )


    missing_atom_records.append(
        {
            "chain":
                chain,

            "residue_name":
                residue_name,

            "residue_number":
                number,

            "atom_name":
                missing_atom,

            "insertion_code":
                insertion,

            "model":
                model,

            "is_gefitinib_5A_pocket":
                in_pocket,
        }
    )


missing_atoms_df = pd.DataFrame(
    missing_atom_records
)


# =========================================
# Coordinate residue numbering
# =========================================

coordinate_residues = set()


with open(
    CLEAN_RECEPTOR,
    "r"
) as handle:

    for line in handle:

        if not line.startswith(
            "ATOM"
        ):
            continue


        chain = (
            line[21].strip()
            or "_"
        )

        if chain != "A":
            continue


        try:
            resseq = int(
                line[22:26]
            )

        except Exception:
            continue


        coordinate_residues.add(
            resseq
        )


coordinate_min = min(
    coordinate_residues
)

coordinate_max = max(
    coordinate_residues
)


numeric_gaps = [
    number
    for number in range(
        coordinate_min,
        coordinate_max + 1
    )
    if number
    not in coordinate_residues
]


# =========================================
# Pocket completeness
# =========================================

missing_pocket_residues = []


if not missing_residues_df.empty:

    missing_pocket_residues = (
        missing_residues_df[
            missing_residues_df[
                "is_gefitinib_5A_pocket"
            ]
            == True
        ]
        .to_dict(
            "records"
        )
    )


missing_pocket_atoms = []


if not missing_atoms_df.empty:

    missing_pocket_atoms = (
        missing_atoms_df[
            missing_atoms_df[
                "is_gefitinib_5A_pocket"
            ]
            == True
        ]
        .to_dict(
            "records"
        )
    )


# =========================================
# Save tables
# =========================================

missing_residues_df.to_csv(
    MISSING_RESIDUES_FILE,
    index=False
)

missing_atoms_df.to_csv(
    MISSING_ATOMS_FILE,
    index=False
)


# =========================================
# Decision logic
# =========================================

pocket_residue_problem = (
    len(
        missing_pocket_residues
    )
    > 0
)

pocket_atom_problem = (
    len(
        missing_pocket_atoms
    )
    > 0
)


if (
    not pocket_residue_problem
    and not pocket_atom_problem
):

    docking_pocket_status = (
        "PASS"
    )

else:

    docking_pocket_status = (
        "REVIEW_REQUIRED"
    )


# =========================================
# Summary
# =========================================

summary = {

    "pdb_id":
        "4WKQ",

    "coordinate_residue_count":
        len(
            coordinate_residues
        ),

    "coordinate_residue_range":
        [
            coordinate_min,
            coordinate_max,
        ],

    "numeric_coordinate_gaps":
        numeric_gaps,

    "official_missing_residue_count_chain_A":
        len(
            missing_residues_df
        ),

    "official_missing_atom_count_chain_A":
        len(
            missing_atoms_df
        ),

    "gefitinib_5A_pocket_residues":
        sorted(
            pocket_residue_numbers
        ),

    "missing_pocket_residues":
        missing_pocket_residues,

    "missing_pocket_atoms":
        missing_pocket_atoms,

    "docking_pocket_status":
        docking_pocket_status,
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
    "PHASE 3.2B — STRUCTURE COMPLETENESS"
)

print(
    "================================="
)


print(
    "Coordinate residues:",
    len(
        coordinate_residues
    )
)


print(
    "Coordinate range:",
    coordinate_min,
    "-",
    coordinate_max
)


print(
    "\nNumeric coordinate gaps:"
)

print(
    numeric_gaps
)


print(
    "\nOfficial missing residues "
    "(chain A):",
    len(
        missing_residues_df
    )
)


if not missing_residues_df.empty:

    print(
        missing_residues_df.to_string(
            index=False
        )
    )

else:

    print(
        "None reported."
    )


print(
    "\nOfficial missing atoms "
    "(chain A):",
    len(
        missing_atoms_df
    )
)


if not missing_atoms_df.empty:

    print(
        missing_atoms_df.to_string(
            index=False
        )
    )

else:

    print(
        "None reported."
    )


print(
    "\n---------------------------------"
)

print(
    "GEFITINIB POCKET COMPLETENESS"
)

print(
    "---------------------------------"
)


print(
    "5 A pocket residues:"
)

print(
    sorted(
        pocket_residue_numbers
    )
)


print(
    "\nMissing pocket residues:",
    len(
        missing_pocket_residues
    )
)


if missing_pocket_residues:

    print(
        missing_pocket_residues
    )


print(
    "\nMissing pocket atoms:",
    len(
        missing_pocket_atoms
    )
)


if missing_pocket_atoms:

    print(
        missing_pocket_atoms
    )


print(
    "\nDocking pocket status:",
    docking_pocket_status
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
    "\nSaved completeness summary:"
)

print(
    SUMMARY_FILE
)
