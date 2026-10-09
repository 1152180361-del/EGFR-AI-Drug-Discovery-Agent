from pathlib import Path
from collections import defaultdict
import json


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "raw"
    / "4WKQ.pdb"
)

OUTPUT_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "receptor"
    / "4WKQ_nativeCys_clean.pdb"
)

REPORT_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "4WKQ_receptor_cleanup.json"
)

CYS_FILE = (
    ROOT
    / "docking"
    / "4WKQ"
    / "reports"
    / "CYS797_after_cleanup.txt"
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


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


def replace_columns(
    line,
    record=None,
    resname=None,
    altloc=None,
):

    chars = list(
        line.ljust(80)
    )

    if record is not None:

        text = (
            f"{record:<6}"
        )

        chars[0:6] = list(
            text
        )


    if altloc is not None:

        chars[16] = (
            altloc
            if altloc
            else " "
        )


    if resname is not None:

        text = (
            f"{resname:>3}"
        )

        chars[17:20] = list(
            text
        )


    return "".join(
        chars
    ).rstrip()


# =========================================
# Load raw structure
# =========================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input PDB not found: "
        f"{INPUT_FILE}"
    )


with open(
    INPUT_FILE,
    "r"
) as handle:

    raw_lines = (
        handle.readlines()
    )


# =========================================
# Candidate protein atoms
# =========================================

candidates = []

removed_counts = defaultdict(
    int
)

original_csx_atoms = []


for line_number, line in enumerate(
    raw_lines
):

    atom = parse_atom_line(
        line
    )

    if atom is None:

        continue


    # -------------------------------------
    # Only chain A
    # -------------------------------------

    if atom["chain"] != "A":

        removed_counts[
            "non_chain_A"
        ] += 1

        continue


    # -------------------------------------
    # CSX A797 -> native CYS A797
    # -------------------------------------

    if (
        atom["resname"] == "CSX"
        and atom["resseq"] == 797
    ):

        original_csx_atoms.append(
            atom
        )


        # CSX contains an extra oxygen
        # attached to sulfur.
        # Native cysteine does not.
        if atom["atom_name"] == "OD":

            removed_counts[
                "CSX797_OD"
            ] += 1

            continue


        cleaned_line = (
            replace_columns(
                atom["line"],
                record="ATOM",
                resname="CYS",
                altloc=atom[
                    "altloc"
                ],
            )
        )


        cleaned_atom = (
            parse_atom_line(
                cleaned_line
            )
        )


        candidates.append(
            {
                "line_number":
                    line_number,

                "atom":
                    cleaned_atom,

                "line":
                    cleaned_line,
            }
        )

        continue


    # -------------------------------------
    # Remove all other HETATM records
    #
    # This removes:
    # IRE / gefitinib
    # Na+
    # MES
    # crystallographic waters
    # -------------------------------------

    if atom["record"] == "HETATM":

        removed_counts[
            atom["resname"]
        ] += 1

        continue


    # -------------------------------------
    # Keep standard protein ATOM
    # -------------------------------------

    candidates.append(
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
#
# Preference:
#
# blank altloc
# > altloc A
# > other altloc
#
# If same preference:
# choose highest occupancy
# =========================================

atom_groups = defaultdict(
    list
)


for item in candidates:

    atom = item[
        "atom"
    ]

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


selected_items = []

altloc_removed = 0


def altloc_rank(atom):

    altloc = atom[
        "altloc"
    ]

    if altloc == "":
        return 3

    if altloc == "A":
        return 2

    return 1


for key, group in atom_groups.items():

    best = max(
        group,
        key=lambda item: (
            altloc_rank(
                item[
                    "atom"
                ]
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


    # Clear altloc marker in final receptor
    cleaned_line = replace_columns(
        best["line"],
        altloc="",
    )


    selected_items.append(
        {
            "line_number":
                best[
                    "line_number"
                ],

            "line":
                cleaned_line,

            "atom":
                parse_atom_line(
                    cleaned_line
                ),
        }
    )


# =========================================
# Preserve original coordinate order
# =========================================

selected_items.sort(
    key=lambda item:
        item[
            "line_number"
        ]
)


# =========================================
# Write clean receptor
# =========================================

with open(
    OUTPUT_FILE,
    "w"
) as handle:

    for item in selected_items:

        handle.write(
            item[
                "line"
            ]
            + "\n"
        )

    handle.write(
        "TER\n"
    )

    handle.write(
        "END\n"
    )


# =========================================
# Final receptor QC
# =========================================

final_atoms = [
    item["atom"]
    for item in selected_items
]


final_residues = set()


for atom in final_atoms:

    final_residues.add(
        (
            atom["chain"],
            atom["resseq"],
            atom["icode"],
            atom["resname"],
        )
    )


# =========================================
# CYS797 QC
# =========================================

cys797_atoms = [
    atom
    for atom in final_atoms
    if (
        atom["chain"] == "A"
        and atom["resseq"] == 797
        and atom["resname"] == "CYS"
    )
]


cys797_atom_names = [
    atom[
        "atom_name"
    ]
    for atom in cys797_atoms
]


expected_cys_atoms = {
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "SG",
}


actual_cys_atoms = set(
    cys797_atom_names
)


missing_cys_atoms = (
    expected_cys_atoms
    - actual_cys_atoms
)


unexpected_cys_atoms = (
    actual_cys_atoms
    - expected_cys_atoms
)


# =========================================
# Write CYS797 records
# =========================================

with open(
    CYS_FILE,
    "w"
) as handle:

    for item in selected_items:

        atom = item[
            "atom"
        ]

        if (
            atom["chain"] == "A"
            and atom["resseq"] == 797
            and atom["resname"] == "CYS"
        ):

            handle.write(
                item[
                    "line"
                ]
                + "\n"
            )


# =========================================
# Check forbidden residues remain
# =========================================

forbidden_resnames = {
    "IRE",
    "MES",
    "NA",
    "HOH",
    "WAT",
    "CSX",
}


remaining_forbidden = sorted(
    {
        atom[
            "resname"
        ]
        for atom in final_atoms
        if atom[
            "resname"
        ]
        in forbidden_resnames
    }
)


# =========================================
# Protein sequence range
# =========================================

residue_numbers = sorted(
    {
        atom[
            "resseq"
        ]
        for atom in final_atoms
    }
)


# =========================================
# Safety checks
# =========================================

if missing_cys_atoms:

    raise RuntimeError(
        "CYS797 missing atoms: "
        + ", ".join(
            sorted(
                missing_cys_atoms
            )
        )
    )


if unexpected_cys_atoms:

    raise RuntimeError(
        "CYS797 unexpected atoms: "
        + ", ".join(
            sorted(
                unexpected_cys_atoms
            )
        )
    )


if remaining_forbidden:

    raise RuntimeError(
        "Forbidden residues remain: "
        + ", ".join(
            remaining_forbidden
        )
    )


if len(cys797_atoms) != 6:

    raise RuntimeError(
        "CYS797 should contain "
        "exactly 6 heavy atoms."
    )


# =========================================
# Report
# =========================================

report = {

    "source":
        "4WKQ.pdb",

    "chain":
        "A",

    "original_CSX797_atoms":
        [
            atom[
                "atom_name"
            ]
            for atom
            in original_csx_atoms
        ],

    "final_CYS797_atoms":
        cys797_atom_names,

    "CSX797_OD_removed":
        removed_counts[
            "CSX797_OD"
        ],

    "final_atom_count":
        len(
            final_atoms
        ),

    "final_residue_count":
        len(
            final_residues
        ),

    "first_residue_number":
        min(
            residue_numbers
        ),

    "last_residue_number":
        max(
            residue_numbers
        ),

    "alternate_atoms_removed":
        altloc_removed,

    "removed_atom_counts":
        dict(
            removed_counts
        ),

    "remaining_forbidden_residues":
        remaining_forbidden,

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
    "PHASE 3.2A — RECEPTOR NORMALIZATION"
)

print(
    "================================="
)


print(
    "Final protein atoms:",
    len(
        final_atoms
    )
)


print(
    "Final protein residues:",
    len(
        final_residues
    )
)


print(
    "Residue number range:",
    min(
        residue_numbers
    ),
    "-",
    max(
        residue_numbers
    )
)


print(
    "\nOriginal CSX797 atoms:"
)

print(
    [
        atom[
            "atom_name"
        ]
        for atom
        in original_csx_atoms
    ]
)


print(
    "\nFinal CYS797 atoms:"
)

print(
    cys797_atom_names
)


print(
    "\nRemoved CSX OD atoms:",
    removed_counts[
        "CSX797_OD"
    ]
)


print(
    "\nAlternate-location atoms removed:",
    altloc_removed
)


print(
    "\nRemoved coordinate atoms:"
)


for name, count in sorted(
    removed_counts.items()
):

    print(
        f"{name}: {count}"
    )


print(
    "\nRemaining forbidden residues:",
    remaining_forbidden
)


print(
    "\nCYS797 QC: PASS"
)

print(
    "Receptor cleanup QC: PASS"
)


print(
    "\nSaved clean receptor:"
)

print(
    OUTPUT_FILE
)


print(
    "\nSaved CYS797 records:"
)

print(
    CYS_FILE
)


print(
    "\nSaved cleanup report:"
)

print(
    REPORT_FILE
)
