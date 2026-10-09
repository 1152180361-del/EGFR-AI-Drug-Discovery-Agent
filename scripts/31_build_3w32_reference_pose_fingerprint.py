from pathlib import Path
import json
import math
import pandas as pd


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

RECEPTOR = (
    ROOT
    / "docking/3W32/receptor/3W32_clean_receptor.pdb"
)

LIGAND = (
    ROOT
    / "docking/3W32/ligand/W32_3W32_crystal.pdb"
)

OUT_DIR = (
    ROOT
    / "docking/3W32/reference_pose"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONTACT_CSV = (
    OUT_DIR
    / "W32_reference_residue_contacts.csv"
)

REPORT_JSON = (
    OUT_DIR
    / "W32_reference_pose_fingerprint.json"
)


# ============================================================
# Settings
# ============================================================

CONTACT_4A = 4.0
CONTACT_5A = 5.0

HINGE_RESNUM = 793

KEY_RESIDUES = [
    718, 719, 720,
    743, 744, 745,
    762, 766,
    788, 789, 790,
    791, 792, 793, 794,
    796, 797,
    844,
    853, 854, 855, 856,
]


# ============================================================
# PDB helpers
# ============================================================

def element_from_pdb(line):

    e = line[76:78].strip()

    if e:
        return e.capitalize()

    name = line[12:16].strip()

    letters = "".join(
        c for c in name
        if c.isalpha()
    )

    if len(letters) >= 2 and letters[:2].upper() in {
        "CL", "BR",
    }:
        return (
            letters[0].upper()
            + letters[1].lower()
        )

    return letters[:1].upper()


def parse_atoms(path, records):

    atoms = []

    with open(path) as f:

        for line in f:

            if not line.startswith(records):
                continue

            element = element_from_pdb(
                line
            )

            if element.upper() == "H":
                continue

            atoms.append(
                {
                    "atom_name":
                        line[12:16].strip(),

                    "resname":
                        line[17:20].strip(),

                    "chain":
                        line[21].strip(),

                    "resnum":
                        int(
                            line[22:26]
                        ),

                    "element":
                        element,

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
                }
            )

    return atoms


def distance(a, b):

    return math.sqrt(
        (
            a["x"] - b["x"]
        ) ** 2
        +
        (
            a["y"] - b["y"]
        ) ** 2
        +
        (
            a["z"] - b["z"]
        ) ** 2
    )


# ============================================================
# Load structures
# ============================================================

protein = parse_atoms(
    RECEPTOR,
    ("ATOM",),
)

ligand = parse_atoms(
    LIGAND,
    ("ATOM", "HETATM"),
)


if not protein:
    raise RuntimeError(
        "No receptor atoms loaded."
    )

if not ligand:
    raise RuntimeError(
        "No ligand atoms loaded."
    )


print(
    "Protein heavy atoms:",
    len(protein)
)

print(
    "W32 heavy atoms:",
    len(ligand)
)


# ============================================================
# Group receptor by residue
# ============================================================

residues = {}

for atom in protein:

    key = (
        atom["chain"],
        atom["resnum"],
        atom["resname"],
    )

    residues.setdefault(
        key,
        []
    ).append(atom)


# ============================================================
# Minimum ligand distance per residue
# ============================================================

records = []


for (
    chain,
    resnum,
    resname,
), atoms in residues.items():

    best = None

    for pa in atoms:

        for la in ligand:

            d = distance(
                pa,
                la,
            )

            if (
                best is None
                or d < best["distance"]
            ):

                best = {
                    "distance":
                        d,

                    "protein_atom":
                        pa["atom_name"],

                    "ligand_atom":
                        la["atom_name"],

                    "ligand_element":
                        la["element"],
                }


    records.append(
        {
            "chain":
                chain,

            "resnum":
                resnum,

            "resname":
                resname,

            "min_distance_A":
                best["distance"],

            "nearest_protein_atom":
                best["protein_atom"],

            "nearest_ligand_atom":
                best["ligand_atom"],

            "nearest_ligand_element":
                best["ligand_element"],

            "contact_4A":
                best["distance"]
                <= CONTACT_4A,

            "contact_5A":
                best["distance"]
                <= CONTACT_5A,

            "key_residue":
                resnum
                in KEY_RESIDUES,
        }
    )


contact_df = (
    pd.DataFrame(
        records
    )
    .sort_values(
        "min_distance_A"
    )
    .reset_index(
        drop=True
    )
)


contact_df.to_csv(
    CONTACT_CSV,
    index=False,
)


# ============================================================
# Contact sets
# ============================================================

contacts_4A = (
    contact_df[
        contact_df[
            "contact_4A"
        ]
    ]
)


contacts_5A = (
    contact_df[
        contact_df[
            "contact_5A"
        ]
    ]
)


def residue_label(row):

    return (
        f"{row['resname']}"
        f"{int(row['resnum'])}"
    )


contact_labels_4A = [
    residue_label(row)
    for _, row
    in contacts_4A.iterrows()
]


contact_labels_5A = [
    residue_label(row)
    for _, row
    in contacts_5A.iterrows()
]


# ============================================================
# Key-residue table
# ============================================================

key_df = (
    contact_df[
        contact_df[
            "key_residue"
        ]
    ]
    .sort_values(
        "resnum"
    )
)


# ============================================================
# Hinge geometry around MET793
#
# This is deliberately called "heteroatom proximity",
# NOT a confirmed H-bond.
#
# Full H-bond assignment requires donor/acceptor
# chemistry and angular geometry.
# ============================================================

hinge_atoms = [
    a
    for a in protein
    if (
        a["chain"] == "A"
        and a["resnum"] == HINGE_RESNUM
        and a["atom_name"] in {
            "N",
            "O",
        }
    )
]


ligand_hetero = [
    a
    for a in ligand
    if a["element"].upper()
    in {
        "N",
        "O",
        "S",
    }
]


hinge_results = {}


for hinge_atom in hinge_atoms:

    best = None

    for la in ligand_hetero:

        d = distance(
            hinge_atom,
            la,
        )

        if (
            best is None
            or d < best[
                "distance_A"
            ]
        ):

            best = {
                "distance_A":
                    d,

                "ligand_atom":
                    la[
                        "atom_name"
                    ],

                "ligand_element":
                    la[
                        "element"
                    ],
            }


    hinge_results[
        hinge_atom[
            "atom_name"
        ]
    ] = best


# ============================================================
# Core ATP-pocket contact summary
# ============================================================

core_contact_rows = (
    key_df[
        key_df[
            "min_distance_A"
        ]
        <= CONTACT_5A
    ]
)


core_contacts = [
    residue_label(row)
    for _, row
    in core_contact_rows.iterrows()
]


# ============================================================
# Report
# ============================================================

report = {
    "reference":
        "3W32 crystal W32 pose",

    "protein_heavy_atoms":
        len(protein),

    "ligand_heavy_atoms":
        len(ligand),

    "contact_thresholds_A": {
        "strict":
            CONTACT_4A,

        "extended":
            CONTACT_5A,
    },

    "contacts_within_4A":
        contact_labels_4A,

    "contacts_within_5A":
        contact_labels_5A,

    "n_contacts_4A":
        len(
            contact_labels_4A
        ),

    "n_contacts_5A":
        len(
            contact_labels_5A
        ),

    "key_ATP_pocket_contacts_within_5A":
        core_contacts,

    "MET793_backbone_heteroatom_proximity":
        hinge_results,

    "note":
        (
            "MET793 values are geometric "
            "heteroatom proximities only; "
            "they are not automatically "
            "classified as hydrogen bonds."
        ),

    "contact_csv":
        str(
            CONTACT_CSV
        ),
}


REPORT_JSON.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


# ============================================================
# Terminal output
# ============================================================

print(
    "\n========================================"
)

print(
    "3W32 / W32 REFERENCE POSE FINGERPRINT"
)

print(
    "========================================"
)


print(
    "\nContacts <= 4.0 A:"
)

print(
    ", ".join(
        contact_labels_4A
    )
)


print(
    "\nContacts <= 5.0 A:"
)

print(
    ", ".join(
        contact_labels_5A
    )
)


print(
    "\nKey ATP-pocket contacts <= 5.0 A:"
)

print(
    ", ".join(
        core_contacts
    )
)


print(
    "\nMET793 backbone heteroatom proximity:"
)


for atom_name in [
    "N",
    "O",
]:

    value = (
        hinge_results.get(
            atom_name
        )
    )

    if value is None:

        print(
            atom_name,
            ": not available"
        )

    else:

        print(
            f"{atom_name}: "
            f"{value['distance_A']:.3f} A "
            f"to ligand "
            f"{value['ligand_atom']} "
            f"({value['ligand_element']})"
        )


print(
    "\nNearest 20 residues:"
)

print(
    contact_df[
        [
            "resname",
            "resnum",
            "min_distance_A",
            "nearest_protein_atom",
            "nearest_ligand_atom",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    CONTACT_CSV
)

print(
    REPORT_JSON
)


print(
    "\n========================================"
)

print(
    "REFERENCE FINGERPRINT: PASS"
)

print(
    "========================================"
)
