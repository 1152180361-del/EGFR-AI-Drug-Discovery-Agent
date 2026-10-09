from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

RECEPTOR = (
    ROOT
    / "docking/3W32/receptor/3W32_clean_receptor.pdb"
)

REFERENCE_JSON = (
    ROOT
    / "docking/3W32/reference_pose/"
      "W32_reference_pose_fingerprint.json"
)

DOCKING_RESULTS = (
    ROOT
    / "docking/3W32/benchmark/"
      "benchmark_docking_results.csv"
)

OUT_CSV = (
    ROOT
    / "docking/3W32/benchmark/"
      "benchmark_pose_fingerprint_features.csv"
)

OUT_JSON = (
    ROOT
    / "docking/3W32/benchmark/"
      "benchmark_pose_fingerprint_summary.json"
)


CONTACT_CUTOFF = 5.0
HINGE_GEOMETRY_CUTOFF = 3.5


# ============================================================
# Helpers
# ============================================================

def parse_receptor(path):

    residues = {}
    atoms_by_key = {}

    with open(path) as f:

        for line in f:

            if not line.startswith("ATOM"):
                continue

            element = (
                line[76:78]
                .strip()
                .upper()
            )

            if element == "H":
                continue

            chain = line[21].strip()
            resname = line[17:20].strip()
            resnum = int(line[22:26])
            atom_name = line[12:16].strip()

            xyz = np.array(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=float,
            )

            key = (
                chain,
                resnum,
                resname,
            )

            residues.setdefault(
                key,
                []
            ).append(xyz)

            atoms_by_key[
                (
                    chain,
                    resnum,
                    atom_name,
                )
            ] = xyz

    residues = {
        key: np.asarray(
            coords,
            dtype=float,
        )
        for key, coords
        in residues.items()
    }

    return (
        residues,
        atoms_by_key,
    )


def parse_top1_pdbqt(path):

    atoms = []

    in_model_1 = False
    saw_model = False

    with open(
        path,
        errors="replace",
    ) as f:

        for line in f:

            if line.startswith("MODEL"):

                saw_model = True

                try:
                    model_num = int(
                        line.split()[1]
                    )
                except Exception:
                    model_num = 1

                if model_num == 1:
                    in_model_1 = True
                    continue

                if in_model_1:
                    break

                continue

            if line.startswith("ENDMDL"):

                if in_model_1:
                    break

                continue

            if saw_model and not in_model_1:
                continue

            if not line.startswith(
                ("ATOM", "HETATM")
            ):
                continue

            try:

                x = float(
                    line[30:38]
                )

                y = float(
                    line[38:46]
                )

                z = float(
                    line[46:54]
                )

            except Exception:
                continue


            fields = line.split()

            atom_type = (
                fields[-1]
                if fields
                else ""
            )

            atom_type_upper = (
                atom_type.upper()
            )


            # Skip hydrogens
            if atom_type_upper.startswith("H"):
                continue


            atom_name = (
                line[12:16]
                .strip()
            )


            hetero = (
                atom_type_upper.startswith("N")
                or atom_type_upper.startswith("O")
                or atom_type_upper.startswith("S")
            )


            atoms.append(
                {
                    "atom_name":
                        atom_name,

                    "atom_type":
                        atom_type,

                    "hetero":
                        hetero,

                    "xyz":
                        np.array(
                            [x, y, z],
                            dtype=float,
                        ),
                }
            )

    return atoms


def residue_label(
    resname,
    resnum,
):

    return (
        f"{resname}{resnum}"
    )


def min_distance_between_sets(
    a,
    b,
):

    diff = (
        a[:, None, :]
        - b[None, :, :]
    )

    d2 = np.sum(
        diff * diff,
        axis=2,
    )

    return float(
        np.sqrt(
            np.min(
                d2
            )
        )
    )


# ============================================================
# Load reference fingerprint
# ============================================================

reference = json.loads(
    REFERENCE_JSON.read_text()
)


reference_contacts = set(
    reference[
        "contacts_within_5A"
    ]
)


reference_key_contacts = set(
    reference[
        "key_ATP_pocket_contacts_within_5A"
    ]
)


print(
    "Reference contacts <=5 A:",
    len(reference_contacts)
)

print(
    "Reference key ATP contacts:",
    len(reference_key_contacts)
)


# ============================================================
# Receptor
# ============================================================

residues, receptor_atoms = (
    parse_receptor(
        RECEPTOR
    )
)


print(
    "Receptor residues:",
    len(residues)
)


met793_N = (
    receptor_atoms.get(
        (
            "A",
            793,
            "N",
        )
    )
)

met793_O = (
    receptor_atoms.get(
        (
            "A",
            793,
            "O",
        )
    )
)


if (
    met793_N is None
    or met793_O is None
):

    raise RuntimeError(
        "MET793 backbone N/O not found."
    )


# ============================================================
# Docking table
# ============================================================

df = pd.read_csv(
    DOCKING_RESULTS
)


df = df[
    df[
        "status"
    ].isin(
        [
            "PASS",
            "PASS_EXISTING",
        ]
    )
].copy()


print(
    "Docked molecules:",
    len(df)
)


# ============================================================
# Process each Top-1 pose
# ============================================================

records = []


for i, row in df.iterrows():

    bid = str(
        row[
            "benchmark_id"
        ]
    )

    pose_path = Path(
        row[
            "docked_pdbqt"
        ]
    )


    atoms = (
        parse_top1_pdbqt(
            pose_path
        )
    )


    if not atoms:

        raise RuntimeError(
            f"No Top-1 atoms parsed: {bid}"
        )


    ligand_xyz = np.asarray(
        [
            a["xyz"]
            for a in atoms
        ],
        dtype=float,
    )


    hetero_xyz = np.asarray(
        [
            a["xyz"]
            for a in atoms
            if a[
                "hetero"
            ]
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # Residue contacts <=5 Å
    # --------------------------------------------------------

    pose_contacts = set()


    for (
        chain,
        resnum,
        resname,
    ), protein_xyz in (
        residues.items()
    ):

        d = (
            min_distance_between_sets(
                protein_xyz,
                ligand_xyz,
            )
        )


        if d <= CONTACT_CUTOFF:

            pose_contacts.add(
                residue_label(
                    resname,
                    resnum,
                )
            )


    # --------------------------------------------------------
    # Reference-contact recovery
    # --------------------------------------------------------

    intersection = (
        pose_contacts
        & reference_contacts
    )


    union = (
        pose_contacts
        | reference_contacts
    )


    reference_recovery = (
        len(intersection)
        / len(reference_contacts)
        if reference_contacts
        else np.nan
    )


    jaccard = (
        len(intersection)
        / len(union)
        if union
        else np.nan
    )


    # --------------------------------------------------------
    # Key ATP-pocket coverage
    # --------------------------------------------------------

    key_hits = (
        pose_contacts
        & reference_key_contacts
    )


    key_coverage = (
        len(key_hits)
        / len(
            reference_key_contacts
        )
        if reference_key_contacts
        else np.nan
    )


    # --------------------------------------------------------
    # MET793 hinge geometry
    #
    # Only ligand N/O/S-like PDBQT atom types.
    # This is geometric proximity,
    # NOT automatic H-bond assignment.
    # --------------------------------------------------------

    if len(hetero_xyz) > 0:

        dN = float(
            np.min(
                np.linalg.norm(
                    hetero_xyz
                    - met793_N,
                    axis=1,
                )
            )
        )


        dO = float(
            np.min(
                np.linalg.norm(
                    hetero_xyz
                    - met793_O,
                    axis=1,
                )
            )
        )


    else:

        dN = np.nan
        dO = np.nan


    hinge_N_close = (
        bool(
            np.isfinite(dN)
            and dN
            <= HINGE_GEOMETRY_CUTOFF
        )
    )


    hinge_O_close = (
        bool(
            np.isfinite(dO)
            and dO
            <= HINGE_GEOMETRY_CUTOFF
        )
    )


    hinge_close_count = (
        int(
            hinge_N_close
        )
        +
        int(
            hinge_O_close
        )
    )


    records.append(
        {
            "benchmark_id":
                bid,

            "benchmark_label":
                row[
                    "benchmark_label"
                ],

            "binary_label":
                int(
                    row[
                        "binary_label"
                    ]
                ),

            "pIC50":
                float(
                    row[
                        "pIC50"
                    ]
                ),

            "top1_vina_score":
                float(
                    row[
                        "top1_vina_score"
                    ]
                ),

            "ligand_heavy_atoms":
                len(
                    atoms
                ),

            "ligand_hetero_atoms":
                int(
                    len(
                        hetero_xyz
                    )
                ),

            "pose_contact_count_5A":
                len(
                    pose_contacts
                ),

            "reference_contacts_recovered":
                len(
                    intersection
                ),

            "reference_contact_recovery":
                float(
                    reference_recovery
                ),

            "reference_contact_jaccard":
                float(
                    jaccard
                ),

            "key_ATP_contacts_recovered":
                len(
                    key_hits
                ),

            "key_ATP_contact_coverage":
                float(
                    key_coverage
                ),

            "MET793_N_hetero_distance_A":
                dN,

            "MET793_O_hetero_distance_A":
                dO,

            "MET793_N_close_3p5A":
                hinge_N_close,

            "MET793_O_close_3p5A":
                hinge_O_close,

            "MET793_close_count":
                hinge_close_count,

            "pose_contacts_5A":
                ";".join(
                    sorted(
                        pose_contacts
                    )
                ),

            "reference_contacts_hit":
                ";".join(
                    sorted(
                        intersection
                    )
                ),

            "key_contacts_hit":
                ";".join(
                    sorted(
                        key_hits
                    )
                ),
        }
    )


    if (
        (len(records) % 10 == 0)
        or len(records) == len(df)
    ):

        print(
            f"Processed "
            f"{len(records)}/{len(df)}"
        )


# ============================================================
# DataFrame
# ============================================================

features = pd.DataFrame(
    records
)


features.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Group summaries
# ============================================================

metrics = [
    "reference_contact_recovery",
    "reference_contact_jaccard",
    "key_ATP_contact_coverage",
    "MET793_N_hetero_distance_A",
    "MET793_O_hetero_distance_A",
    "MET793_close_count",
]


group_summary = {}


for label in [
    "ACTIVE",
    "LOW_ACTIVE",
]:

    subset = features[
        features[
            "benchmark_label"
        ]
        == label
    ]


    group_summary[
        label
    ] = {}


    for metric in metrics:

        values = (
            subset[
                metric
            ]
            .astype(float)
        )


        group_summary[
            label
        ][
            metric
        ] = {
            "mean":
                float(
                    values.mean()
                ),

            "median":
                float(
                    values.median()
                ),

            "std":
                float(
                    values.std()
                ),
        }


    group_summary[
        label
    ][
        "MET793_N_close_fraction"
    ] = float(
        subset[
            "MET793_N_close_3p5A"
        ].mean()
    )


    group_summary[
        label
    ][
        "MET793_O_close_fraction"
    ] = float(
        subset[
            "MET793_O_close_3p5A"
        ].mean()
    )


# ============================================================
# JSON summary
# ============================================================

report = {
    "reference_contact_count_5A":
        len(
            reference_contacts
        ),

    "reference_key_ATP_contact_count":
        len(
            reference_key_contacts
        ),

    "contact_cutoff_A":
        CONTACT_CUTOFF,

    "hinge_geometry_cutoff_A":
        HINGE_GEOMETRY_CUTOFF,

    "n_poses":
        len(
            features
        ),

    "group_summary":
        group_summary,

    "output_csv":
        str(
            OUT_CSV
        ),

    "note":
        (
            "MET793 hinge features are "
            "geometric heteroatom proximities "
            "only and are not equivalent to "
            "hydrogen-bond assignments."
        ),
}


OUT_JSON.write_text(
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
    "EGFR BENCHMARK POSE FINGERPRINTS"
)

print(
    "========================================"
)


print(
    "\nReference:"
)

print(
    "5A contacts:",
    len(
        reference_contacts
    )
)

print(
    "Key ATP contacts:",
    len(
        reference_key_contacts
    )
)


print(
    "\n----------------------------------------"
)

print(
    "GROUP SUMMARY"
)

print(
    "----------------------------------------"
)


for label in [
    "ACTIVE",
    "LOW_ACTIVE",
]:

    s = (
        group_summary[
            label
        ]
    )


    print(
        f"\n{label}"
    )


    print(
        "Reference-contact recovery:",
        f"{s['reference_contact_recovery']['mean']:.3f}"
    )


    print(
        "Contact Jaccard:",
        f"{s['reference_contact_jaccard']['mean']:.3f}"
    )


    print(
        "Key ATP coverage:",
        f"{s['key_ATP_contact_coverage']['mean']:.3f}"
    )


    print(
        "MET793 N distance:",
        f"{s['MET793_N_hetero_distance_A']['mean']:.3f} A"
    )


    print(
        "MET793 O distance:",
        f"{s['MET793_O_hetero_distance_A']['mean']:.3f} A"
    )


    print(
        "MET793 N <=3.5A fraction:",
        f"{s['MET793_N_close_fraction']:.3f}"
    )


    print(
        "MET793 O <=3.5A fraction:",
        f"{s['MET793_O_close_fraction']:.3f}"
    )


print(
    "\n----------------------------------------"
)

print(
    "TOP 10 BY REFERENCE-CONTACT JACCARD"
)

print(
    "----------------------------------------"
)


top = (
    features
    .sort_values(
        "reference_contact_jaccard",
        ascending=False,
    )
    .head(10)
)


print(
    top[
        [
            "benchmark_id",
            "benchmark_label",
            "pIC50",
            "top1_vina_score",
            "reference_contact_recovery",
            "reference_contact_jaccard",
            "key_ATP_contact_coverage",
            "MET793_N_hetero_distance_A",
            "MET793_O_hetero_distance_A",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    OUT_CSV
)

print(
    OUT_JSON
)


print(
    "\n========================================"
)

print(
    "POSE FINGERPRINT EXTRACTION: PASS"
)

print(
    "========================================"
)
