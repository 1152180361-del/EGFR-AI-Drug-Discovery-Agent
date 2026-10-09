from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)

RECEPTOR = (
    ROOT
    / "docking/3W32/receptor/"
      "3W32_clean_receptor.pdb"
)

REFERENCE_JSON = (
    ROOT
    / "docking/3W32/reference_pose/"
      "W32_reference_pose_fingerprint.json"
)

DOCKING_RESULTS = (
    ROOT
    / "docking/3W32/phase4C/reports/"
      "phase4C_docking_results.csv"
)

OUT_DIR = (
    ROOT
    / "docking/3W32/phase4C/reports"
)

FEATURES_OUT = (
    OUT_DIR
    / "phase4C_top1_pose_features.csv"
)

REPORT_OUT = (
    OUT_DIR
    / "phase4C_top1_pose_features_report.json"
)


# ============================================================
# Predefined geometry settings
# ============================================================

CONTACT_CUTOFF = 5.0

HINGE_PROXIMITY_CUTOFF = 3.5

# Purely a steric sanity check.
# Not used as a potency score.
SEVERE_CLASH_CUTOFF = 1.5


# ============================================================
# PDB receptor parser
# ============================================================

def pdb_element(line):

    e = (
        line[76:78]
        .strip()
    )

    if e:
        return e.capitalize()

    name = (
        line[12:16]
        .strip()
    )

    letters = "".join(
        c for c in name
        if c.isalpha()
    )

    if (
        len(letters) >= 2
        and letters[:2].upper()
        in {
            "CL",
            "BR",
        }
    ):

        return (
            letters[0].upper()
            + letters[1].lower()
        )

    return (
        letters[:1].upper()
    )


def parse_receptor(path):

    residues = {}

    atoms_by_key = {}

    all_coords = []


    with open(
        path,
        errors="replace",
    ) as f:

        for line in f:

            if not line.startswith(
                "ATOM"
            ):
                continue


            element = (
                pdb_element(
                    line
                )
            )


            if element.upper() == "H":
                continue


            chain = (
                line[21]
                .strip()
            )

            resname = (
                line[17:20]
                .strip()
            )

            resnum = int(
                line[22:26]
            )

            atom_name = (
                line[12:16]
                .strip()
            )


            xyz = np.array(
                [
                    float(
                        line[30:38]
                    ),
                    float(
                        line[38:46]
                    ),
                    float(
                        line[46:54]
                    ),
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
            ).append(
                xyz
            )


            atoms_by_key[
                (
                    chain,
                    resnum,
                    atom_name,
                )
            ] = xyz


            all_coords.append(
                xyz
            )


    residues = {
        key: np.asarray(
            value,
            dtype=float,
        )
        for key, value
        in residues.items()
    }


    return (
        residues,
        atoms_by_key,
        np.asarray(
            all_coords,
            dtype=float,
        ),
    )


# ============================================================
# PDBQT Top-1 parser
# ============================================================

def parse_top1_pdbqt(path):

    atoms = []

    saw_model = False
    in_model_1 = False


    with open(
        path,
        errors="replace",
    ) as f:

        for line in f:

            if line.startswith(
                "MODEL"
            ):

                saw_model = True

                parts = line.split()

                try:

                    model_num = int(
                        parts[1]
                    )

                except Exception:

                    model_num = 1


                if model_num == 1:

                    in_model_1 = True
                    continue


                if in_model_1:

                    break


                continue


            if line.startswith(
                "ENDMDL"
            ):

                if in_model_1:
                    break

                continue


            if (
                saw_model
                and not in_model_1
            ):

                continue


            if not line.startswith(
                (
                    "ATOM",
                    "HETATM",
                )
            ):

                continue


            try:

                xyz = np.array(
                    [
                        float(
                            line[30:38]
                        ),
                        float(
                            line[38:46]
                        ),
                        float(
                            line[46:54]
                        ),
                    ],
                    dtype=float,
                )

            except Exception:

                continue


            fields = (
                line.split()
            )


            atom_type = (
                fields[-1]
                if fields
                else ""
            )


            atom_type_upper = (
                atom_type.upper()
            )


            # Skip ligand hydrogens
            if atom_type_upper.startswith(
                "H"
            ):

                continue


            atom_name = (
                line[12:16]
                .strip()
            )


            is_hetero = (
                atom_type_upper.startswith(
                    "N"
                )
                or atom_type_upper.startswith(
                    "O"
                )
                or atom_type_upper.startswith(
                    "S"
                )
            )


            atoms.append(
                {
                    "atom_name":
                        atom_name,

                    "atom_type":
                        atom_type,

                    "hetero":
                        is_hetero,

                    "xyz":
                        xyz,
                }
            )


    return atoms


# ============================================================
# Geometry helpers
# ============================================================

def residue_label(
    resname,
    resnum,
):

    return (
        f"{resname}{resnum}"
    )


def minimum_distance(
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


def residue_distance(
    residues,
    resnum,
    ligand_xyz,
):

    candidates = [
        coords
        for (
            chain,
            rn,
            resname,
        ), coords
        in residues.items()
        if (
            chain == "A"
            and rn == resnum
        )
    ]


    if not candidates:

        return np.nan


    protein_xyz = np.concatenate(
        candidates,
        axis=0,
    )


    return minimum_distance(
        protein_xyz,
        ligand_xyz,
    )


# ============================================================
# Load reference
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
    "\n========================================"
)

print(
    "PHASE 4E-1 TOP-1 POSE ANALYSIS"
)

print(
    "========================================"
)


print(
    "\nReference contacts <=5 A:",
    len(
        reference_contacts
    )
)


print(
    "Reference key ATP contacts:",
    len(
        reference_key_contacts
    )
)


# ============================================================
# Load receptor
# ============================================================

(
    residues,
    receptor_atoms,
    receptor_xyz,
) = parse_receptor(
    RECEPTOR
)


print(
    "Receptor residues:",
    len(
        residues
    )
)


print(
    "Receptor heavy atoms:",
    len(
        receptor_xyz
    )
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
        "MET793 backbone N/O missing."
    )


# ============================================================
# Load docking results
# ============================================================

dock = pd.read_csv(
    DOCKING_RESULTS
)


dock = dock[
    dock[
        "status"
    ].isin(
        [
            "PASS",
            "PASS_EXISTING",
        ]
    )
].copy()


print(
    "Docked poses to analyze:",
    len(
        dock
    )
)


# ============================================================
# Analyze
# ============================================================

records = []


for n, (_, row) in enumerate(
    dock.iterrows(),
    start=1,
):

    cid = str(
        row[
            "phase4C_candidate_id"
        ]
    )


    pose_path = Path(
        row[
            "docked_pdbqt"
        ]
    )


    atoms = parse_top1_pdbqt(
        pose_path
    )


    if not atoms:

        print(
            f"{cid}: FAIL no atoms"
        )

        continue


    ligand_xyz = np.asarray(
        [
            atom[
                "xyz"
            ]
            for atom in atoms
        ],
        dtype=float,
    )


    hetero_xyz = np.asarray(
        [
            atom[
                "xyz"
            ]
            for atom in atoms
            if atom[
                "hetero"
            ]
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # Residue contacts <= 5 A
    # --------------------------------------------------------

    pose_contacts = set()


    for (
        chain,
        resnum,
        resname,
    ), protein_xyz in (
        residues.items()
    ):

        d = minimum_distance(
            protein_xyz,
            ligand_xyz,
        )


        if d <= CONTACT_CUTOFF:

            pose_contacts.add(
                residue_label(
                    resname,
                    resnum,
                )
            )


    # --------------------------------------------------------
    # W32-reference similarity
    # --------------------------------------------------------

    reference_hit = (
        pose_contacts
        & reference_contacts
    )


    union = (
        pose_contacts
        | reference_contacts
    )


    reference_recovery = (
        len(
            reference_hit
        )
        / len(
            reference_contacts
        )
    )


    reference_jaccard = (
        len(
            reference_hit
        )
        / len(
            union
        )
        if union
        else np.nan
    )


    key_hit = (
        pose_contacts
        & reference_key_contacts
    )


    key_coverage = (
        len(
            key_hit
        )
        / len(
            reference_key_contacts
        )
    )


    # --------------------------------------------------------
    # MET793 backbone geometric proximity
    #
    # Not automatically interpreted as H-bond.
    # --------------------------------------------------------

    if len(
        hetero_xyz
    ) > 0:

        met793_N_distance = float(
            np.min(
                np.linalg.norm(
                    hetero_xyz
                    - met793_N,
                    axis=1,
                )
            )
        )


        met793_O_distance = float(
            np.min(
                np.linalg.norm(
                    hetero_xyz
                    - met793_O,
                    axis=1,
                )
            )
        )


    else:

        met793_N_distance = np.nan
        met793_O_distance = np.nan


    met793_N_close = bool(
        np.isfinite(
            met793_N_distance
        )
        and
        met793_N_distance
        <= HINGE_PROXIMITY_CUTOFF
    )


    met793_O_close = bool(
        np.isfinite(
            met793_O_distance
        )
        and
        met793_O_distance
        <= HINGE_PROXIMITY_CUTOFF
    )


    # --------------------------------------------------------
    # Important ATP-site residue distances
    #
    # Descriptive only.
    # --------------------------------------------------------

    d_leu718 = residue_distance(
        residues,
        718,
        ligand_xyz,
    )


    d_lys745 = residue_distance(
        residues,
        745,
        ligand_xyz,
    )


    d_met793 = residue_distance(
        residues,
        793,
        ligand_xyz,
    )


    d_cys797 = residue_distance(
        residues,
        797,
        ligand_xyz,
    )


    d_leu844 = residue_distance(
        residues,
        844,
        ligand_xyz,
    )


    d_asp855 = residue_distance(
        residues,
        855,
        ligand_xyz,
    )


    # --------------------------------------------------------
    # Steric sanity check
    # --------------------------------------------------------

    diff = (
        receptor_xyz[:, None, :]
        - ligand_xyz[None, :, :]
    )


    all_distances = np.sqrt(
        np.sum(
            diff * diff,
            axis=2,
        )
    )


    min_receptor_distance = float(
        np.min(
            all_distances
        )
    )


    severe_clash_pairs = int(
        np.sum(
            all_distances
            < SEVERE_CLASH_CUTOFF
        )
    )


    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    records.append(
        {
            "phase4C_candidate_id":
                cid,

            "candidate_track":
                row[
                    "candidate_track"
                ],

            "zinc_id":
                row[
                    "zinc_id"
                ],

            "qsar_predicted_pIC50":
                row[
                    "qsar_predicted_pIC50"
                ],

            "max_training_tanimoto":
                row[
                    "max_training_tanimoto"
                ],

            "rf_tree_sd_pIC50":
                row[
                    "rf_tree_sd_pIC50"
                ],

            "top1_vina_score":
                row[
                    "top1_vina_score"
                ],

            "ligand_heavy_atoms":
                len(
                    atoms
                ),

            "ligand_hetero_atoms":
                len(
                    hetero_xyz
                ),

            "pose_contact_count_5A":
                len(
                    pose_contacts
                ),

            "reference_contacts_recovered":
                len(
                    reference_hit
                ),

            "reference_contact_recovery":
                reference_recovery,

            "reference_contact_jaccard":
                reference_jaccard,

            "key_ATP_contacts_recovered":
                len(
                    key_hit
                ),

            "key_ATP_contact_coverage":
                key_coverage,

            "MET793_N_hetero_distance_A":
                met793_N_distance,

            "MET793_O_hetero_distance_A":
                met793_O_distance,

            "MET793_N_close_3p5A":
                met793_N_close,

            "MET793_O_close_3p5A":
                met793_O_close,

            "MET793_any_close_3p5A":
                (
                    met793_N_close
                    or met793_O_close
                ),

            "LEU718_min_distance_A":
                d_leu718,

            "LYS745_min_distance_A":
                d_lys745,

            "MET793_min_distance_A":
                d_met793,

            "CYS797_min_distance_A":
                d_cys797,

            "LEU844_min_distance_A":
                d_leu844,

            "ASP855_min_distance_A":
                d_asp855,

            "min_receptor_heavy_distance_A":
                min_receptor_distance,

            "severe_clash_pairs_lt1p5A":
                severe_clash_pairs,

            "pose_contacts_5A":
                ";".join(
                    sorted(
                        pose_contacts
                    )
                ),

            "reference_contacts_hit":
                ";".join(
                    sorted(
                        reference_hit
                    )
                ),

            "key_ATP_contacts_hit":
                ";".join(
                    sorted(
                        key_hit
                    )
                ),

            "docked_pdbqt":
                str(
                    pose_path
                ),
        }
    )


    print(
        f"[{n:02d}/{len(dock):02d}] "
        f"{cid} | "
        f"Jaccard="
        f"{reference_jaccard:.3f} | "
        f"KeyCov="
        f"{key_coverage:.3f} | "
        f"MET793 N/O="
        f"{met793_N_distance:.2f}/"
        f"{met793_O_distance:.2f} A | "
        f"clash={severe_clash_pairs}",
        flush=True,
    )


# ============================================================
# Save
# ============================================================

features = pd.DataFrame(
    records
)


features.to_csv(
    FEATURES_OUT,
    index=False,
)


# ============================================================
# Track summaries
# ============================================================

track_summary = {}


for track in features[
    "candidate_track"
].unique():

    sub = features[
        features[
            "candidate_track"
        ] == track
    ]


    track_summary[
        str(track)
    ] = {
        "n":
            int(
                len(sub)
            ),

        "mean_reference_recovery":
            float(
                sub[
                    "reference_contact_recovery"
                ].mean()
            ),

        "mean_reference_jaccard":
            float(
                sub[
                    "reference_contact_jaccard"
                ].mean()
            ),

        "mean_key_ATP_coverage":
            float(
                sub[
                    "key_ATP_contact_coverage"
                ].mean()
            ),

        "MET793_any_close_fraction":
            float(
                sub[
                    "MET793_any_close_3p5A"
                ].mean()
            ),

        "zero_severe_clash_fraction":
            float(
                (
                    sub[
                        "severe_clash_pairs_lt1p5A"
                    ] == 0
                ).mean()
            ),
    }


# ============================================================
# Report
# ============================================================

report = {
    "n_analyzed":
        int(
            len(
                features
            )
        ),

    "reference_contacts_5A":
        int(
            len(
                reference_contacts
            )
        ),

    "reference_key_ATP_contacts":
        int(
            len(
                reference_key_contacts
            )
        ),

    "settings": {
        "contact_cutoff_A":
            CONTACT_CUTOFF,

        "MET793_geometric_proximity_cutoff_A":
            HINGE_PROXIMITY_CUTOFF,

        "severe_clash_cutoff_A":
            SEVERE_CLASH_CUTOFF,
    },

    "track_summary":
        track_summary,

    "interpretation_note":
        (
            "Reference-contact metrics describe similarity "
            "to the W32 crystal binding mode and must not "
            "be interpreted as potency predictors. "
            "MET793 N/O values are heteroatom proximity "
            "metrics, not automatic hydrogen-bond calls."
        ),

    "output":
        str(
            FEATURES_OUT
        ),
}


REPORT_OUT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


# ============================================================
# Terminal summary
# ============================================================

print(
    "\n========================================"
)

print(
    "PHASE 4E-1 STRUCTURAL SUMMARY"
)

print(
    "========================================"
)


print(
    "\nAnalyzed Top-1 poses:",
    len(
        features
    )
)


print(
    "\nTrack summaries:"
)


for track, values in (
    track_summary.items()
):

    print(
        f"\n{track}"
    )

    print(
        "  Reference recovery:",
        f"{values['mean_reference_recovery']:.3f}"
    )

    print(
        "  Reference Jaccard:",
        f"{values['mean_reference_jaccard']:.3f}"
    )

    print(
        "  Key ATP coverage:",
        f"{values['mean_key_ATP_coverage']:.3f}"
    )

    print(
        "  MET793 <=3.5A fraction:",
        f"{values['MET793_any_close_fraction']:.3f}"
    )

    print(
        "  Zero severe-clash fraction:",
        f"{values['zero_severe_clash_fraction']:.3f}"
    )


# ============================================================
# Supported candidates
# ============================================================

print(
    "\n----------------------------------------"
)

print(
    "SUPPORTED CANDIDATES"
)

print(
    "----------------------------------------"
)


supported = features[
    features[
        "candidate_track"
    ].isin(
        [
            "SUPPORTED_A",
            "SUPPORTED_B_HIGH_SPREAD",
        ]
    )
].copy()


supported = supported.sort_values(
    "qsar_predicted_pIC50",
    ascending=False,
)


display = [
    "phase4C_candidate_id",
    "candidate_track",
    "zinc_id",
    "qsar_predicted_pIC50",
    "max_training_tanimoto",
    "rf_tree_sd_pIC50",
    "top1_vina_score",
    "reference_contact_jaccard",
    "key_ATP_contact_coverage",
    "MET793_N_hetero_distance_A",
    "MET793_O_hetero_distance_A",
    "severe_clash_pairs_lt1p5A",
]


print(
    supported[
        display
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


# ============================================================
# Exploratory candidates with most W32-like contact patterns
#
# This is descriptive, not a potency ranking.
# ============================================================

print(
    "\n----------------------------------------"
)

print(
    "TOP 15 EXPLORATORY BY W32-CONTACT JACCARD"
)

print(
    "DESCRIPTIVE ONLY - NOT POTENCY RANKING"
)

print(
    "----------------------------------------"
)


explore = features[
    features[
        "candidate_track"
    ] == "EXPLORATORY_OOD"
].copy()


explore = explore.sort_values(
    [
        "reference_contact_jaccard",
        "key_ATP_contact_coverage",
    ],
    ascending=[
        False,
        False,
    ],
)


print(
    explore[
        display
    ]
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


print(
    "\nSaved:"
)

print(
    FEATURES_OUT
)

print(
    REPORT_OUT
)


print(
    "\n========================================"
)

print(
    "PHASE 4E-1 POSE ANALYSIS: PASS"
)

print(
    "========================================"
)
