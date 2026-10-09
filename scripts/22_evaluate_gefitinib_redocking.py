from pathlib import Path
import re

import numpy as np
import pandas as pd

from rdkit import Chem


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_SDF = (
    ROOT
    / "docking"
    / "2ITY"
    / "ligand"
    / "gefitinib_2ITY_crystal_H.sdf"
)

DOCKED_SDF = (
    ROOT
    / "docking"
    / "2ITY"
    / "redocking"
    / "gefitinib_redocked.sdf"
)

DOCKED_PDBQT = (
    ROOT
    / "docking"
    / "2ITY"
    / "redocking"
    / "gefitinib_redocked.pdbqt"
)

REPORT_FILE = (
    ROOT
    / "docking"
    / "2ITY"
    / "reports"
    / "gefitinib_redocking_rmsd.csv"
)


# =========================================
# Load reference ligand
# =========================================

reference_supplier = (
    Chem.SDMolSupplier(
        str(REFERENCE_SDF),
        removeHs=False,
    )
)

reference = (
    reference_supplier[0]
)

if reference is None:
    raise RuntimeError(
        "Failed to load reference SDF."
    )


# =========================================
# Remove hydrogens
#
# RMSD is evaluated using heavy atoms only.
# =========================================

reference_heavy = Chem.RemoveHs(
    reference
)

reference_conf = (
    reference_heavy.GetConformer()
)

reference_xyz = np.asarray(
    [
        list(
            reference_conf.GetAtomPosition(i)
        )
        for i in range(
            reference_heavy.GetNumAtoms()
        )
    ],
    dtype=float,
)


print(
    "Reference heavy atoms:",
    reference_heavy.GetNumAtoms()
)


# =========================================
# Read Vina scores from PDBQT
# =========================================

scores = []

pattern = re.compile(
    r"REMARK VINA RESULT:\s+"
    r"([-+]?\d*\.?\d+)"
)


with open(
    DOCKED_PDBQT,
    "r"
) as handle:

    for line in handle:

        match = pattern.search(
            line
        )

        if match:

            scores.append(
                float(
                    match.group(1)
                )
            )


print(
    "Vina scores found:",
    len(scores)
)


# =========================================
# Load docked poses
# =========================================

supplier = Chem.SDMolSupplier(
    str(DOCKED_SDF),
    removeHs=False,
)


poses = [
    mol
    for mol in supplier
    if mol is not None
]


if not poses:

    raise RuntimeError(
        "No docked poses loaded."
    )


print(
    "Docked poses:",
    len(poses)
)


# =========================================
# Same-frame symmetry-aware RMSD
#
# IMPORTANT:
#
# No Kabsch alignment.
# No translation.
# No rotation.
#
# We only try chemically equivalent
# graph mappings and calculate coordinate
# RMSD directly in receptor coordinates.
# =========================================

def same_frame_best_rmsd(
    reference_mol,
    pose_mol,
):

    ref = Chem.RemoveHs(
        reference_mol
    )

    pose = Chem.RemoveHs(
        pose_mol
    )


    if (
        ref.GetNumAtoms()
        != pose.GetNumAtoms()
    ):

        raise RuntimeError(
            "Heavy atom count mismatch: "
            f"reference={ref.GetNumAtoms()}, "
            f"pose={pose.GetNumAtoms()}"
        )


    ref_conf = (
        ref.GetConformer()
    )

    pose_conf = (
        pose.GetConformer()
    )


    ref_xyz = np.asarray(
        [
            list(
                ref_conf.GetAtomPosition(i)
            )
            for i in range(
                ref.GetNumAtoms()
            )
        ],
        dtype=float,
    )


    # -------------------------------------
    # Find all chemically valid mappings
    #
    # pose.GetSubstructMatches(ref)
    # returns pose atom indices corresponding
    # to reference atom indices.
    #
    # Multiple mappings account for molecular
    # graph symmetry.
    # -------------------------------------

    matches = pose.GetSubstructMatches(
        ref,
        uniquify=False,
        maxMatches=100000,
        useChirality=True,
    )


    # If strict chirality mapping fails,
    # retry without chirality.
    if not matches:

        matches = pose.GetSubstructMatches(
            ref,
            uniquify=False,
            maxMatches=100000,
            useChirality=False,
        )


    if not matches:

        raise RuntimeError(
            "No graph mapping between "
            "reference and docked pose."
        )


    best_rmsd = None
    best_match = None


    for match in matches:

        pose_xyz = np.asarray(
            [
                list(
                    pose_conf.GetAtomPosition(
                        pose_index
                    )
                )
                for pose_index
                in match
            ],
            dtype=float,
        )


        differences = (
            pose_xyz
            - ref_xyz
        )


        rmsd = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        differences ** 2,
                        axis=1,
                    )
                )
            )
        )


        if (
            best_rmsd is None
            or rmsd < best_rmsd
        ):

            best_rmsd = rmsd
            best_match = match


    return (
        best_rmsd,
        len(matches),
    )


# =========================================
# Evaluate all poses
# =========================================

records = []


for i, pose in enumerate(
    poses,
    start=1,
):

    rmsd, n_mappings = (
        same_frame_best_rmsd(
            reference,
            pose,
        )
    )


    vina_score = (
        scores[i - 1]
        if i - 1 < len(scores)
        else np.nan
    )


    records.append(
        {
            "mode":
                i,

            "vina_score_kcal_mol":
                vina_score,

            "crystal_pose_RMSD_A":
                rmsd,

            "symmetry_mappings_tested":
                n_mappings,
        }
    )


results = pd.DataFrame(
    records
)


results.to_csv(
    REPORT_FILE,
    index=False,
)


# =========================================
# Identify top-ranked pose
# =========================================

top_pose = (
    results.iloc[0]
)


# =========================================
# Identify geometrically closest pose
# =========================================

best_index = (
    results[
        "crystal_pose_RMSD_A"
    ].idxmin()
)

best_pose = (
    results.loc[
        best_index
    ]
)


# =========================================
# Protocol interpretation
# =========================================

TOP1_THRESHOLD = 2.0


top1_pass = (
    top_pose[
        "crystal_pose_RMSD_A"
    ]
    <= TOP1_THRESHOLD
)


any_pose_pass = (
    best_pose[
        "crystal_pose_RMSD_A"
    ]
    <= TOP1_THRESHOLD
)


if top1_pass:

    protocol_status = (
        "PASS"
    )

elif any_pose_pass:

    protocol_status = (
        "SAMPLING_PASS_RANKING_FAIL"
    )

else:

    protocol_status = (
        "FAIL"
    )


# =========================================
# Terminal output
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 3.5 — GEFITINIB REDOCKING"
)

print(
    "================================="
)


print(
    "\nDocked pose results:"
)


print(
    results.to_string(
        index=False
    )
)


print(
    "\n---------------------------------"
)

print(
    "TOP-RANKED POSE"
)

print(
    "---------------------------------"
)


print(
    "Mode:",
    int(
        top_pose[
            "mode"
        ]
    )
)

print(
    "Vina score:",
    round(
        float(
            top_pose[
                "vina_score_kcal_mol"
            ]
        ),
        3,
    )
)

print(
    "Crystal RMSD:",
    round(
        float(
            top_pose[
                "crystal_pose_RMSD_A"
            ]
        ),
        3,
    ),
    "A",
)


print(
    "\n---------------------------------"
)

print(
    "BEST GEOMETRIC POSE"
)

print(
    "---------------------------------"
)


print(
    "Mode:",
    int(
        best_pose[
            "mode"
        ]
    )
)

print(
    "Vina score:",
    round(
        float(
            best_pose[
                "vina_score_kcal_mol"
            ]
        ),
        3,
    )
)

print(
    "Crystal RMSD:",
    round(
        float(
            best_pose[
                "crystal_pose_RMSD_A"
            ]
        ),
        3,
    ),
    "A",
)


print(
    "\n================================="
)

print(
    "REDOCKING STATUS:",
    protocol_status
)

print(
    "================================="
)


if protocol_status == "PASS":

    print(
        "Top-ranked docking pose "
        "reproduces the crystal pose "
        "within 2.0 A."
    )


elif (
    protocol_status
    == "SAMPLING_PASS_RANKING_FAIL"
):

    print(
        "A near-native pose was sampled, "
        "but Vina did not rank it first."
    )


else:

    print(
        "No docked pose reproduces "
        "the crystal pose within 2.0 A."
    )


print(
    "\nSaved RMSD report:"
)

print(
    REPORT_FILE
)
