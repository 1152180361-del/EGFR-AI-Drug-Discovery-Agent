from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

REFERENCE_SDF = (
    ROOT
    / "docking/3W32/ligand/W32_3W32_crystal_H.sdf"
)

DOCKED_SDF = (
    ROOT
    / "docking/3W32/redocking/W32_redocked.sdf"
)

DOCKED_PDBQT = (
    ROOT
    / "docking/3W32/redocking/W32_redocked.pdbqt"
)

REPORT_OUT = (
    ROOT
    / "docking/3W32/reports/W32_redocking_RMSD.csv"
)


# ============================================================
# Load reference
# ============================================================

ref_supplier = Chem.SDMolSupplier(
    str(REFERENCE_SDF),
    removeHs=False,
)

reference = ref_supplier[0]

if reference is None:
    raise RuntimeError(
        "Failed to load crystal reference W32."
    )


reference_heavy = Chem.RemoveHs(
    reference
)

print(
    "Reference heavy atoms:",
    reference_heavy.GetNumAtoms()
)


# ============================================================
# Read Vina scores
# ============================================================

scores = []

pattern = re.compile(
    r"REMARK VINA RESULT:\s+"
    r"([-+]?\d*\.?\d+)"
)

with open(DOCKED_PDBQT) as f:

    for line in f:

        match = pattern.search(line)

        if match:
            scores.append(
                float(match.group(1))
            )


print(
    "Vina scores found:",
    len(scores)
)


# ============================================================
# Load docked poses
# ============================================================

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
        "No docked W32 poses loaded."
    )


print(
    "Docked poses:",
    len(poses)
)


# ============================================================
# Same-frame symmetry-aware RMSD
#
# Important:
# NO rotation
# NO translation
# NO Kabsch alignment
#
# Only chemically equivalent atom mappings
# are permitted.
# ============================================================

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

    # ----------------------------------------
    # Find chemically equivalent atom mappings
    # ----------------------------------------

    matches = pose.GetSubstructMatches(
        ref,
        uniquify=False,
        maxMatches=100000,
        useChirality=True,
    )

    if not matches:

        matches = pose.GetSubstructMatches(
            ref,
            uniquify=False,
            maxMatches=100000,
            useChirality=False,
        )

    if not matches:

        # Try reverse mapping as a fallback
        reverse_matches = ref.GetSubstructMatches(
            pose,
            uniquify=False,
            maxMatches=100000,
            useChirality=False,
        )

        if not reverse_matches:
            raise RuntimeError(
                "No chemical graph mapping "
                "between crystal and docked W32."
            )

        best_rmsd = None
        n_mappings = 0

        for reverse_match in reverse_matches:

            pose_xyz = np.asarray(
                [
                    list(
                        pose_conf.GetAtomPosition(i)
                    )
                    for i in range(
                        pose.GetNumAtoms()
                    )
                ],
                dtype=float,
            )

            ordered_ref_xyz = np.asarray(
                [
                    list(
                        ref_conf.GetAtomPosition(
                            ref_index
                        )
                    )
                    for ref_index
                    in reverse_match
                ],
                dtype=float,
            )

            diff = (
                pose_xyz
                - ordered_ref_xyz
            )

            rmsd = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            diff ** 2,
                            axis=1,
                        )
                    )
                )
            )

            n_mappings += 1

            if (
                best_rmsd is None
                or rmsd < best_rmsd
            ):
                best_rmsd = rmsd

        return (
            best_rmsd,
            n_mappings,
        )


    best_rmsd = None

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

        diff = (
            pose_xyz
            - ref_xyz
        )

        rmsd = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        diff ** 2,
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

    return (
        best_rmsd,
        len(matches),
    )


# ============================================================
# Evaluate all poses
# ============================================================

records = []

for mode, pose in enumerate(
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
        scores[mode - 1]
        if mode - 1 < len(scores)
        else np.nan
    )

    records.append(
        {
            "mode":
                mode,

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
    REPORT_OUT,
    index=False,
)


# ============================================================
# Top-ranked and geometrically best poses
# ============================================================

top1 = results.iloc[0]

best_idx = (
    results[
        "crystal_pose_RMSD_A"
    ].idxmin()
)

best = results.loc[
    best_idx
]


# ============================================================
# Validation classification
# ============================================================

THRESHOLD = 2.0

if (
    top1[
        "crystal_pose_RMSD_A"
    ]
    <= THRESHOLD
):

    status = "PASS"

elif (
    best[
        "crystal_pose_RMSD_A"
    ]
    <= THRESHOLD
):

    status = (
        "SAMPLING_PASS_RANKING_FAIL"
    )

else:

    status = "FAIL"


# ============================================================
# Output
# ============================================================

print(
    "\n======================================"
)

print(
    "3W32 / W32 SELF-REDOCKING VALIDATION"
)

print(
    "======================================"
)

print(
    "\nAll poses:"
)

print(
    results.to_string(
        index=False
    )
)


print(
    "\n--------------------------------------"
)

print(
    "TOP-RANKED POSE"
)

print(
    "--------------------------------------"
)

print(
    "Mode:",
    int(top1["mode"])
)

print(
    "Vina score:",
    f"{top1['vina_score_kcal_mol']:.3f}",
    "kcal/mol"
)

print(
    "Crystal RMSD:",
    f"{top1['crystal_pose_RMSD_A']:.3f}",
    "A"
)


print(
    "\n--------------------------------------"
)

print(
    "BEST GEOMETRIC POSE"
)

print(
    "--------------------------------------"
)

print(
    "Mode:",
    int(best["mode"])
)

print(
    "Vina score:",
    f"{best['vina_score_kcal_mol']:.3f}",
    "kcal/mol"
)

print(
    "Crystal RMSD:",
    f"{best['crystal_pose_RMSD_A']:.3f}",
    "A"
)


print(
    "\n======================================"
)

print(
    "REDOCKING STATUS:",
    status
)

print(
    "======================================"
)


if status == "PASS":

    print(
        "Top-ranked pose reproduces "
        "the experimental crystal pose "
        "within 2.0 A."
    )

elif (
    status
    == "SAMPLING_PASS_RANKING_FAIL"
):

    print(
        "A crystal-like pose was sampled, "
        "but Vina did not rank it first."
    )

else:

    print(
        "No pose reproduces the crystal "
        "pose within 2.0 A."
    )


print(
    "\nSaved report:"
)

print(
    REPORT_OUT
)
