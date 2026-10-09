from pathlib import Path
import re

import numpy as np
import pandas as pd

from rdkit import Chem


# ============================================================
# Project paths
# ============================================================

ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

REFERENCE_SDF = (
    ROOT
    / "docking"
    / "2ITY"
    / "ligand"
    / "gefitinib_2ITY_crystal_H.sdf"
)

ROUND1_SDF = (
    ROOT
    / "docking"
    / "2ITY"
    / "redocking"
    / "gefitinib_redocked.sdf"
)

ROUND1_PDBQT = (
    ROOT
    / "docking"
    / "2ITY"
    / "redocking"
    / "gefitinib_redocked.pdbqt"
)

ROUND2_SDF = (
    ROOT
    / "docking"
    / "2ITY"
    / "redocking_Hopt"
    / "gefitinib_redocked_Hopt.sdf"
)

ROUND2_PDBQT = (
    ROOT
    / "docking"
    / "2ITY"
    / "redocking_Hopt"
    / "gefitinib_redocked_Hopt.pdbqt"
)

OUT_CSV = (
    ROOT
    / "docking"
    / "2ITY"
    / "reports"
    / "gefitinib_redocking_round_comparison.csv"
)


# ============================================================
# Utilities
# ============================================================

def load_first_sdf(path):
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
    )

    mol = supplier[0]

    if mol is None:
        raise RuntimeError(
            f"Failed to read reference molecule: {path}"
        )

    return mol


def load_all_sdf(path):
    supplier = Chem.SDMolSupplier(
        str(path),
        removeHs=False,
    )

    mols = [
        mol
        for mol in supplier
        if mol is not None
    ]

    if not mols:
        raise RuntimeError(
            f"No molecules read from: {path}"
        )

    return mols


def read_vina_scores(path):
    scores = []

    pattern = re.compile(
        r"REMARK VINA RESULT:\s+"
        r"([-+]?\d*\.?\d+)"
    )

    with open(path) as handle:
        for line in handle:
            match = pattern.search(line)

            if match:
                scores.append(
                    float(match.group(1))
                )

    return scores


# ============================================================
# Same-coordinate-frame symmetry-aware RMSD
#
# IMPORTANT:
# - no translation
# - no rotation
# - no Kabsch fitting
#
# Only chemically equivalent atom mappings are tested.
# ============================================================

def same_frame_best_rmsd(
    reference_mol,
    pose_mol,
):
    ref = Chem.RemoveHs(reference_mol)
    pose = Chem.RemoveHs(pose_mol)

    if ref.GetNumAtoms() != pose.GetNumAtoms():
        raise RuntimeError(
            "Heavy-atom count mismatch: "
            f"reference={ref.GetNumAtoms()}, "
            f"pose={pose.GetNumAtoms()}"
        )

    ref_conf = ref.GetConformer()
    pose_conf = pose.GetConformer()

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
        raise RuntimeError(
            "No valid graph mapping between "
            "reference and docked pose."
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

        diff = pose_xyz - ref_xyz

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

    return best_rmsd, len(matches)


# ============================================================
# Evaluate one docking round
# ============================================================

def evaluate_round(
    name,
    sdf_path,
    pdbqt_path,
    reference,
):
    poses = load_all_sdf(
        sdf_path
    )

    scores = read_vina_scores(
        pdbqt_path
    )

    records = []

    for mode, pose in enumerate(
        poses,
        start=1,
    ):
        rmsd, mappings = (
            same_frame_best_rmsd(
                reference,
                pose,
            )
        )

        score = (
            scores[mode - 1]
            if mode - 1 < len(scores)
            else np.nan
        )

        records.append(
            {
                "round":
                    name,

                "mode":
                    mode,

                "vina_score_kcal_mol":
                    score,

                "crystal_pose_RMSD_A":
                    rmsd,

                "symmetry_mappings_tested":
                    mappings,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Load reference
# ============================================================

reference = load_first_sdf(
    REFERENCE_SDF
)

reference_heavy = Chem.RemoveHs(
    reference
)

print(
    "Reference heavy atoms:",
    reference_heavy.GetNumAtoms()
)


# ============================================================
# Evaluate both rounds
# ============================================================

round1 = evaluate_round(
    name="Round1_heavyfixed",
    sdf_path=ROUND1_SDF,
    pdbqt_path=ROUND1_PDBQT,
    reference=reference,
)

round2 = evaluate_round(
    name="Round2_Reduce2_Hopt",
    sdf_path=ROUND2_SDF,
    pdbqt_path=ROUND2_PDBQT,
    reference=reference,
)

combined = pd.concat(
    [
        round1,
        round2,
    ],
    ignore_index=True,
)

combined.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Summaries
# ============================================================

def summarize(df):
    top1 = df.iloc[0]

    best_idx = (
        df[
            "crystal_pose_RMSD_A"
        ].idxmin()
    )

    best = df.loc[
        best_idx
    ]

    if (
        top1[
            "crystal_pose_RMSD_A"
        ]
        <= 2.0
    ):
        status = "PASS"

    elif (
        best[
            "crystal_pose_RMSD_A"
        ]
        <= 2.0
    ):
        status = (
            "SAMPLING_PASS_RANKING_FAIL"
        )

    else:
        status = "FAIL"

    return {
        "top1_mode":
            int(top1["mode"]),

        "top1_score":
            float(
                top1[
                    "vina_score_kcal_mol"
                ]
            ),

        "top1_rmsd":
            float(
                top1[
                    "crystal_pose_RMSD_A"
                ]
            ),

        "best_mode":
            int(best["mode"]),

        "best_score":
            float(
                best[
                    "vina_score_kcal_mol"
                ]
            ),

        "best_rmsd":
            float(
                best[
                    "crystal_pose_RMSD_A"
                ]
            ),

        "status":
            status,
    }


s1 = summarize(
    round1
)

s2 = summarize(
    round2
)


# ============================================================
# Print complete Round 2 table
# ============================================================

print(
    "\n======================================"
)

print(
    "ROUND 2 — REDUCE2 H-OPT REDOCKING"
)

print(
    "======================================"
)

print(
    round2.to_string(
        index=False
    )
)


# ============================================================
# Comparative summary
# ============================================================

print(
    "\n======================================"
)

print(
    "REDOCKING ROUND COMPARISON"
)

print(
    "======================================"
)


print(
    "\nRound 1 — heavy-atom repaired receptor"
)

print(
    f"Top-1 RMSD : "
    f"{s1['top1_rmsd']:.3f} A"
)

print(
    f"Best RMSD  : "
    f"{s1['best_rmsd']:.3f} A "
    f"(mode {s1['best_mode']})"
)

print(
    f"Top-1 score: "
    f"{s1['top1_score']:.3f} kcal/mol"
)

print(
    f"Status     : "
    f"{s1['status']}"
)


print(
    "\nRound 2 — Reduce2 H-optimized receptor"
)

print(
    f"Top-1 RMSD : "
    f"{s2['top1_rmsd']:.3f} A"
)

print(
    f"Best RMSD  : "
    f"{s2['best_rmsd']:.3f} A "
    f"(mode {s2['best_mode']})"
)

print(
    f"Top-1 score: "
    f"{s2['top1_score']:.3f} kcal/mol"
)

print(
    f"Status     : "
    f"{s2['status']}"
)


# ============================================================
# Change after H optimization
# ============================================================

delta_top1 = (
    s2["top1_rmsd"]
    - s1["top1_rmsd"]
)

delta_best = (
    s2["best_rmsd"]
    - s1["best_rmsd"]
)


print(
    "\n--------------------------------------"
)

print(
    "EFFECT OF REDUCE2 H-OPT"
)

print(
    "--------------------------------------"
)

print(
    f"Top-1 RMSD change: "
    f"{delta_top1:+.3f} A"
)

print(
    f"Best RMSD change : "
    f"{delta_best:+.3f} A"
)


if (
    s2["status"] == "PASS"
):
    interpretation = (
        "PASS: receptor H optimization "
        "enabled the top-ranked pose to "
        "reproduce the crystal pose."
    )

elif (
    s2["status"]
    == "SAMPLING_PASS_RANKING_FAIL"
):
    interpretation = (
        "PARTIAL PASS: a crystal-like pose "
        "was sampled, but Vina did not rank "
        "it first."
    )

elif (
    s2["best_rmsd"]
    < s1["best_rmsd"]
):
    interpretation = (
        "IMPROVED BUT STILL FAIL: "
        "H optimization improved pose "
        "reproduction, but RMSD remains "
        "above 2.0 A."
    )

else:
    interpretation = (
        "NO MEANINGFUL RESCUE: "
        "H optimization did not recover "
        "a crystal-like pose. "
        "Further tuning of 2ITY should "
        "not be prioritized."
    )


print(
    "\nInterpretation:"
)

print(
    interpretation
)


print(
    "\nSaved comparison:"
)

print(
    OUT_CSV
)
