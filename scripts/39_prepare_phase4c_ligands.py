from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)

INPUT = (
    ROOT
    / "results/zinc15_screening/"
      "zinc15_phase4C_candidates.csv"
)

OUT_ROOT = (
    ROOT
    / "docking/3W32/phase4C"
)

LIGAND_DIR = (
    OUT_ROOT
    / "ligands"
)

REPORT_DIR = (
    OUT_ROOT
    / "reports"
)

LIGAND_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_OUT = (
    REPORT_DIR
    / "phase4C_ligand_preparation.csv"
)

REPORT_OUT = (
    REPORT_DIR
    / "phase4C_ligand_preparation_report.json"
)


# ============================================================
# Settings
# ============================================================

EMBED_SEED = 42

MEEKO_TIMEOUT = 180

MAX_EMBED_ATTEMPTS = 3


# ============================================================
# Helpers
# ============================================================

def formal_charge(mol):

    return int(
        sum(
            atom.GetFormalCharge()
            for atom in mol.GetAtoms()
        )
    )


def embed_and_optimize(
    smiles,
    sdf_path,
):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:

        return (
            False,
            None,
            None,
            "MolFromSmiles_failed",
        )


    mol = Chem.AddHs(
        mol
    )


    params = (
        AllChem.ETKDGv3()
    )

    params.randomSeed = (
        EMBED_SEED
    )

    params.useRandomCoords = False


    embed_result = -1


    for attempt in range(
        MAX_EMBED_ATTEMPTS
    ):

        if attempt > 0:

            params.useRandomCoords = True

            params.randomSeed = (
                EMBED_SEED
                + attempt
            )


        embed_result = (
            AllChem.EmbedMolecule(
                mol,
                params,
            )
        )


        if embed_result == 0:

            break


    if embed_result != 0:

        return (
            False,
            None,
            None,
            "ETKDG_embedding_failed",
        )


    optimization = "NONE"


    # ----------------------------------------
    # MMFF94 first
    # ----------------------------------------

    try:

        if (
            AllChem.MMFFHasAllMoleculeParams(
                mol
            )
        ):

            AllChem.MMFFOptimizeMolecule(
                mol,
                mmffVariant="MMFF94",
                maxIters=1000,
            )

            optimization = "MMFF94"

        else:

            raise ValueError(
                "MMFF parameters unavailable"
            )

    except Exception:

        # ------------------------------------
        # UFF fallback
        # ------------------------------------

        try:

            AllChem.UFFOptimizeMolecule(
                mol,
                maxIters=1000,
            )

            optimization = "UFF"

        except Exception as exc:

            return (
                False,
                None,
                None,
                "geometry_optimization_failed: "
                + str(exc)[:200],
            )


    # ----------------------------------------
    # Write SDF
    # ----------------------------------------

    writer = Chem.SDWriter(
        str(
            sdf_path
        )
    )


    if writer is None:

        return (
            False,
            None,
            None,
            "SDWriter_failed",
        )


    writer.write(
        mol
    )

    writer.close()


    if (
        not sdf_path.exists()
        or sdf_path.stat().st_size == 0
    ):

        return (
            False,
            None,
            None,
            "empty_SDF",
        )


    return (
        True,
        mol,
        optimization,
        "",
    )


def prepare_meeko(
    sdf_path,
    pdbqt_path,
    log_path,
):

    cmd = [
        "mk_prepare_ligand.py",
        "-i",
        str(
            sdf_path
        ),
        "-o",
        str(
            pdbqt_path
        ),
    ]


    try:

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=MEEKO_TIMEOUT,
        )

    except subprocess.TimeoutExpired:

        log_path.write_text(
            "MEEKO TIMEOUT\n"
        )

        return (
            False,
            -15,
            "timeout",
        )


    log_path.write_text(
        result.stdout
        if result.stdout
        else ""
    )


    valid = (
        result.returncode == 0
        and pdbqt_path.exists()
        and pdbqt_path.stat().st_size > 0
    )


    if valid:

        return (
            True,
            result.returncode,
            "",
        )


    return (
        False,
        result.returncode,
        "Meeko_failed",
    )


# ============================================================
# Load candidates
# ============================================================

df = pd.read_csv(
    INPUT
)


print(
    "\n========================================"
)

print(
    "PHASE 4D-1 LIGAND PREPARATION"
)

print(
    "========================================"
)


print(
    "\nCandidates:",
    len(df)
)


required = [
    "phase4C_candidate_id",
    "candidate_track",
    "zinc_id",
    "standardized_smiles",
    "qsar_predicted_pIC50",
    "max_training_tanimoto",
    "rf_tree_sd_pIC50",
]


missing = [
    x
    for x in required
    if x not in df.columns
]


if missing:

    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# ============================================================
# Process
# ============================================================

records = []


for i, row in df.iterrows():

    cid = str(
        row[
            "phase4C_candidate_id"
        ]
    )

    zinc_id = str(
        row[
            "zinc_id"
        ]
    )

    smiles = str(
        row[
            "standardized_smiles"
        ]
    )


    ligand_subdir = (
        LIGAND_DIR
        / cid
    )

    ligand_subdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    sdf_path = (
        ligand_subdir
        / f"{cid}.sdf"
    )

    pdbqt_path = (
        ligand_subdir
        / f"{cid}.pdbqt"
    )

    log_path = (
        ligand_subdir
        / f"{cid}_meeko.log"
    )


    print(
        f"\n[{i+1:02d}/{len(df):02d}] "
        f"{cid} | {zinc_id}",
        flush=True,
    )


    # ----------------------------------------
    # If valid PDBQT already exists,
    # reuse it.
    # ----------------------------------------

    if (
        pdbqt_path.exists()
        and pdbqt_path.stat().st_size > 0
    ):

        mol = Chem.MolFromSmiles(
            smiles
        )

        charge = (
            formal_charge(
                mol
            )
            if mol is not None
            else np.nan
        )


        records.append(
            {
                "phase4C_candidate_id":
                    cid,

                "candidate_track":
                    row[
                        "candidate_track"
                    ],

                "zinc_id":
                    zinc_id,

                "standardized_smiles":
                    smiles,

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

                "formal_charge":
                    charge,

                "optimization":
                    "REUSED",

                "status":
                    "PASS_EXISTING",

                "error":
                    "",

                "sdf_path":
                    str(
                        sdf_path
                    ),

                "pdbqt_path":
                    str(
                        pdbqt_path
                    ),
            }
        )


        print(
            "  REUSE existing PDBQT",
            flush=True,
        )

        continue


    # ----------------------------------------
    # Generate 3D
    # ----------------------------------------

    ok, mol, optimization, error = (
        embed_and_optimize(
            smiles,
            sdf_path,
        )
    )


    if not ok:

        records.append(
            {
                "phase4C_candidate_id":
                    cid,

                "candidate_track":
                    row[
                        "candidate_track"
                    ],

                "zinc_id":
                    zinc_id,

                "standardized_smiles":
                    smiles,

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

                "formal_charge":
                    np.nan,

                "optimization":
                    "",

                "status":
                    "FAIL",

                "error":
                    error,

                "sdf_path":
                    str(
                        sdf_path
                    ),

                "pdbqt_path":
                    str(
                        pdbqt_path
                    ),
            }
        )


        print(
            "  FAIL:",
            error,
            flush=True,
        )

        continue


    charge = formal_charge(
        mol
    )


    # ----------------------------------------
    # Meeko
    # ----------------------------------------

    meeko_ok, exit_code, meeko_error = (
        prepare_meeko(
            sdf_path,
            pdbqt_path,
            log_path,
        )
    )


    if meeko_ok:

        status = "PASS"
        error = ""

        print(
            f"  PASS | "
            f"{optimization} | "
            f"charge={charge:+d}",
            flush=True,
        )

    else:

        status = "FAIL"

        error = (
            f"{meeko_error}; "
            f"exit={exit_code}"
        )

        print(
            "  FAIL:",
            error,
            flush=True,
        )


    records.append(
        {
            "phase4C_candidate_id":
                cid,

            "candidate_track":
                row[
                    "candidate_track"
                ],

            "zinc_id":
                zinc_id,

            "standardized_smiles":
                smiles,

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

            "formal_charge":
                charge,

            "optimization":
                optimization,

            "status":
                status,

            "error":
                error,

            "sdf_path":
                str(
                    sdf_path
                ),

            "pdbqt_path":
                str(
                    pdbqt_path
                ),
        }
    )


# ============================================================
# Save summary
# ============================================================

result = pd.DataFrame(
    records
)


result.to_csv(
    SUMMARY_OUT,
    index=False,
)


success_mask = (
    result[
        "status"
    ].isin(
        [
            "PASS",
            "PASS_EXISTING",
        ]
    )
)


n_success = int(
    success_mask.sum()
)

n_fail = int(
    (~success_mask).sum()
)


track_summary = {}


for track in result[
    "candidate_track"
].unique():

    sub = result[
        result[
            "candidate_track"
        ] == track
    ]

    sub_ok = (
        sub[
            "status"
        ].isin(
            [
                "PASS",
                "PASS_EXISTING",
            ]
        )
    )


    track_summary[
        str(track)
    ] = {
        "total":
            int(
                len(sub)
            ),

        "pass":
            int(
                sub_ok.sum()
            ),

        "fail":
            int(
                (~sub_ok).sum()
            ),
    }


charge_counts = (
    result.loc[
        success_mask,
        "formal_charge"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


optimization_counts = (
    result.loc[
        success_mask,
        "optimization"
    ]
    .value_counts(
        dropna=False
    )
)


report = {
    "input_candidates":
        int(
            len(df)
        ),

    "successful_ligand_preparations":
        n_success,

    "failed_ligand_preparations":
        n_fail,

    "track_summary":
        track_summary,

    "formal_charge_distribution": {
        str(k):
            int(v)
        for k, v
        in charge_counts.items()
    },

    "optimization_distribution": {
        str(k):
            int(v)
        for k, v
        in optimization_counts.items()
    },

    "protocol": {
        "input":
            "standardized SMILES",

        "3D_embedding":
            "RDKit ETKDGv3",

        "random_seed":
            EMBED_SEED,

        "geometry_optimization":
            "MMFF94 with UFF fallback",

        "hydrogens":
            "RDKit AddHs",

        "PDBQT":
            "Meeko",

        "protonation_note":
            (
                "No explicit pH-dependent microstate "
                "enumeration was performed. "
                "Standardized input charge states "
                "were used."
            ),
    },

    "summary_csv":
        str(
            SUMMARY_OUT
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
    "PHASE 4D-1 LIGAND PREPARATION SUMMARY"
)

print(
    "========================================"
)


print(
    "\nInput candidates:",
    len(df)
)

print(
    "PASS:",
    n_success
)

print(
    "FAIL:",
    n_fail
)


print(
    "\nTrack summary:"
)

for track, values in (
    track_summary.items()
):

    print(
        f"{track}: "
        f"{values['pass']}/"
        f"{values['total']} PASS"
    )


print(
    "\nFormal charge distribution:"
)

print(
    charge_counts.to_string()
)


print(
    "\nOptimization methods:"
)

print(
    optimization_counts.to_string()
)


if n_fail > 0:

    print(
        "\nFAILED LIGANDS:"
    )

    print(
        result.loc[
            ~success_mask,
            [
                "phase4C_candidate_id",
                "zinc_id",
                "error",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\nSaved:"
)

print(
    SUMMARY_OUT
)

print(
    REPORT_OUT
)


print(
    "\n========================================"
)

if n_fail == 0:

    print(
        "PHASE 4D-1 STATUS: PASS"
    )

else:

    print(
        "PHASE 4D-1 STATUS: "
        "PASS_WITH_FAILURES"
    )

print(
    "========================================"
)
