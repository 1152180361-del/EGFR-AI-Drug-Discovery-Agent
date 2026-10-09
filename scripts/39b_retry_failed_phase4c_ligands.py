from pathlib import Path
import json
import subprocess

import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

SUMMARY = (
    ROOT
    / "docking/3W32/phase4C/reports/"
      "phase4C_ligand_preparation.csv"
)

REPORT = (
    ROOT
    / "docking/3W32/phase4C/reports/"
      "phase4C_ligand_preparation_report.json"
)

LIGAND_DIR = (
    ROOT
    / "docking/3W32/phase4C/ligands"
)

MEEKO_TIMEOUT = 180


def fresh_mol(smiles):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    return Chem.AddHs(
        mol
    )


def try_etkdg_v3(smiles):

    for seed in range(
        1001,
        1021,
    ):

        mol = fresh_mol(
            smiles
        )

        if mol is None:
            return None, None


        params = (
            AllChem.ETKDGv3()
        )

        params.randomSeed = seed
        params.useRandomCoords = True
        params.enforceChirality = True


        if hasattr(
            params,
            "maxIterations"
        ):
            params.maxIterations = 2000


        if hasattr(
            params,
            "ignoreSmoothingFailures"
        ):
            params.ignoreSmoothingFailures = True


        if hasattr(
            params,
            "useSmallRingTorsions"
        ):
            params.useSmallRingTorsions = True


        if hasattr(
            params,
            "useMacrocycleTorsions"
        ):
            params.useMacrocycleTorsions = True


        code = AllChem.EmbedMolecule(
            mol,
            params,
        )


        if code == 0:
            return (
                mol,
                f"ETKDGv3_random_seed_{seed}"
            )


    return None, None


def try_etkdg_v2(smiles):

    for seed in range(
        2001,
        2021,
    ):

        mol = fresh_mol(
            smiles
        )

        if mol is None:
            return None, None


        params = (
            AllChem.ETKDGv2()
        )

        params.randomSeed = seed
        params.useRandomCoords = True
        params.enforceChirality = True


        if hasattr(
            params,
            "maxIterations"
        ):
            params.maxIterations = 3000


        if hasattr(
            params,
            "ignoreSmoothingFailures"
        ):
            params.ignoreSmoothingFailures = True


        code = AllChem.EmbedMolecule(
            mol,
            params,
        )


        if code == 0:
            return (
                mol,
                f"ETKDGv2_random_seed_{seed}"
            )


    return None, None


def try_distance_geometry(smiles):

    for seed in range(
        3001,
        3021,
    ):

        mol = fresh_mol(
            smiles
        )

        if mol is None:
            return None, None


        try:

            code = AllChem.EmbedMolecule(
                mol,
                randomSeed=seed,
                useRandomCoords=True,
                useBasicKnowledge=False,
                useExpTorsionAnglePrefs=False,
                enforceChirality=True,
                ignoreSmoothingFailures=True,
            )

        except Exception:
            continue


        if code == 0:
            return (
                mol,
                f"distance_geometry_seed_{seed}"
            )


    return None, None


def optimize(mol):

    try:

        if AllChem.MMFFHasAllMoleculeParams(
            mol
        ):

            AllChem.MMFFOptimizeMolecule(
                mol,
                mmffVariant="MMFF94",
                maxIters=2000,
            )

            return "MMFF94"

    except Exception:
        pass


    try:

        AllChem.UFFOptimizeMolecule(
            mol,
            maxIters=2000,
        )

        return "UFF"

    except Exception:

        return None


def write_sdf(
    mol,
    path,
):

    writer = Chem.SDWriter(
        str(path)
    )

    writer.write(
        mol
    )

    writer.close()


def run_meeko(
    sdf,
    pdbqt,
    log,
):

    cmd = [
        "mk_prepare_ligand.py",
        "-i",
        str(sdf),
        "-o",
        str(pdbqt),
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

        log.write_text(
            "MEEKO TIMEOUT\n"
        )

        return False


    log.write_text(
        result.stdout or ""
    )


    return (
        result.returncode == 0
        and pdbqt.exists()
        and pdbqt.stat().st_size > 0
    )


df = pd.read_csv(
    SUMMARY
)


failed_indices = df.index[
    df[
        "status"
    ] == "FAIL"
].tolist()


print(
    "\n========================================"
)

print(
    "PHASE 4D-1 FAILED LIGAND RETRY"
)

print(
    "========================================"
)

print(
    "\nFailed ligands to retry:",
    len(failed_indices)
)


for idx in failed_indices:

    row = df.loc[
        idx
    ]


    cid = str(
        row[
            "phase4C_candidate_id"
        ]
    )

    zinc = str(
        row[
            "zinc_id"
        ]
    )

    smiles = str(
        row[
            "standardized_smiles"
        ]
    )


    subdir = (
        LIGAND_DIR
        / cid
    )

    subdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    sdf = (
        subdir
        / f"{cid}.sdf"
    )

    pdbqt = (
        subdir
        / f"{cid}.pdbqt"
    )

    log = (
        subdir
        / f"{cid}_meeko_retry.log"
    )


    print(
        f"\nRetrying {cid} | {zinc}"
    )


    mol = None
    strategy = None


    # ----------------------------------------
    # Strategy 1
    # ----------------------------------------

    mol, strategy = (
        try_etkdg_v3(
            smiles
        )
    )


    # ----------------------------------------
    # Strategy 2
    # ----------------------------------------

    if mol is None:

        mol, strategy = (
            try_etkdg_v2(
                smiles
            )
        )


    # ----------------------------------------
    # Strategy 3
    # ----------------------------------------

    if mol is None:

        mol, strategy = (
            try_distance_geometry(
                smiles
            )
        )


    if mol is None:

        print(
            "  FAIL: all embedding strategies failed"
        )

        df.loc[
            idx,
            "error"
        ] = (
            "all_retry_embedding_strategies_failed"
        )

        continue


    print(
        "  Embedded with:",
        strategy
    )


    opt = optimize(
        mol
    )


    if opt is None:

        print(
            "  FAIL: geometry optimization failed"
        )

        df.loc[
            idx,
            "error"
        ] = (
            "retry_geometry_optimization_failed"
        )

        continue


    write_sdf(
        mol,
        sdf,
    )


    meeko_ok = run_meeko(
        sdf,
        pdbqt,
        log,
    )


    if not meeko_ok:

        print(
            "  FAIL: Meeko preparation failed"
        )

        df.loc[
            idx,
            "error"
        ] = (
            "retry_Meeko_failed"
        )

        continue


    charge = int(
        Chem.GetFormalCharge(
            mol
        )
    )


    df.loc[
        idx,
        "formal_charge"
    ] = charge


    df.loc[
        idx,
        "optimization"
    ] = (
        opt
        + "_RETRY_"
        + strategy
    )


    df.loc[
        idx,
        "status"
    ] = "PASS"


    df.loc[
        idx,
        "error"
    ] = ""


    df.loc[
        idx,
        "sdf_path"
    ] = str(
        sdf
    )


    df.loc[
        idx,
        "pdbqt_path"
    ] = str(
        pdbqt
    )


    print(
        f"  PASS | {opt} | "
        f"charge={charge:+d}"
    )


# ============================================================
# Save updated summary
# ============================================================

df.to_csv(
    SUMMARY,
    index=False,
)


success = df[
    "status"
].isin(
    [
        "PASS",
        "PASS_EXISTING",
    ]
)


n_success = int(
    success.sum()
)

n_fail = int(
    (~success).sum()
)


track_summary = {}


for track in df[
    "candidate_track"
].unique():

    sub = df[
        df[
            "candidate_track"
        ]
        == track
    ]

    ok = sub[
        "status"
    ].isin(
        [
            "PASS",
            "PASS_EXISTING",
        ]
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
                ok.sum()
            ),

        "fail":
            int(
                (~ok).sum()
            ),
    }


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

    "retry_completed":
        True,

    "protocol_note":
        (
            "Failed ETKDGv3 embeddings were retried "
            "using multiple random-coordinate ETKDGv3 "
            "seeds, ETKDGv2 fallback, and distance-"
            "geometry fallback while preserving "
            "stereochemistry."
        ),
}


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


print(
    "\n========================================"
)

print(
    "RETRY SUMMARY"
)

print(
    "========================================"
)


print(
    "\nPASS:",
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


if n_fail > 0:

    print(
        "\nRemaining failures:"
    )

    print(
        df.loc[
            ~success,
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
    "\n========================================"
)

if n_fail == 0:

    print(
        "PHASE 4D-1 FINAL STATUS: PASS"
    )

else:

    print(
        "PHASE 4D-1 FINAL STATUS: "
        "PASS_WITH_REMAINING_FAILURES"
    )

print(
    "========================================"
)
