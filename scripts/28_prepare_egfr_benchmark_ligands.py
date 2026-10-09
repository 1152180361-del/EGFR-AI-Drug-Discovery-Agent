from pathlib import Path
import json
import subprocess

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

BENCHMARK_CSV = (
    ROOT
    / "docking/3W32/benchmark/egfr_retrospective_benchmark.csv"
)

BASE_DIR = (
    ROOT
    / "docking/3W32/benchmark"
)

SDF_DIR = BASE_DIR / "sdf"
PDBQT_DIR = BASE_DIR / "pdbqt"
LOG_DIR = BASE_DIR / "logs"

MANIFEST_OUT = (
    BASE_DIR
    / "benchmark_ligand_preparation_manifest.csv"
)

REPORT_OUT = (
    BASE_DIR
    / "benchmark_ligand_preparation_report.json"
)


for d in [
    SDF_DIR,
    PDBQT_DIR,
    LOG_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Settings
# ============================================================

BASE_SEED = 42


# ============================================================
# Load benchmark
# ============================================================

df = pd.read_csv(
    BENCHMARK_CSV
)

required = [
    "benchmark_id",
    "canonical_smiles",
    "benchmark_label",
    "binary_label",
    "pIC50",
]

missing = [
    x
    for x in required
    if x not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing benchmark columns: {missing}"
    )


print(
    "Benchmark molecules:",
    len(df)
)


# ============================================================
# Helpers
# ============================================================

def optimize_geometry(
    mol,
):
    """
    Optimize an already embedded H-added molecule.

    MMFF94 first.
    UFF fallback.
    Returns optimization method/status.
    """

    if AllChem.MMFFHasAllMoleculeParams(
        mol
    ):

        result = (
            AllChem.MMFFOptimizeMolecule(
                mol,
                mmffVariant="MMFF94",
                maxIters=1000,
            )
        )

        return (
            "MMFF94",
            int(result),
        )

    if AllChem.UFFHasAllMoleculeParams(
        mol
    ):

        result = (
            AllChem.UFFOptimizeMolecule(
                mol,
                maxIters=1000,
            )
        )

        return (
            "UFF",
            int(result),
        )

    return (
        "NONE",
        None,
    )


# ============================================================
# Prepare molecules
# ============================================================

records = []


for i, row in df.iterrows():

    benchmark_id = str(
        row["benchmark_id"]
    )

    smiles = str(
        row["canonical_smiles"]
    )

    sdf_path = (
        SDF_DIR
        / f"{benchmark_id}.sdf"
    )

    pdbqt_path = (
        PDBQT_DIR
        / f"{benchmark_id}.pdbqt"
    )

    log_path = (
        LOG_DIR
        / f"{benchmark_id}.log"
    )


    record = {
        "benchmark_id":
            benchmark_id,

        "benchmark_label":
            row["benchmark_label"],

        "binary_label":
            int(
                row["binary_label"]
            ),

        "pIC50":
            float(
                row["pIC50"]
            ),

        "canonical_smiles":
            smiles,

        "sdf_path":
            str(
                sdf_path
            ),

        "pdbqt_path":
            str(
                pdbqt_path
            ),

        "rdkit_parse":
            False,

        "embed_status":
            None,

        "optimization_method":
            None,

        "optimization_status":
            None,

        "formal_charge":
            None,

        "heavy_atoms":
            None,

        "total_atoms_with_H":
            None,

        "meeko_exit_code":
            None,

        "pdbqt_created":
            False,

        "status":
            "FAIL",

        "failure_reason":
            "",
    }


    # --------------------------------------------------------
    # Parse SMILES
    # --------------------------------------------------------

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        record[
            "failure_reason"
        ] = "RDKit SMILES parse failed"

        records.append(
            record
        )

        continue


    record[
        "rdkit_parse"
    ] = True

    record[
        "formal_charge"
    ] = int(
        Chem.GetFormalCharge(
            mol
        )
    )

    record[
        "heavy_atoms"
    ] = int(
        mol.GetNumHeavyAtoms()
    )


    # --------------------------------------------------------
    # Add hydrogens
    # --------------------------------------------------------

    mol_H = Chem.AddHs(
        mol
    )

    record[
        "total_atoms_with_H"
    ] = int(
        mol_H.GetNumAtoms()
    )


    # --------------------------------------------------------
    # Generate reproducible 3D conformer
    # --------------------------------------------------------

    params = (
        AllChem.ETKDGv3()
    )

    params.randomSeed = (
        BASE_SEED + i
    )

    params.useRandomCoords = False

    embed_code = (
        AllChem.EmbedMolecule(
            mol_H,
            params,
        )
    )


    # Fallback with random coordinates
    if embed_code != 0:

        params2 = (
            AllChem.ETKDGv3()
        )

        params2.randomSeed = (
            BASE_SEED + i
        )

        params2.useRandomCoords = True

        embed_code = (
            AllChem.EmbedMolecule(
                mol_H,
                params2,
            )
        )


    record[
        "embed_status"
    ] = int(
        embed_code
    )


    if embed_code != 0:

        record[
            "failure_reason"
        ] = "RDKit 3D embedding failed"

        records.append(
            record
        )

        continue


    # --------------------------------------------------------
    # Geometry optimization
    # --------------------------------------------------------

    try:

        method, opt_status = (
            optimize_geometry(
                mol_H
            )
        )

        record[
            "optimization_method"
        ] = method

        record[
            "optimization_status"
        ] = opt_status

    except Exception as exc:

        record[
            "optimization_method"
        ] = "ERROR"

        record[
            "optimization_status"
        ] = None

        # Geometry still exists.
        # Do not discard molecule solely
        # because force-field minimization failed.


    # --------------------------------------------------------
    # Molecule metadata
    # --------------------------------------------------------

    mol_H.SetProp(
        "_Name",
        benchmark_id,
    )

    mol_H.SetProp(
        "benchmark_id",
        benchmark_id,
    )

    mol_H.SetProp(
        "benchmark_label",
        str(
            row["benchmark_label"]
        ),
    )

    mol_H.SetProp(
        "binary_label",
        str(
            int(
                row["binary_label"]
            )
        ),
    )

    mol_H.SetProp(
        "experimental_pIC50",
        str(
            float(
                row["pIC50"]
            )
        ),
    )

    mol_H.SetProp(
        "input_smiles",
        smiles,
    )


    # --------------------------------------------------------
    # Write SDF
    # --------------------------------------------------------

    writer = Chem.SDWriter(
        str(
            sdf_path
        )
    )

    writer.write(
        mol_H
    )

    writer.close()


    if not sdf_path.exists():

        record[
            "failure_reason"
        ] = "SDF not written"

        records.append(
            record
        )

        continue


    # --------------------------------------------------------
    # Meeko ligand preparation
    # --------------------------------------------------------

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


    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


    log_path.write_text(
        result.stdout
    )


    record[
        "meeko_exit_code"
    ] = int(
        result.returncode
    )


    if (
        result.returncode == 0
        and pdbqt_path.exists()
        and pdbqt_path.stat().st_size > 0
    ):

        record[
            "pdbqt_created"
        ] = True

        record[
            "status"
        ] = "PASS"

    else:

        record[
            "failure_reason"
        ] = (
            "Meeko ligand preparation failed"
        )


    records.append(
        record
    )


    if (
        (i + 1) % 10 == 0
        or i == len(df) - 1
    ):

        print(
            f"Processed "
            f"{i + 1}/{len(df)}"
        )


# ============================================================
# Build manifest
# ============================================================

manifest = pd.DataFrame(
    records
)

manifest.to_csv(
    MANIFEST_OUT,
    index=False,
)


# ============================================================
# Summary
# ============================================================

n_total = len(
    manifest
)

n_pass = int(
    (
        manifest["status"]
        == "PASS"
    ).sum()
)

n_fail = (
    n_total
    - n_pass
)


active_pass = int(
    (
        (
            manifest[
                "benchmark_label"
            ]
            == "ACTIVE"
        )
        &
        (
            manifest[
                "status"
            ]
            == "PASS"
        )
    ).sum()
)


low_pass = int(
    (
        (
            manifest[
                "benchmark_label"
            ]
            == "LOW_ACTIVE"
        )
        &
        (
            manifest[
                "status"
            ]
            == "PASS"
        )
    ).sum()
)


formal_charge_counts = (
    manifest[
        "formal_charge"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


optimization_counts = (
    manifest[
        "optimization_method"
    ]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# Final QC status
# ============================================================

if n_pass == n_total:

    overall_status = "PASS"

elif n_pass >= 95:

    overall_status = (
        "REVIEW_REQUIRED"
    )

else:

    overall_status = "FAIL"


# ============================================================
# JSON report
# ============================================================

report = {
    "benchmark_size":
        int(
            n_total
        ),

    "successful_ligands":
        int(
            n_pass
        ),

    "failed_ligands":
        int(
            n_fail
        ),

    "active_success":
        int(
            active_pass
        ),

    "low_active_success":
        int(
            low_pass
        ),

    "formal_charge_counts": {
        str(k): int(v)
        for k, v
        in formal_charge_counts.items()
    },

    "optimization_method_counts": {
        str(k): int(v)
        for k, v
        in optimization_counts.items()
    },

    "manifest":
        str(
            MANIFEST_OUT
        ),

    "status":
        overall_status,
}


REPORT_OUT.write_text(
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
    "EGFR BENCHMARK LIGAND PREPARATION"
)

print(
    "========================================"
)


print(
    "\nTotal:",
    n_total
)

print(
    "PASS:",
    n_pass
)

print(
    "FAIL:",
    n_fail
)

print(
    "ACTIVE prepared:",
    active_pass,
    "/ 50"
)

print(
    "LOW_ACTIVE prepared:",
    low_pass,
    "/ 50"
)


print(
    "\nFormal charge distribution:"
)

print(
    formal_charge_counts.to_string()
)


print(
    "\nOptimization methods:"
)

print(
    optimization_counts.to_string()
)


if n_fail > 0:

    print(
        "\nFailed molecules:"
    )

    failed = manifest[
        manifest["status"]
        != "PASS"
    ]

    print(
        failed[
            [
                "benchmark_id",
                "benchmark_label",
                "pIC50",
                "embed_status",
                "meeko_exit_code",
                "failure_reason",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\nManifest:"
)

print(
    MANIFEST_OUT
)

print(
    "\nReport:"
)

print(
    REPORT_OUT
)


print(
    "\n========================================"
)

print(
    "LIGAND PREPARATION STATUS:",
    overall_status
)

print(
    "========================================"
)
