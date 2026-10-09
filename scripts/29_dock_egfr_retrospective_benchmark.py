from pathlib import Path
import json
import re
import subprocess
import time

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

BASE = ROOT / "docking/3W32/benchmark"

BENCHMARK_CSV = (
    BASE
    / "egfr_retrospective_benchmark.csv"
)

PREP_MANIFEST = (
    BASE
    / "benchmark_ligand_preparation_manifest.csv"
)

RECEPTOR = (
    ROOT
    / "docking/3W32/receptor/3W32.pdbqt"
)

BOX_CONFIG = (
    ROOT
    / "docking/3W32/receptor/3W32.box.txt"
)

DOCK_DIR = (
    BASE
    / "docked_pdbqt"
)

LOG_DIR = (
    BASE
    / "docking_logs"
)

RESULTS_OUT = (
    BASE
    / "benchmark_docking_results.csv"
)

REPORT_OUT = (
    BASE
    / "benchmark_docking_report.json"
)


DOCK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Docking settings
# ============================================================

EXHAUSTIVENESS = 16
NUM_MODES = 10
ENERGY_RANGE = 5
SEED = 42

# Use a controlled CPU count per ligand.
CPU = 4

# One ligand should not monopolize the workflow.
TIMEOUT_SECONDS = 900


# ============================================================
# Basic file checks
# ============================================================

required_files = [
    BENCHMARK_CSV,
    PREP_MANIFEST,
    RECEPTOR,
    BOX_CONFIG,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file missing: {path}"
        )


# ============================================================
# Load benchmark and ligand-preparation manifest
# ============================================================

benchmark = pd.read_csv(
    BENCHMARK_CSV
)

manifest = pd.read_csv(
    PREP_MANIFEST
)


if len(benchmark) != 100:

    print(
        f"WARNING: benchmark contains "
        f"{len(benchmark)} molecules, not 100."
    )


# Keep only successfully prepared ligands
manifest_pass = (
    manifest[
        manifest["status"]
        == "PASS"
    ]
    .copy()
)


merged = benchmark.merge(
    manifest_pass[
        [
            "benchmark_id",
            "pdbqt_path",
        ]
    ],
    on="benchmark_id",
    how="inner",
)


print(
    "Benchmark rows:",
    len(benchmark)
)

print(
    "Prepared ligand rows:",
    len(manifest_pass)
)

print(
    "Docking-ready rows:",
    len(merged)
)


if len(merged) != len(benchmark):

    missing_ids = sorted(
        set(
            benchmark[
                "benchmark_id"
            ]
        )
        -
        set(
            merged[
                "benchmark_id"
            ]
        )
    )

    raise RuntimeError(
        "Some benchmark ligands are not "
        "prepared successfully:\n"
        f"{missing_ids}"
    )


# ============================================================
# Helpers
# ============================================================

vina_result_pattern = re.compile(
    r"REMARK VINA RESULT:\s*"
    r"([-+]?\d+(?:\.\d+)?)"
)


def parse_vina_scores(
    pdbqt_path,
):

    if not pdbqt_path.exists():

        return []

    scores = []

    with open(
        pdbqt_path,
        errors="replace",
    ) as f:

        for line in f:

            match = (
                vina_result_pattern
                .search(
                    line
                )
            )

            if match:

                scores.append(
                    float(
                        match.group(1)
                    )
                )

    return scores


def remove_partial_output(
    path,
):

    try:

        if path.exists():
            path.unlink()

    except Exception:
        pass


# ============================================================
# Docking
# ============================================================

records = []

total = len(
    merged
)


for position, row in (
    merged.iterrows()
):

    benchmark_id = str(
        row[
            "benchmark_id"
        ]
    )

    ligand_path = Path(
        row[
            "pdbqt_path"
        ]
    )

    output_path = (
        DOCK_DIR
        / f"{benchmark_id}_docked.pdbqt"
    )

    log_path = (
        LOG_DIR
        / f"{benchmark_id}.log"
    )


    print(
        f"\n[{position + 1:03d}/{total:03d}] "
        f"{benchmark_id}  "
        f"{row['benchmark_label']}  "
        f"pIC50={float(row['pIC50']):.3f}",
        flush=True,
    )


    record = {
        "benchmark_id":
            benchmark_id,

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

        "ligand_pdbqt":
            str(
                ligand_path
            ),

        "docked_pdbqt":
            str(
                output_path
            ),

        "top1_vina_score":
            np.nan,

        "best_vina_score":
            np.nan,

        "n_modes":
            0,

        "runtime_seconds":
            np.nan,

        "exit_code":
            None,

        "status":
            "FAIL",

        "message":
            "",
    }


    # --------------------------------------------------------
    # Ligand check
    # --------------------------------------------------------

    if (
        not ligand_path.exists()
        or ligand_path.stat().st_size == 0
    ):

        record[
            "message"
        ] = "Ligand PDBQT missing"

        records.append(
            record
        )

        print(
            "  FAIL: ligand PDBQT missing",
            flush=True,
        )

        continue


    # --------------------------------------------------------
    # Resume support
    #
    # If valid output already exists,
    # do not redock it.
    # --------------------------------------------------------

    existing_scores = (
        parse_vina_scores(
            output_path
        )
    )


    if existing_scores:

        record[
            "top1_vina_score"
        ] = existing_scores[0]

        record[
            "best_vina_score"
        ] = min(
            existing_scores
        )

        record[
            "n_modes"
        ] = len(
            existing_scores
        )

        record[
            "exit_code"
        ] = 0

        record[
            "status"
        ] = "PASS_EXISTING"

        record[
            "message"
        ] = "Valid existing docking output reused"

        records.append(
            record
        )

        print(
            f"  REUSED: "
            f"Top1={existing_scores[0]:.3f} "
            f"kcal/mol, "
            f"modes={len(existing_scores)}",
            flush=True,
        )

        continue


    # Remove incomplete previous output
    remove_partial_output(
        output_path
    )


    # --------------------------------------------------------
    # Vina command
    # --------------------------------------------------------

    cmd = [
        "vina",

        "--receptor",
        str(
            RECEPTOR
        ),

        "--ligand",
        str(
            ligand_path
        ),

        "--config",
        str(
            BOX_CONFIG
        ),

        "--exhaustiveness",
        str(
            EXHAUSTIVENESS
        ),

        "--num_modes",
        str(
            NUM_MODES
        ),

        "--energy_range",
        str(
            ENERGY_RANGE
        ),

        "--seed",
        str(
            SEED
        ),

        "--cpu",
        str(
            CPU
        ),

        "--out",
        str(
            output_path
        ),
    ]


    start = time.time()


    try:

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        runtime = (
            time.time()
            - start
        )


        log_path.write_text(
            result.stdout
            if result.stdout
            else ""
        )


        record[
            "runtime_seconds"
        ] = round(
            runtime,
            3,
        )

        record[
            "exit_code"
        ] = int(
            result.returncode
        )


        scores = (
            parse_vina_scores(
                output_path
            )
        )


        if (
            result.returncode == 0
            and scores
        ):

            record[
                "top1_vina_score"
            ] = scores[0]

            record[
                "best_vina_score"
            ] = min(
                scores
            )

            record[
                "n_modes"
            ] = len(
                scores
            )

            record[
                "status"
            ] = "PASS"

            record[
                "message"
            ] = "Docking completed"

            print(
                f"  PASS: "
                f"Top1={scores[0]:.3f} "
                f"kcal/mol | "
                f"modes={len(scores)} | "
                f"{runtime:.1f}s",
                flush=True,
            )


        else:

            record[
                "message"
            ] = (
                "Vina returned no valid "
                "docking poses"
            )

            remove_partial_output(
                output_path
            )

            print(
                f"  FAIL: "
                f"exit={result.returncode}",
                flush=True,
            )


    except subprocess.TimeoutExpired as exc:

        runtime = (
            time.time()
            - start
        )


        output = (
            exc.stdout
            if exc.stdout
            else ""
        )


        if isinstance(
            output,
            bytes
        ):

            output = output.decode(
                errors="replace"
            )


        log_path.write_text(
            str(
                output
            )
            + "\n\n"
            + f"TIMEOUT AFTER "
            + f"{TIMEOUT_SECONDS} SECONDS\n"
        )


        remove_partial_output(
            output_path
        )


        record[
            "runtime_seconds"
        ] = round(
            runtime,
            3,
        )

        record[
            "exit_code"
        ] = -999

        record[
            "status"
        ] = "TIMEOUT"

        record[
            "message"
        ] = (
            f"Docking timeout "
            f">{TIMEOUT_SECONDS}s"
        )


        print(
            f"  TIMEOUT after "
            f"{TIMEOUT_SECONDS}s",
            flush=True,
        )


    except Exception as exc:

        runtime = (
            time.time()
            - start
        )


        record[
            "runtime_seconds"
        ] = round(
            runtime,
            3,
        )

        record[
            "status"
        ] = "ERROR"

        record[
            "message"
        ] = str(
            exc
        )


        print(
            "  ERROR:",
            exc,
            flush=True,
        )


    records.append(
        record
    )


    # --------------------------------------------------------
    # Incremental checkpoint
    #
    # Save after every molecule.
    # --------------------------------------------------------

    pd.DataFrame(
        records
    ).to_csv(
        RESULTS_OUT,
        index=False,
    )


# ============================================================
# Final results
# ============================================================

results = pd.DataFrame(
    records
)


results.to_csv(
    RESULTS_OUT,
    index=False,
)


success_mask = (
    results[
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

n_fail = (
    len(results)
    - n_success
)


active = results[
    results[
        "benchmark_label"
    ]
    == "ACTIVE"
]


low = results[
    results[
        "benchmark_label"
    ]
    == "LOW_ACTIVE"
]


active_success = int(
    active[
        "status"
    ].isin(
        [
            "PASS",
            "PASS_EXISTING",
        ]
    ).sum()
)


low_success = int(
    low[
        "status"
    ].isin(
        [
            "PASS",
            "PASS_EXISTING",
        ]
    ).sum()
)


successful = results[
    success_mask
].copy()


if len(successful) > 0:

    mean_runtime = float(
        successful[
            "runtime_seconds"
        ]
        .dropna()
        .mean()
    )

    active_mean_score = float(
        successful[
            successful[
                "benchmark_label"
            ]
            == "ACTIVE"
        ][
            "top1_vina_score"
        ].mean()
    )

    low_mean_score = float(
        successful[
            successful[
                "benchmark_label"
            ]
            == "LOW_ACTIVE"
        ][
            "top1_vina_score"
        ].mean()
    )

else:

    mean_runtime = np.nan
    active_mean_score = np.nan
    low_mean_score = np.nan


# ============================================================
# Overall status
# ============================================================

if n_success == len(
    results
):

    overall_status = "PASS"

elif n_success >= 95:

    overall_status = (
        "REVIEW_REQUIRED"
    )

else:

    overall_status = "FAIL"


# ============================================================
# JSON report
# ============================================================

report = {
    "receptor":
        str(
            RECEPTOR
        ),

    "box_config":
        str(
            BOX_CONFIG
        ),

    "n_ligands":
        int(
            len(results)
        ),

    "successful_dockings":
        int(
            n_success
        ),

    "failed_dockings":
        int(
            n_fail
        ),

    "active_success":
        int(
            active_success
        ),

    "low_active_success":
        int(
            low_success
        ),

    "settings": {
        "exhaustiveness":
            EXHAUSTIVENESS,

        "num_modes":
            NUM_MODES,

        "energy_range":
            ENERGY_RANGE,

        "seed":
            SEED,

        "cpu":
            CPU,

        "timeout_seconds":
            TIMEOUT_SECONDS,
    },

    "mean_runtime_seconds":
        (
            None
            if np.isnan(
                mean_runtime
            )
            else mean_runtime
        ),

    "active_mean_top1_score":
        (
            None
            if np.isnan(
                active_mean_score
            )
            else active_mean_score
        ),

    "low_active_mean_top1_score":
        (
            None
            if np.isnan(
                low_mean_score
            )
            else low_mean_score
        ),

    "results_csv":
        str(
            RESULTS_OUT
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
# Terminal summary
# ============================================================

print(
    "\n========================================"
)

print(
    "EGFR RETROSPECTIVE BENCHMARK DOCKING"
)

print(
    "========================================"
)


print(
    "\nTotal ligands:",
    len(
        results
    )
)

print(
    "Successful:",
    n_success
)

print(
    "Failed:",
    n_fail
)

print(
    "ACTIVE:",
    active_success,
    "/",
    len(
        active
    )
)

print(
    "LOW_ACTIVE:",
    low_success,
    "/",
    len(
        low
    )
)


print(
    "\nDocking settings:"
)

print(
    "Exhaustiveness:",
    EXHAUSTIVENESS
)

print(
    "Num modes:",
    NUM_MODES
)

print(
    "Energy range:",
    ENERGY_RANGE
)

print(
    "Seed:",
    SEED
)

print(
    "CPU:",
    CPU
)


if len(
    successful
) > 0:

    print(
        "\nMean Top-1 Vina score:"
    )

    print(
        "ACTIVE:",
        round(
            active_mean_score,
            3,
        ),
        "kcal/mol"
    )

    print(
        "LOW_ACTIVE:",
        round(
            low_mean_score,
            3,
        ),
        "kcal/mol"
    )


if n_fail > 0:

    print(
        "\nUnsuccessful ligands:"
    )

    print(
        results[
            ~success_mask
        ][
            [
                "benchmark_id",
                "benchmark_label",
                "pIC50",
                "status",
                "exit_code",
                "message",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\nResults:"
)

print(
    RESULTS_OUT
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
    "DOCKING STATUS:",
    overall_status
)

print(
    "========================================"
)
