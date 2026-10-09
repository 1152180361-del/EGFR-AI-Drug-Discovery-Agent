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

ROOT = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)

RECEPTOR = (
    ROOT
    / "docking/3W32/receptor/3W32.pdbqt"
)

BOX_FILE = (
    ROOT
    / "docking/3W32/receptor/3W32.box.txt"
)

LIGAND_SUMMARY = (
    ROOT
    / "docking/3W32/phase4C/reports/"
      "phase4C_ligand_preparation.csv"
)

OUT_ROOT = (
    ROOT
    / "docking/3W32/phase4C"
)

DOCKING_DIR = (
    OUT_ROOT
    / "docking"
)

REPORT_DIR = (
    OUT_ROOT
    / "reports"
)

DOCKING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESULTS_OUT = (
    REPORT_DIR
    / "phase4C_docking_results.csv"
)

REPORT_OUT = (
    REPORT_DIR
    / "phase4C_docking_report.json"
)


# ============================================================
# Frozen docking protocol
# ============================================================

EXHAUSTIVENESS = 16
NUM_MODES = 10
ENERGY_RANGE = 5
SEED = 42
CPU = 4

TIMEOUT_SECONDS = 900


# ============================================================
# Box parser
# ============================================================

def parse_box(path):

    values = {}

    with open(path) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if "=" not in line:
                continue

            key, value = line.split(
                "=",
                1,
            )

            values[
                key.strip()
            ] = float(
                value.strip()
            )


    required = [
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
    ]


    missing = [
        x
        for x in required
        if x not in values
    ]


    if missing:

        raise RuntimeError(
            f"Missing box parameters: "
            f"{missing}"
        )


    return values


# ============================================================
# Vina output parser
#
# Parse:
# REMARK VINA RESULT: -9.123 0.000 0.000
# ============================================================

VINA_RE = re.compile(
    r"REMARK\s+VINA\s+RESULT:\s+"
    r"([-+]?\d+(?:\.\d+)?)"
)


def parse_vina_scores(
    pdbqt_path,
):

    scores = []


    if not pdbqt_path.exists():

        return scores


    if pdbqt_path.stat().st_size == 0:

        return scores


    with open(
        pdbqt_path,
        errors="replace",
    ) as f:

        for line in f:

            match = VINA_RE.search(
                line
            )

            if match:

                try:

                    scores.append(
                        float(
                            match.group(1)
                        )
                    )

                except Exception:

                    pass


    return scores


# ============================================================
# Input validation
# ============================================================

if not RECEPTOR.exists():

    raise FileNotFoundError(
        RECEPTOR
    )


if not BOX_FILE.exists():

    raise FileNotFoundError(
        BOX_FILE
    )


if not LIGAND_SUMMARY.exists():

    raise FileNotFoundError(
        LIGAND_SUMMARY
    )


box = parse_box(
    BOX_FILE
)


ligands = pd.read_csv(
    LIGAND_SUMMARY
)


valid_status = [
    "PASS",
    "PASS_EXISTING",
]


eligible = ligands[
    ligands[
        "status"
    ].isin(
        valid_status
    )
].copy()


failed_prep = ligands[
    ~ligands[
        "status"
    ].isin(
        valid_status
    )
].copy()


print(
    "\n========================================"
)

print(
    "PHASE 4D-2 3W32 BATCH DOCKING"
)

print(
    "========================================"
)


print(
    "\nLigand-preparation records:",
    len(ligands)
)

print(
    "Eligible for docking:",
    len(eligible)
)

print(
    "Preparation failures excluded:",
    len(failed_prep)
)


if len(failed_prep) > 0:

    print(
        "\nExcluded ligand-preparation failures:"
    )

    print(
        failed_prep[
            [
                "phase4C_candidate_id",
                "candidate_track",
                "zinc_id",
                "error",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\nFrozen receptor:"
)

print(
    RECEPTOR
)


print(
    "\nFrozen box:"
)

for key in [
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
]:

    print(
        f"{key}: "
        f"{box[key]:.3f}"
    )


print(
    "\nFrozen Vina parameters:"
)

print(
    "exhaustiveness:",
    EXHAUSTIVENESS
)

print(
    "num_modes:",
    NUM_MODES
)

print(
    "energy_range:",
    ENERGY_RANGE
)

print(
    "seed:",
    SEED
)

print(
    "cpu:",
    CPU
)


# ============================================================
# Docking
# ============================================================

records = []

overall_start = time.time()


for n, (_, row) in enumerate(
    eligible.iterrows(),
    start=1,
):

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

    track = str(
        row[
            "candidate_track"
        ]
    )

    ligand_path = Path(
        row[
            "pdbqt_path"
        ]
    )


    candidate_dir = (
        DOCKING_DIR
        / cid
    )

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    out_pdbqt = (
        candidate_dir
        / f"{cid}_docked.pdbqt"
    )

    log_path = (
        candidate_dir
        / f"{cid}_vina.log"
    )


    print(
        f"\n[{n:02d}/{len(eligible):02d}] "
        f"{cid} | {track} | {zinc_id}",
        flush=True,
    )


    # --------------------------------------------------------
    # Validate input ligand
    # --------------------------------------------------------

    if (
        not ligand_path.exists()
        or ligand_path.stat().st_size == 0
    ):

        print(
            "  FAIL: ligand PDBQT missing",
            flush=True,
        )


        records.append(
            {
                "phase4C_candidate_id":
                    cid,

                "candidate_track":
                    track,

                "zinc_id":
                    zinc_id,

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
                    row.get(
                        "formal_charge",
                        np.nan,
                    ),

                "status":
                    "FAIL",

                "error":
                    "ligand_PDBQT_missing",

                "n_modes":
                    0,

                "top1_vina_score":
                    np.nan,

                "best_vina_score":
                    np.nan,

                "elapsed_seconds":
                    0.0,

                "ligand_pdbqt":
                    str(
                        ligand_path
                    ),

                "docked_pdbqt":
                    str(
                        out_pdbqt
                    ),

                "vina_log":
                    str(
                        log_path
                    ),
            }
        )

        continue


    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    existing_scores = (
        parse_vina_scores(
            out_pdbqt
        )
    )


    if len(existing_scores) > 0:

        print(
            f"  REUSE | "
            f"modes={len(existing_scores)} | "
            f"Top1={existing_scores[0]:.3f}",
            flush=True,
        )


        records.append(
            {
                "phase4C_candidate_id":
                    cid,

                "candidate_track":
                    track,

                "zinc_id":
                    zinc_id,

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
                    row.get(
                        "formal_charge",
                        np.nan,
                    ),

                "status":
                    "PASS_EXISTING",

                "error":
                    "",

                "n_modes":
                    len(
                        existing_scores
                    ),

                "top1_vina_score":
                    existing_scores[0],

                "best_vina_score":
                    min(
                        existing_scores
                    ),

                "elapsed_seconds":
                    0.0,

                "ligand_pdbqt":
                    str(
                        ligand_path
                    ),

                "docked_pdbqt":
                    str(
                        out_pdbqt
                    ),

                "vina_log":
                    str(
                        log_path
                    ),
            }
        )

        continue


    # --------------------------------------------------------
    # Remove invalid partial output before rerun
    # --------------------------------------------------------

    if out_pdbqt.exists():

        try:

            out_pdbqt.unlink()

        except Exception:

            pass


    # --------------------------------------------------------
    # Frozen Vina command
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

        "--center_x",
        str(
            box[
                "center_x"
            ]
        ),

        "--center_y",
        str(
            box[
                "center_y"
            ]
        ),

        "--center_z",
        str(
            box[
                "center_z"
            ]
        ),

        "--size_x",
        str(
            box[
                "size_x"
            ]
        ),

        "--size_y",
        str(
            box[
                "size_y"
            ]
        ),

        "--size_z",
        str(
            box[
                "size_z"
            ]
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
            out_pdbqt
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


        elapsed = (
            time.time()
            - start
        )


        log_path.write_text(
            result.stdout
            if result.stdout
            else ""
        )


        scores = (
            parse_vina_scores(
                out_pdbqt
            )
        )


        if (
            result.returncode == 0
            and len(scores) > 0
        ):

            status = "PASS"

            error = ""


            print(
                f"  PASS | "
                f"modes={len(scores)} | "
                f"Top1={scores[0]:.3f} | "
                f"{elapsed:.1f}s",
                flush=True,
            )


        else:

            status = "FAIL"

            error = (
                "Vina_failed_or_no_scores;"
                f" exit={result.returncode}"
            )


            print(
                "  FAIL:",
                error,
                flush=True,
            )


    except subprocess.TimeoutExpired as exc:

        elapsed = (
            time.time()
            - start
        )


        log_text = ""

        if exc.stdout:

            if isinstance(
                exc.stdout,
                bytes,
            ):

                log_text += (
                    exc.stdout.decode(
                        errors="replace"
                    )
                )

            else:

                log_text += str(
                    exc.stdout
                )


        log_text += (
            "\nVINA TIMEOUT\n"
        )


        log_path.write_text(
            log_text
        )


        scores = (
            parse_vina_scores(
                out_pdbqt
            )
        )


        status = "FAIL"

        error = (
            "Vina_timeout"
        )


        print(
            f"  FAIL: timeout after "
            f"{elapsed:.1f}s",
            flush=True,
        )


    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    records.append(
        {
            "phase4C_candidate_id":
                cid,

            "candidate_track":
                track,

            "zinc_id":
                zinc_id,

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
                row.get(
                    "formal_charge",
                    np.nan,
                ),

            "status":
                status,

            "error":
                error,

            "n_modes":
                len(
                    scores
                ),

            "top1_vina_score":
                (
                    scores[0]
                    if len(scores) > 0
                    else np.nan
                ),

            "best_vina_score":
                (
                    min(
                        scores
                    )
                    if len(scores) > 0
                    else np.nan
                ),

            "elapsed_seconds":
                elapsed,

            "ligand_pdbqt":
                str(
                    ligand_path
                ),

            "docked_pdbqt":
                str(
                    out_pdbqt
                ),

            "vina_log":
                str(
                    log_path
                ),
        }
    )


    # --------------------------------------------------------
    # Incremental save
    #
    # Important for unexpected shutdowns.
    # --------------------------------------------------------

    partial = pd.DataFrame(
        records
    )

    partial.to_csv(
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


success = results[
    success_mask
].copy()


fail = results[
    ~success_mask
].copy()


# ============================================================
# Track summary
# ============================================================

track_summary = {}


for track in results[
    "candidate_track"
].unique():

    sub = results[
        results[
            "candidate_track"
        ] == track
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


# ============================================================
# Score summaries
#
# Score is stored for structural comparison only.
# It is NOT treated as predicted binding free energy
# or potency.
# ============================================================

score_summary = {}


for track in success[
    "candidate_track"
].unique():

    values = success.loc[
        success[
            "candidate_track"
        ]
        == track,
        "top1_vina_score",
    ].astype(float)


    score_summary[
        str(track)
    ] = {
        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                values.mean()
            ),

        "median":
            float(
                values.median()
            ),

        "min":
            float(
                values.min()
            ),

        "max":
            float(
                values.max()
            ),
    }


# ============================================================
# Ranking table
#
# This is diagnostic only.
# It is NOT a potency ranking.
# ============================================================

diagnostic_rank = (
    success
    .sort_values(
        "top1_vina_score",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)


diagnostic_rank[
    "vina_diagnostic_rank"
] = np.arange(
    1,
    len(
        diagnostic_rank
    ) + 1,
)


# ============================================================
# Report
# ============================================================

total_elapsed = (
    time.time()
    - overall_start
)


report = {
    "production_receptor":
        str(
            RECEPTOR
        ),

    "box": {
        k:
            float(v)
        for k, v
        in box.items()
    },

    "protocol": {
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

    "input_prepared_ligands":
        int(
            len(
                eligible
            )
        ),

    "excluded_ligand_preparation_failures":
        int(
            len(
                failed_prep
            )
        ),

    "successful_dockings":
        int(
            len(
                success
            )
        ),

    "failed_dockings":
        int(
            len(
                fail
            )
        ),

    "track_summary":
        track_summary,

    "top1_score_summary":
        score_summary,

    "scientific_interpretation":
        (
            "Vina score is retained as a docking "
            "diagnostic only. Previous retrospective "
            "benchmarking showed that the frozen "
            "3W32/Vina score did not reliably "
            "discriminate EGFR biochemical activity. "
            "Therefore these scores are not interpreted "
            "as binding free energies or potency "
            "predictions."
        ),

    "elapsed_minutes":
        float(
            total_elapsed
            / 60.0
        ),

    "results_csv":
        str(
            RESULTS_OUT
        ),
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
    "PHASE 4D-2 DOCKING SUMMARY"
)

print(
    "========================================"
)


print(
    "\nPrepared ligands:",
    len(
        eligible
    )
)

print(
    "Docking PASS:",
    len(
        success
    )
)

print(
    "Docking FAIL:",
    len(
        fail
    )
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
    "\nTop-1 Vina score summary "
    "(diagnostic only):"
)


for track, values in (
    score_summary.items()
):

    print(
        f"{track}: "
        f"n={values['n']} | "
        f"mean={values['mean']:.3f} | "
        f"median={values['median']:.3f} | "
        f"range="
        f"{values['min']:.3f} to "
        f"{values['max']:.3f}"
    )


print(
    "\n----------------------------------------"
)

print(
    "TOP 15 BY VINA SCORE"
)

print(
    "DIAGNOSTIC ONLY - NOT POTENCY RANKING"
)

print(
    "----------------------------------------"
)


display_cols = [
    "vina_diagnostic_rank",
    "phase4C_candidate_id",
    "candidate_track",
    "zinc_id",
    "qsar_predicted_pIC50",
    "max_training_tanimoto",
    "rf_tree_sd_pIC50",
    "top1_vina_score",
]


print(
    diagnostic_rank[
        display_cols
    ]
    .head(15)
    .to_string(
        index=False
    )
)


if len(fail) > 0:

    print(
        "\nFAILED DOCKINGS:"
    )


    print(
        fail[
            [
                "phase4C_candidate_id",
                "zinc_id",
                "error",
            ]
        ]
        .to_string(
            index=False
        )
    )


print(
    "\nElapsed:",
    f"{total_elapsed/60:.1f} min"
)


print(
    "\nSaved:"
)

print(
    RESULTS_OUT
)

print(
    REPORT_OUT
)


print(
    "\n========================================"
)

if len(fail) == 0:

    print(
        "PHASE 4D-2 STATUS: PASS"
    )

else:

    print(
        "PHASE 4D-2 STATUS: "
        "PASS_WITH_FAILURES"
    )

print(
    "========================================"
)
