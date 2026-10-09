from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)

RAW = (
    ROOT
    / "data/raw/external/zinc15_250k/"
      "zinc15_250K_2D.csv"
)

TRAINING = (
    ROOT
    / "data/processed/egfr_qsar_modeling.csv"
)

WORKER = (
    ROOT
    / "scripts/35_worker_standardize_zinc15.py"
)

CHUNK_DIR = (
    ROOT
    / "data/interim/zinc15_250k_chunks"
)

PROCESSED_DIR = (
    ROOT
    / "data/processed/external"
)

CHUNK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


STANDARDIZED_OUT = (
    PROCESSED_DIR
    / "zinc15_250k_standardized.csv"
)

SCREENING_OUT = (
    PROCESSED_DIR
    / "zinc15_250k_screening_library.csv"
)

FAILURES_OUT = (
    PROCESSED_DIR
    / "zinc15_250k_standardization_failures.csv"
)

OVERLAP_OUT = (
    PROCESSED_DIR
    / "zinc15_250k_training_overlaps.csv"
)

REPORT_OUT = (
    PROCESSED_DIR
    / "zinc15_250k_standardization_report.json"
)


# ============================================================
# Settings
# ============================================================

CHUNK_SIZE = 2000

MIN_SPLIT_SIZE = 1


# ============================================================
# Load raw library
# ============================================================

raw = pd.read_csv(
    RAW
)


print(
    "\n========================================"
)

print(
    "ZINC15 250K STANDARDIZATION"
)

print(
    "========================================"
)


print(
    "\nRaw rows:",
    len(raw)
)


required = [
    "smiles",
    "zinc_id",
]


missing = [
    x
    for x in required
    if x not in raw.columns
]


if missing:

    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# Stable original row ID
raw = raw.copy()

raw[
    "_source_row"
] = np.arange(
    len(raw)
)


# ============================================================
# ZINC metadata QC / eligibility
# ============================================================

if "reactive" in raw.columns:

    reactive_counts = (
        raw[
            "reactive"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

else:

    reactive_counts = None


if "purchasable" in raw.columns:

    purchasable_counts = (
        raw[
            "purchasable"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

else:

    purchasable_counts = None


eligible = raw.copy()


# Exclude explicitly reactive entries
if "reactive" in eligible.columns:

    eligible = eligible[
        eligible[
            "reactive"
        ] == 0
    ].copy()


# Require purchasability when the field exists.
# ZINC uses a nonzero value for purchasable entries.
if "purchasable" in eligible.columns:

    eligible = eligible[
        pd.to_numeric(
            eligible[
                "purchasable"
            ],
            errors="coerce",
        ).fillna(0) > 0
    ].copy()


eligible = eligible.reset_index(
    drop=True
)


print(
    "Eligible after "
    "reactive/purchasable QC:",
    len(eligible)
)


# ============================================================
# Robust subprocess standardization
# ============================================================

def output_is_valid(
    path,
    expected_rows,
):

    if not path.exists():
        return False

    try:

        df = pd.read_csv(
            path
        )

        return (
            len(df)
            == expected_rows
            and
            "standardization_status"
            in df.columns
        )

    except Exception:

        return False


def process_block(
    block,
    depth=0,
):

    """
    Run RDKit in a child process.

    If the process crashes, recursively split
    the block until the offending molecule is
    isolated.

    Existing valid block outputs are reused,
    making the workflow restart-safe.
    """

    n = len(
        block
    )

    if n == 0:

        return pd.DataFrame()


    first_row = int(
        block[
            "_source_row"
        ].min()
    )

    last_row = int(
        block[
            "_source_row"
        ].max()
    )


    tag = (
        f"{first_row:06d}_"
        f"{last_row:06d}_"
        f"n{n}"
    )


    input_path = (
        CHUNK_DIR
        / f"{tag}_input.csv"
    )

    output_path = (
        CHUNK_DIR
        / f"{tag}_output.csv"
    )

    log_path = (
        CHUNK_DIR
        / f"{tag}.log"
    )


    # ----------------------------------------
    # Resume support
    # ----------------------------------------

    if output_is_valid(
        output_path,
        n,
    ):

        print(
            f"REUSE {tag}",
            flush=True,
        )

        return pd.read_csv(
            output_path
        )


    block.to_csv(
        input_path,
        index=False,
    )


    cmd = [
        sys.executable,
        str(
            WORKER
        ),
        str(
            input_path
        ),
        str(
            output_path
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
        if result.stdout
        else ""
    )


    if (
        result.returncode == 0
        and output_is_valid(
            output_path,
            n,
        )
    ):

        print(
            f"PASS  {tag}",
            flush=True,
        )

        return pd.read_csv(
            output_path
        )


    # ----------------------------------------
    # Child process failed
    # ----------------------------------------

    print(
        f"CHILD FAILURE: {tag} "
        f"exit={result.returncode}",
        flush=True,
    )


    if n <= MIN_SPLIT_SIZE:

        # Preserve the offending molecule as
        # an explicit failed record.
        failed = block.copy()

        failed[
            "standardization_status"
        ] = "FAIL"

        failed[
            "standardization_error"
        ] = (
            "worker_process_failure_"
            f"exit_{result.returncode}"
        )

        failed[
            "standardized_smiles"
        ] = ""

        failed[
            "standardized_inchikey"
        ] = ""

        failed[
            "standardization_changed"
        ] = False

        failed.to_csv(
            output_path,
            index=False,
        )

        return failed


    # ----------------------------------------
    # Recursive bisection
    # ----------------------------------------

    midpoint = (
        n // 2
    )

    left = block.iloc[
        :midpoint
    ].copy()

    right = block.iloc[
        midpoint:
    ].copy()


    left_result = (
        process_block(
            left,
            depth + 1,
        )
    )

    right_result = (
        process_block(
            right,
            depth + 1,
        )
    )


    combined = pd.concat(
        [
            left_result,
            right_result,
        ],
        ignore_index=True,
    )


    combined.to_csv(
        output_path,
        index=False,
    )


    return combined


# ============================================================
# Top-level chunks
# ============================================================

all_results = []

n_total = len(
    eligible
)


for start in range(
    0,
    n_total,
    CHUNK_SIZE,
):

    end = min(
        start + CHUNK_SIZE,
        n_total,
    )


    print(
        f"\nTOP-LEVEL BLOCK "
        f"{start + 1}-{end} / "
        f"{n_total}",
        flush=True,
    )


    block = (
        eligible.iloc[
            start:end
        ]
        .copy()
    )


    result = process_block(
        block
    )


    all_results.append(
        result
    )


# ============================================================
# Merge
# ============================================================

standardized = pd.concat(
    all_results,
    ignore_index=True,
)


standardized = (
    standardized
    .sort_values(
        "_source_row"
    )
    .reset_index(
        drop=True
    )
)


standardized.to_csv(
    STANDARDIZED_OUT,
    index=False,
)


# ============================================================
# Standardization QC
# ============================================================

pass_df = standardized[
    standardized[
        "standardization_status"
    ] == "PASS"
].copy()


fail_df = standardized[
    standardized[
        "standardization_status"
    ] != "PASS"
].copy()


fail_df.to_csv(
    FAILURES_OUT,
    index=False,
)


n_changed = int(
    pass_df[
        "standardization_changed"
    ]
    .astype(bool)
    .sum()
)


# ============================================================
# Deduplicate standardized structures
# ============================================================

duplicate_counts = (
    pass_df
    .groupby(
        "standardized_smiles"
    )
    .size()
    .rename(
        "n_zinc_ids_same_standardized_structure"
    )
)


pass_df = pass_df.merge(
    duplicate_counts,
    left_on="standardized_smiles",
    right_index=True,
    how="left",
)


before_dedup = len(
    pass_df
)


dedup = (
    pass_df
    .sort_values(
        [
            "_source_row",
        ]
    )
    .drop_duplicates(
        subset=[
            "standardized_smiles"
        ],
        keep="first",
    )
    .copy()
)


after_dedup = len(
    dedup
)


# ============================================================
# Load the 4,089-compound EGFR training set
# ============================================================

training = pd.read_csv(
    TRAINING
)


train_smiles = set(
    training[
        "standardized_smiles"
    ]
    .dropna()
    .astype(str)
)


if (
    "standardized_inchikey"
    in training.columns
):

    train_inchikey = set(
        training[
            "standardized_inchikey"
        ]
        .dropna()
        .astype(str)
    )

else:

    train_inchikey = set()


# ============================================================
# Exact training-set leakage flags
# ============================================================

dedup[
    "exact_smiles_overlap_training"
] = (
    dedup[
        "standardized_smiles"
    ]
    .isin(
        train_smiles
    )
)


if train_inchikey:

    dedup[
        "exact_inchikey_overlap_training"
    ] = (
        dedup[
            "standardized_inchikey"
        ]
        .isin(
            train_inchikey
        )
    )

else:

    dedup[
        "exact_inchikey_overlap_training"
    ] = False


dedup[
    "exact_training_overlap"
] = (
    dedup[
        "exact_smiles_overlap_training"
    ]
    |
    dedup[
        "exact_inchikey_overlap_training"
    ]
)


overlap = dedup[
    dedup[
        "exact_training_overlap"
    ]
].copy()


overlap.to_csv(
    OVERLAP_OUT,
    index=False,
)


screening = dedup[
    ~dedup[
        "exact_training_overlap"
    ]
].copy()


screening = (
    screening
    .reset_index(
        drop=True
    )
)


screening[
    "screening_library_id"
] = [
    f"ZINC15_SCREEN_{i+1:06d}"
    for i in range(
        len(screening)
    )
]


screening.to_csv(
    SCREENING_OUT,
    index=False,
)


# ============================================================
# Report
# ============================================================

report = {
    "raw_rows":
        int(
            len(raw)
        ),

    "eligible_after_metadata_QC":
        int(
            len(eligible)
        ),

    "standardization_PASS":
        int(
            len(pass_df)
        ),

    "standardization_FAIL":
        int(
            len(fail_df)
        ),

    "standardization_changed":
        int(
            n_changed
        ),

    "unique_standardized_structures":
        int(
            after_dedup
        ),

    "duplicates_removed_after_standardization":
        int(
            before_dedup
            - after_dedup
        ),

    "training_dataset_size":
        int(
            len(training)
        ),

    "exact_training_overlaps_removed":
        int(
            len(overlap)
        ),

    "final_screening_library_size":
        int(
            len(screening)
        ),

    "reactive_distribution":
        (
            None
            if reactive_counts is None
            else {
                str(k): int(v)
                for k, v
                in reactive_counts.items()
            }
        ),

    "purchasable_distribution":
        (
            None
            if purchasable_counts is None
            else {
                str(k): int(v)
                for k, v
                in purchasable_counts.items()
            }
        ),

    "standardization_protocol": [
        "RDKit Cleanup",
        "FragmentParent",
        "Uncharger",
        "SanitizeMol",
        "require carbon",
        "canonical isomeric SMILES",
        "InChIKey",
    ],

    "tautomer_canonicalization":
        False,

    "outputs": {
        "standardized_all":
            str(
                STANDARDIZED_OUT
            ),

        "screening_library":
            str(
                SCREENING_OUT
            ),

        "failures":
            str(
                FAILURES_OUT
            ),

        "training_overlaps":
            str(
                OVERLAP_OUT
            ),
    },
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
    "ZINC15 SCREENING LIBRARY QC"
)

print(
    "========================================"
)


print(
    "\nRaw ZINC15 rows:",
    len(raw)
)

print(
    "Eligible after metadata QC:",
    len(eligible)
)


if reactive_counts is not None:

    print(
        "\nReactive distribution:"
    )

    print(
        reactive_counts.to_string()
    )


if purchasable_counts is not None:

    print(
        "\nPurchasable distribution:"
    )

    print(
        purchasable_counts.to_string()
    )


print(
    "\nStandardization PASS:",
    len(pass_df)
)

print(
    "Standardization FAIL:",
    len(fail_df)
)

print(
    "Structures changed by standardization:",
    n_changed
)


print(
    "\nUnique standardized structures:",
    after_dedup
)

print(
    "Duplicates removed:",
    before_dedup
    - after_dedup
)


print(
    "\nEGFR training compounds:",
    len(training)
)

print(
    "Exact training overlaps removed:",
    len(overlap)
)


print(
    "\nFINAL EXTERNAL SCREENING LIBRARY:",
    len(screening)
)


print(
    "\nSaved:"
)

print(
    SCREENING_OUT
)

print(
    REPORT_OUT
)


print(
    "\n========================================"
)

if len(fail_df) == 0:

    print(
        "PHASE 4A STANDARDIZATION: PASS"
    )

else:

    print(
        "PHASE 4A STANDARDIZATION: "
        "PASS_WITH_RECORDED_FAILURES"
    )

print(
    "========================================"
)
