from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================
# Project paths
# =========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_qsar_scaffolds.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
)

FIGURE_DIR = (
    ROOT
    / "figures"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DETAIL_FILE = (
    REPORT_DIR
    / "phase2_repeated_scaffold_metrics.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "phase2_repeated_scaffold_summary.csv"
)

FIGURE_FILE = (
    FIGURE_DIR
    / "phase2_repeated_scaffold_r2.png"
)


# =========================================
# Settings
# =========================================

SEEDS = [
    11,
    22,
    33,
    42,
    55,
]

TRAIN_FRACTION = 0.80
VALID_FRACTION = 0.10

FP_RADIUS = 2
FP_SIZE = 2048


# =========================================
# Model configurations
# =========================================

MODEL_CONFIGS = {

    "RF1": {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },

    "RF4": {
        "n_estimators": 300,
        "max_features": 0.3,
        "min_samples_leaf": 2,
    },
}


# =========================================
# Load dataset
# =========================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    "Input compounds:",
    len(df)
)

print(
    "Unique scaffolds:",
    df[
        "split_scaffold_id"
    ].nunique()
)


# =========================================
# Morgan generator
# =========================================

generator = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_SIZE
    )
)


# =========================================
# Generate fingerprints ONCE
# =========================================

print(
    "\nGenerating Morgan fingerprints..."
)

fingerprints = []


for i, smiles in enumerate(
    df[
        "standardized_smiles"
    ].tolist(),
    start=1
):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:

        raise RuntimeError(
            f"RDKit failed for compound {i}: "
            f"{smiles}"
        )

    fp = generator.GetFingerprint(
        mol
    )

    array = np.zeros(
        FP_SIZE,
        dtype=np.uint8
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        array
    )

    fingerprints.append(
        array
    )

    if i % 500 == 0:

        print(
            f"Processed "
            f"{i}/{len(df)}"
        )


X = np.asarray(
    fingerprints,
    dtype=np.uint8
)

y = df[
    "pIC50"
].to_numpy(
    dtype=np.float32
)


print(
    "Feature matrix:",
    X.shape
)


# =========================================
# Build scaffold groups
#
# No pandas groupby.
# =========================================

scaffold_groups = {}

for index, scaffold_id in enumerate(
    df[
        "split_scaffold_id"
    ].astype(str)
):

    scaffold_groups.setdefault(
        scaffold_id,
        []
    ).append(
        index
    )


# =========================================
# Scaffold split function
# =========================================

def make_scaffold_split(seed):

    scaffold_items = list(
        scaffold_groups.items()
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        scaffold_items
    )

    scaffold_items.sort(
        key=lambda item: len(
            item[1]
        ),
        reverse=True
    )


    n_total = len(df)

    target_train = round(
        n_total
        * TRAIN_FRACTION
    )

    target_valid = round(
        n_total
        * VALID_FRACTION
    )

    target_test = (
        n_total
        - target_train
        - target_valid
    )


    targets = {
        "train":
            target_train,

        "validation":
            target_valid,

        "test":
            target_test,
    }


    assignments = {
        "train": [],
        "validation": [],
        "test": [],
    }


    # =====================================
    # Balanced greedy allocation
    # =====================================

    for scaffold_id, indices in scaffold_items:

        group_size = len(
            indices
        )

        current_sizes = {
            name:
                len(
                    assignments[
                        name
                    ]
                )

            for name
            in assignments
        }


        fitting = []

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            if (
                current_sizes[
                    split_name
                ]
                + group_size
                <= targets[
                    split_name
                ]
            ):

                fitting.append(
                    split_name
                )


        if fitting:

            chosen = min(
                fitting,
                key=lambda name: (
                    current_sizes[
                        name
                    ]
                    /
                    targets[
                        name
                    ]
                )
            )

        else:

            def overflow_score(
                name
            ):

                future = (
                    current_sizes[
                        name
                    ]
                    + group_size
                )

                overflow = max(
                    0,
                    future
                    - targets[
                        name
                    ]
                )

                return (
                    overflow
                    /
                    targets[
                        name
                    ]
                )


            chosen = min(
                [
                    "train",
                    "validation",
                    "test",
                ],
                key=overflow_score
            )


        assignments[
            chosen
        ].extend(
            indices
        )


    # =====================================
    # Safety check
    # =====================================

    train_idx = np.array(
        sorted(
            assignments[
                "train"
            ]
        ),
        dtype=int
    )

    valid_idx = np.array(
        sorted(
            assignments[
                "validation"
            ]
        ),
        dtype=int
    )

    test_idx = np.array(
        sorted(
            assignments[
                "test"
            ]
        ),
        dtype=int
    )


    all_indices = np.concatenate(
        [
            train_idx,
            valid_idx,
            test_idx,
        ]
    )


    if len(
        np.unique(
            all_indices
        )
    ) != len(df):

        raise RuntimeError(
            "Split overlap or "
            "missing compounds."
        )


    return (
        train_idx,
        valid_idx,
        test_idx
    )


# =========================================
# Metrics
# =========================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    spearman = spearmanr(
        y_true,
        y_pred
    ).statistic


    return {
        "MAE":
            float(
                mae
            ),

        "RMSE":
            float(
                rmse
            ),

        "R2":
            float(
                r2
            ),

        "Spearman":
            float(
                spearman
            ),
    }


# =========================================
# Repeated scaffold evaluation
# =========================================

all_results = []


for seed in SEEDS:

    print(
        "\n================================="
    )

    print(
        f"Scaffold split seed: {seed}"
    )

    print(
        "================================="
    )


    (
        train_idx,
        valid_idx,
        test_idx
    ) = make_scaffold_split(
        seed
    )


    print(
        "Train:",
        len(train_idx)
    )

    print(
        "Validation:",
        len(valid_idx)
    )

    print(
        "Test:",
        len(test_idx)
    )


    # =====================================
    # Verify scaffold leakage
    # =====================================

    train_scaffolds = set(
        df.iloc[
            train_idx
        ][
            "split_scaffold_id"
        ].astype(str)
    )

    valid_scaffolds = set(
        df.iloc[
            valid_idx
        ][
            "split_scaffold_id"
        ].astype(str)
    )

    test_scaffolds = set(
        df.iloc[
            test_idx
        ][
            "split_scaffold_id"
        ].astype(str)
    )


    if (
        train_scaffolds
        & valid_scaffolds
    ):

        raise RuntimeError(
            "Train-validation "
            "scaffold leakage."
        )


    if (
        train_scaffolds
        & test_scaffolds
    ):

        raise RuntimeError(
            "Train-test "
            "scaffold leakage."
        )


    if (
        valid_scaffolds
        & test_scaffolds
    ):

        raise RuntimeError(
            "Validation-test "
            "scaffold leakage."
        )


    # =====================================
    # Evaluate model configurations
    # =====================================

    for (
        model_name,
        params
    ) in MODEL_CONFIGS.items():

        print(
            f"\nTraining {model_name}..."
        )

        start_time = time.time()


        model = RandomForestRegressor(

            n_estimators=
                params[
                    "n_estimators"
                ],

            max_features=
                params[
                    "max_features"
                ],

            min_samples_leaf=
                params[
                    "min_samples_leaf"
                ],

            random_state=
                seed,

            n_jobs=-1,
        )


        model.fit(
            X[
                train_idx
            ],
            y[
                train_idx
            ]
        )


        train_pred = (
            model.predict(
                X[
                    train_idx
                ]
            )
        )

        valid_pred = (
            model.predict(
                X[
                    valid_idx
                ]
            )
        )

        test_pred = (
            model.predict(
                X[
                    test_idx
                ]
            )
        )


        train_metrics = (
            calculate_metrics(
                y[
                    train_idx
                ],
                train_pred
            )
        )

        valid_metrics = (
            calculate_metrics(
                y[
                    valid_idx
                ],
                valid_pred
            )
        )

        test_metrics = (
            calculate_metrics(
                y[
                    test_idx
                ],
                test_pred
            )
        )


        elapsed = (
            time.time()
            - start_time
        )


        result = {

            "seed":
                seed,

            "model":
                model_name,

            "n_train":
                len(
                    train_idx
                ),

            "n_validation":
                len(
                    valid_idx
                ),

            "n_test":
                len(
                    test_idx
                ),

            "train_MAE":
                train_metrics[
                    "MAE"
                ],

            "train_RMSE":
                train_metrics[
                    "RMSE"
                ],

            "train_R2":
                train_metrics[
                    "R2"
                ],

            "train_Spearman":
                train_metrics[
                    "Spearman"
                ],

            "valid_MAE":
                valid_metrics[
                    "MAE"
                ],

            "valid_RMSE":
                valid_metrics[
                    "RMSE"
                ],

            "valid_R2":
                valid_metrics[
                    "R2"
                ],

            "valid_Spearman":
                valid_metrics[
                    "Spearman"
                ],

            "test_MAE":
                test_metrics[
                    "MAE"
                ],

            "test_RMSE":
                test_metrics[
                    "RMSE"
                ],

            "test_R2":
                test_metrics[
                    "R2"
                ],

            "test_Spearman":
                test_metrics[
                    "Spearman"
                ],

            "R2_generalization_gap":
                (
                    train_metrics[
                        "R2"
                    ]
                    -
                    valid_metrics[
                        "R2"
                    ]
                ),

            "training_seconds":
                elapsed,
        }


        all_results.append(
            result
        )


        print(
            "Train R2:",
            round(
                train_metrics[
                    "R2"
                ],
                4
            )
        )

        print(
            "Validation R2:",
            round(
                valid_metrics[
                    "R2"
                ],
                4
            )
        )

        print(
            "Validation RMSE:",
            round(
                valid_metrics[
                    "RMSE"
                ],
                4
            )
        )

        print(
            "Validation Spearman:",
            round(
                valid_metrics[
                    "Spearman"
                ],
                4
            )
        )

        print(
            "Test R2:",
            round(
                test_metrics[
                    "R2"
                ],
                4
            )
        )


# =========================================
# Detailed results
# =========================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    DETAIL_FILE,
    index=False
)


# =========================================
# Aggregate summary
#
# Avoid pandas groupby iteration.
# =========================================

summary_records = []


for model_name in MODEL_CONFIGS:

    subset = results_df[
        results_df[
            "model"
        ]
        == model_name
    ]


    summary_record = {
        "model":
            model_name,

        "n_runs":
            len(
                subset
            ),
    }


    metrics = [
        "train_R2",
        "valid_MAE",
        "valid_RMSE",
        "valid_R2",
        "valid_Spearman",
        "test_MAE",
        "test_RMSE",
        "test_R2",
        "test_Spearman",
        "R2_generalization_gap",
    ]


    for metric in metrics:

        values = subset[
            metric
        ].to_numpy(
            dtype=float
        )

        summary_record[
            f"{metric}_mean"
        ] = float(
            np.mean(
                values
            )
        )

        summary_record[
            f"{metric}_std"
        ] = float(
            np.std(
                values,
                ddof=1
            )
        )


    summary_records.append(
        summary_record
    )


summary_df = pd.DataFrame(
    summary_records
)


summary_df = summary_df.sort_values(
    "valid_RMSE_mean",
    ascending=True
).reset_index(
    drop=True
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# =========================================
# Validation-based ranking
# =========================================

print(
    "\n================================="
)

print(
    "REPEATED SCAFFOLD SPLIT SUMMARY"
)

print(
    "================================="
)


display_columns = [
    "model",

    "valid_MAE_mean",
    "valid_MAE_std",

    "valid_RMSE_mean",
    "valid_RMSE_std",

    "valid_R2_mean",
    "valid_R2_std",

    "valid_Spearman_mean",
    "valid_Spearman_std",

    "R2_generalization_gap_mean",
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


# =========================================
# Test stability
#
# Report only.
# Do NOT use for model selection.
# =========================================

print(
    "\n--- Repeated holdout test ---"
)

test_columns = [
    "model",

    "test_MAE_mean",
    "test_MAE_std",

    "test_RMSE_mean",
    "test_RMSE_std",

    "test_R2_mean",
    "test_R2_std",

    "test_Spearman_mean",
    "test_Spearman_std",
]


print(
    summary_df[
        test_columns
    ].to_string(
        index=False
    )
)


# =========================================
# Recommended model based ONLY
# on mean validation RMSE
# =========================================

recommended_model = (
    summary_df
    .iloc[0][
        "model"
    ]
)


print(
    "\nRecommended configuration "
    "from repeated validation:"
)

print(
    recommended_model
)


# =========================================
# Plot validation R2
# =========================================

plt.figure(
    figsize=(8, 5)
)


for model_name in MODEL_CONFIGS:

    subset = results_df[
        results_df[
            "model"
        ]
        == model_name
    ].sort_values(
        "seed"
    )


    plt.plot(
        subset[
            "seed"
        ],
        subset[
            "valid_R2"
        ],
        marker="o",
        label=model_name
    )


plt.xlabel(
    "Scaffold split seed"
)

plt.ylabel(
    "Validation R²"
)

plt.title(
    "EGFR QSAR: "
    "Repeated Scaffold Split Stability"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_FILE,
    dpi=300
)

plt.close()


# =========================================
# Output locations
# =========================================

print(
    "\nSaved detailed metrics:"
)

print(
    DETAIL_FILE
)

print(
    "\nSaved summary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nSaved figure:"
)

print(
    FIGURE_FILE
)
