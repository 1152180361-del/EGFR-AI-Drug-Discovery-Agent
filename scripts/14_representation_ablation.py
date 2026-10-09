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

QSAR_FILE = (
    ROOT
    / "data"
    / "processed"
    / "egfr_qsar_scaffolds.csv"
)

DESCRIPTOR_FILE = (
    ROOT
    / "data"
    / "features"
    / "egfr_physchem_descriptors.csv"
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
    / "phase2_representation_ablation_metrics.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "phase2_representation_ablation_summary.csv"
)

FIGURE_FILE = (
    FIGURE_DIR
    / "phase2_representation_ablation_r2.png"
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

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
}


DESCRIPTOR_COLUMNS = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "AromaticRingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "FormalCharge",
]


# =========================================
# Load QSAR dataset
# =========================================

qsar = pd.read_csv(
    QSAR_FILE,
    low_memory=False
)

descriptors = pd.read_csv(
    DESCRIPTOR_FILE,
    low_memory=False
)

print(
    "QSAR compounds:",
    len(qsar)
)

print(
    "Descriptor compounds:",
    len(descriptors)
)


# =========================================
# Descriptor table lookup
#
# Use standardized InChIKey as identity.
# =========================================

descriptor_lookup = {}

for record in descriptors.to_dict(
    "records"
):

    key = str(
        record[
            "standardized_inchikey"
        ]
    )

    descriptor_lookup[key] = record


# =========================================
# Align descriptors with QSAR dataset
# =========================================

descriptor_matrix = []

missing_descriptor_keys = []


for record in qsar.to_dict(
    "records"
):

    key = str(
        record[
            "standardized_inchikey"
        ]
    )

    if key not in descriptor_lookup:

        missing_descriptor_keys.append(
            key
        )

        continue


    source = descriptor_lookup[
        key
    ]

    descriptor_vector = [
        float(
            source[column]
        )
        for column
        in DESCRIPTOR_COLUMNS
    ]

    descriptor_matrix.append(
        descriptor_vector
    )


if missing_descriptor_keys:

    raise RuntimeError(
        "Descriptor alignment failed. "
        f"Missing compounds: "
        f"{len(missing_descriptor_keys)}"
    )


X_desc = np.asarray(
    descriptor_matrix,
    dtype=np.float32
)


print(
    "Descriptor matrix:",
    X_desc.shape
)


# =========================================
# Generate Morgan fingerprints
# =========================================

generator = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_SIZE
    )
)


morgan_vectors = []


print(
    "\nGenerating Morgan fingerprints..."
)


for i, smiles in enumerate(
    qsar[
        "standardized_smiles"
    ].tolist(),
    start=1
):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:

        raise RuntimeError(
            f"RDKit parsing failed "
            f"at compound {i}: "
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

    morgan_vectors.append(
        array
    )


    if i % 500 == 0:

        print(
            f"Processed "
            f"{i}/{len(qsar)}"
        )


X_morgan = np.asarray(
    morgan_vectors,
    dtype=np.uint8
)


print(
    "Morgan matrix:",
    X_morgan.shape
)


# =========================================
# Combined representation
# =========================================

X_combined = np.concatenate(
    [
        X_morgan.astype(
            np.float32
        ),
        X_desc
    ],
    axis=1
)


print(
    "Combined matrix:",
    X_combined.shape
)


# =========================================
# Labels
# =========================================

y = qsar[
    "pIC50"
].to_numpy(
    dtype=np.float32
)


# =========================================
# Sanity checks
# =========================================

if not (
    len(X_desc)
    == len(X_morgan)
    == len(X_combined)
    == len(y)
    == len(qsar)
):

    raise RuntimeError(
        "Feature alignment failure."
    )


if np.isnan(
    X_desc
).any():

    raise RuntimeError(
        "NaN detected in descriptor matrix."
    )


if np.isinf(
    X_desc
).any():

    raise RuntimeError(
        "Infinite descriptor value detected."
    )


print(
    "\nFeature alignment: PASS"
)


# =========================================
# Representations
# =========================================

REPRESENTATIONS = {
    "Descriptors":
        X_desc,

    "Morgan":
        X_morgan,

    "Morgan+Descriptors":
        X_combined,
}


# =========================================
# Build scaffold groups
# =========================================

scaffold_groups = {}

for index, scaffold_id in enumerate(
    qsar[
        "split_scaffold_id"
    ].astype(str)
):

    scaffold_groups.setdefault(
        scaffold_id,
        []
    ).append(
        index
    )


print(
    "Unique scaffolds:",
    len(scaffold_groups)
)


# =========================================
# Scaffold split function
#
# Same algorithm as previous experiments.
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


    n_total = len(qsar)

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


    for scaffold_id, indices in scaffold_items:

        group_size = len(
            indices
        )


        current_sizes = {
            split_name:
                len(
                    assignments[
                        split_name
                    ]
                )

            for split_name
            in assignments
        }


        fitting_splits = []

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

                fitting_splits.append(
                    split_name
                )


        if fitting_splits:

            chosen_split = min(
                fitting_splits,
                key=lambda split_name: (
                    current_sizes[
                        split_name
                    ]
                    /
                    targets[
                        split_name
                    ]
                )
            )


        else:

            def overflow_score(
                split_name
            ):

                future_size = (
                    current_sizes[
                        split_name
                    ]
                    + group_size
                )

                overflow = max(
                    0,
                    future_size
                    - targets[
                        split_name
                    ]
                )

                return (
                    overflow
                    /
                    targets[
                        split_name
                    ]
                )


            chosen_split = min(
                [
                    "train",
                    "validation",
                    "test",
                ],
                key=overflow_score
            )


        assignments[
            chosen_split
        ].extend(
            indices
        )


    train_idx = np.asarray(
        sorted(
            assignments[
                "train"
            ]
        ),
        dtype=int
    )

    valid_idx = np.asarray(
        sorted(
            assignments[
                "validation"
            ]
        ),
        dtype=int
    )

    test_idx = np.asarray(
        sorted(
            assignments[
                "test"
            ]
        ),
        dtype=int
    )


    # =====================================
    # Scaffold leakage check
    # =====================================

    train_scaffolds = set(
        qsar.iloc[
            train_idx
        ][
            "split_scaffold_id"
        ].astype(str)
    )

    valid_scaffolds = set(
        qsar.iloc[
            valid_idx
        ][
            "split_scaffold_id"
        ].astype(str)
    )

    test_scaffolds = set(
        qsar.iloc[
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
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Spearman": float(spearman),
    }


# =========================================
# Repeated representation ablation
# =========================================

results = []


for seed in SEEDS:

    print(
        "\n================================="
    )

    print(
        f"Scaffold seed: {seed}"
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


    for (
        representation_name,
        X
    ) in REPRESENTATIONS.items():

        print(
            f"\nTraining "
            f"{representation_name}..."
        )

        start_time = time.time()


        model = RandomForestRegressor(

            n_estimators=
                RANDOM_FOREST_PARAMS[
                    "n_estimators"
                ],

            max_features=
                RANDOM_FOREST_PARAMS[
                    "max_features"
                ],

            min_samples_leaf=
                RANDOM_FOREST_PARAMS[
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


        train_prediction = (
            model.predict(
                X[
                    train_idx
                ]
            )
        )

        valid_prediction = (
            model.predict(
                X[
                    valid_idx
                ]
            )
        )

        test_prediction = (
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
                train_prediction
            )
        )

        valid_metrics = (
            calculate_metrics(
                y[
                    valid_idx
                ],
                valid_prediction
            )
        )

        test_metrics = (
            calculate_metrics(
                y[
                    test_idx
                ],
                test_prediction
            )
        )


        elapsed = (
            time.time()
            - start_time
        )


        result = {
            "seed":
                seed,

            "representation":
                representation_name,

            "n_features":
                X.shape[1],

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


        results.append(
            result
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
# Save detailed results
# =========================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    DETAIL_FILE,
    index=False
)


# =========================================
# Aggregate summary
# =========================================

summary_records = []


for representation_name in REPRESENTATIONS:

    subset = results_df[
        results_df[
            "representation"
        ]
        == representation_name
    ]


    summary_record = {
        "representation":
            representation_name,

        "n_features":
            int(
                subset[
                    "n_features"
                ].iloc[0]
            ),

        "n_runs":
            len(subset),
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
# Validation-based comparison
# =========================================

print(
    "\n================================="
)

print(
    "PHASE 2.6 REPRESENTATION ABLATION"
)

print(
    "================================="
)


validation_columns = [
    "representation",
    "n_features",

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
        validation_columns
    ].to_string(
        index=False
    )
)


# =========================================
# Test results
#
# Reporting only.
# Do NOT select representation from test.
# =========================================

print(
    "\n--- Repeated holdout test ---"
)


test_columns = [
    "representation",

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
# Best representation
# based on validation RMSE only
# =========================================

best_representation = (
    summary_df
    .iloc[0][
        "representation"
    ]
)


print(
    "\nBest representation "
    "from repeated validation:"
)

print(
    best_representation
)


# =========================================
# Plot validation R2
# =========================================

plt.figure(
    figsize=(9, 5)
)


for representation_name in REPRESENTATIONS:

    subset = results_df[
        results_df[
            "representation"
        ]
        == representation_name
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

        label=
            representation_name
    )


plt.xlabel(
    "Scaffold split seed"
)

plt.ylabel(
    "Validation R²"
)

plt.title(
    "EGFR QSAR Representation Ablation"
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
