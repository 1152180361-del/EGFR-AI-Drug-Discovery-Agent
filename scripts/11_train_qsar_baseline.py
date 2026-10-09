from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
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

FEATURE_DIR = (
    ROOT
    / "data"
    / "features"
)

MODEL_DIR = (
    ROOT
    / "models"
)

REPORT_DIR = (
    ROOT
    / "reports"
)

FIGURE_DIR = (
    ROOT
    / "figures"
)

RESULT_DIR = (
    ROOT
    / "results"
)

for directory in [
    MODEL_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    RESULT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


TRAIN_FILE = (
    FEATURE_DIR
    / "egfr_train_morgan.npz"
)

VALID_FILE = (
    FEATURE_DIR
    / "egfr_validation_morgan.npz"
)

TEST_FILE = (
    FEATURE_DIR
    / "egfr_test_morgan.npz"
)


# =========================================
# Reproducibility
# =========================================

RANDOM_SEED = 42


# =========================================
# Load feature datasets
# =========================================

train_data = np.load(
    TRAIN_FILE
)

valid_data = np.load(
    VALID_FILE
)

test_data = np.load(
    TEST_FILE
)


X_train = train_data["X"]
y_train = train_data["y"]

X_valid = valid_data["X"]
y_valid = valid_data["y"]

X_test = test_data["X"]
y_test = test_data["y"]


print(
    "Train:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_valid.shape,
    y_valid.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)


# =========================================
# Metric function
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
# Candidate model configurations
#
# Validation set is used ONLY here
# for model / hyperparameter selection.
# =========================================

candidate_configs = []


# -----------------------------------------
# Dummy baseline
# -----------------------------------------

candidate_configs.append(
    {
        "name": "DummyMean",
        "family": "Dummy",
        "params": {},
    }
)


# -----------------------------------------
# Ridge candidates
# -----------------------------------------

for alpha in [
    0.1,
    1.0,
    10.0,
    100.0,
]:

    candidate_configs.append(
        {
            "name":
                f"Ridge_alpha_{alpha}",

            "family":
                "Ridge",

            "params":
                {
                    "alpha": alpha
                },
        }
    )


# -----------------------------------------
# Random Forest candidates
# -----------------------------------------

rf_configs = [
    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
    },
    {
        "n_estimators": 300,
        "max_features": 0.3,
        "min_samples_leaf": 1,
    },
    {
        "n_estimators": 300,
        "max_features": 0.3,
        "min_samples_leaf": 2,
    },
]


for i, params in enumerate(
    rf_configs,
    start=1
):

    candidate_configs.append(
        {
            "name":
                f"RandomForest_{i}",

            "family":
                "RandomForest",

            "params":
                params,
        }
    )


# =========================================
# Build model helper
# =========================================

def build_model(
    config
):

    family = config[
        "family"
    ]

    params = config[
        "params"
    ]


    if family == "Dummy":

        return DummyRegressor(
            strategy="mean"
        )


    if family == "Ridge":

        return Ridge(
            alpha=params[
                "alpha"
            ]
        )


    if family == "RandomForest":

        return RandomForestRegressor(
            n_estimators=params[
                "n_estimators"
            ],

            max_features=params[
                "max_features"
            ],

            min_samples_leaf=params[
                "min_samples_leaf"
            ],

            random_state=RANDOM_SEED,

            n_jobs=-1,
        )


    raise ValueError(
        f"Unknown model family: "
        f"{family}"
    )


# =========================================
# Train candidates
#
# IMPORTANT:
# Test set is NOT touched here.
# =========================================

validation_results = []


print(
    "\n================================="
)

print(
    "MODEL SELECTION ON VALIDATION SET"
)

print(
    "================================="
)


for config in candidate_configs:

    print(
        f"\nTraining: "
        f"{config['name']}"
    )

    start_time = time.time()

    model = build_model(
        config
    )

    model.fit(
        X_train,
        y_train
    )

    pred_train = model.predict(
        X_train
    )

    pred_valid = model.predict(
        X_valid
    )


    train_metrics = (
        calculate_metrics(
            y_train,
            pred_train
        )
    )

    valid_metrics = (
        calculate_metrics(
            y_valid,
            pred_valid
        )
    )


    elapsed = (
        time.time()
        - start_time
    )


    result = {
        "model_name":
            config["name"],

        "family":
            config["family"],

        "params":
            json.dumps(
                config["params"]
            ),

        "train_MAE":
            train_metrics["MAE"],

        "train_RMSE":
            train_metrics["RMSE"],

        "train_R2":
            train_metrics["R2"],

        "train_Spearman":
            train_metrics["Spearman"],

        "valid_MAE":
            valid_metrics["MAE"],

        "valid_RMSE":
            valid_metrics["RMSE"],

        "valid_R2":
            valid_metrics["R2"],

        "valid_Spearman":
            valid_metrics["Spearman"],

        "training_seconds":
            elapsed,
    }


    validation_results.append(
        result
    )


    print(
        "Validation MAE:",
        round(
            valid_metrics["MAE"],
            4
        )
    )

    print(
        "Validation RMSE:",
        round(
            valid_metrics["RMSE"],
            4
        )
    )

    print(
        "Validation R2:",
        round(
            valid_metrics["R2"],
            4
        )
    )

    print(
        "Validation Spearman:",
        round(
            valid_metrics["Spearman"],
            4
        )
    )


# =========================================
# Validation ranking
#
# Primary criterion:
# lowest validation RMSE
# =========================================

results_df = pd.DataFrame(
    validation_results
)

results_df = (
    results_df
    .sort_values(
        "valid_RMSE",
        ascending=True
    )
    .reset_index(
        drop=True
    )
)


VALIDATION_REPORT = (
    REPORT_DIR
    / "phase2_qsar_validation_results.csv"
)

results_df.to_csv(
    VALIDATION_REPORT,
    index=False
)


print(
    "\n================================="
)

print(
    "VALIDATION MODEL RANKING"
)

print(
    "================================="
)

print(
    results_df[
        [
            "model_name",
            "valid_MAE",
            "valid_RMSE",
            "valid_R2",
            "valid_Spearman",
        ]
    ].to_string(
        index=False
    )
)


# =========================================
# Select best configuration
# =========================================

best_name = (
    results_df
    .iloc[0][
        "model_name"
    ]
)


best_config = next(
    config
    for config
    in candidate_configs
    if config[
        "name"
    ] == best_name
)


print(
    "\nSelected model:"
)

print(
    best_name
)

print(
    "Parameters:",
    best_config[
        "params"
    ]
)


# =========================================
# Retrain best model using
# Train + Validation
#
# Test remains untouched until after fit.
# =========================================

X_train_final = np.concatenate(
    [
        X_train,
        X_valid,
    ],
    axis=0
)

y_train_final = np.concatenate(
    [
        y_train,
        y_valid,
    ],
    axis=0
)


print(
    "\nFinal training matrix:",
    X_train_final.shape
)


final_model = build_model(
    best_config
)

final_model.fit(
    X_train_final,
    y_train_final
)


# =========================================
# FINAL TEST EVALUATION
#
# Test is evaluated only here.
# =========================================

test_predictions = (
    final_model.predict(
        X_test
    )
)


test_metrics = (
    calculate_metrics(
        y_test,
        test_predictions
    )
)


print(
    "\n================================="
)

print(
    "FINAL SCAFFOLD TEST PERFORMANCE"
)

print(
    "================================="
)

print(
    "Model:",
    best_name
)

print(
    "MAE:",
    round(
        test_metrics["MAE"],
        4
    )
)

print(
    "RMSE:",
    round(
        test_metrics["RMSE"],
        4
    )
)

print(
    "R2:",
    round(
        test_metrics["R2"],
        4
    )
)

print(
    "Spearman:",
    round(
        test_metrics[
            "Spearman"
        ],
        4
    )
)


# =========================================
# Save final model
# =========================================

MODEL_FILE = (
    MODEL_DIR
    / "egfr_qsar_baseline_best.joblib"
)

joblib.dump(
    final_model,
    MODEL_FILE
)


# =========================================
# Save final model metadata
# =========================================

MODEL_METADATA_FILE = (
    MODEL_DIR
    / "egfr_qsar_baseline_best.json"
)


metadata = {
    "selected_model":
        best_name,

    "family":
        best_config[
            "family"
        ],

    "parameters":
        best_config[
            "params"
        ],

    "random_seed":
        RANDOM_SEED,

    "fingerprint":
        {
            "type":
                "Morgan",

            "radius":
                2,

            "fp_size":
                2048,
        },

    "train_final_n":
        int(
            len(
                y_train_final
            )
        ),

    "test_n":
        int(
            len(
                y_test
            )
        ),

    "test_metrics":
        test_metrics,
}


with open(
    MODEL_METADATA_FILE,
    "w"
) as handle:

    json.dump(
        metadata,
        handle,
        indent=4
    )


# =========================================
# Save test predictions
# =========================================

prediction_df = pd.DataFrame(
    {
        "standardized_inchikey":
            test_data[
                "inchikey"
            ],

        "standardized_smiles":
            test_data[
                "smiles"
            ],

        "scaffold_id":
            test_data[
                "scaffold_id"
            ],

        "observed_pIC50":
            y_test,

        "predicted_pIC50":
            test_predictions,

        "residual":
            (
                y_test
                - test_predictions
            ),

        "absolute_error":
            np.abs(
                y_test
                - test_predictions
            ),
    }
)


PREDICTION_FILE = (
    RESULT_DIR
    / "egfr_qsar_test_predictions.csv"
)

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# =========================================
# Observed vs predicted figure
# =========================================

plt.figure(
    figsize=(6, 6)
)

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.6
)

lower = min(
    float(
        np.min(
            y_test
        )
    ),
    float(
        np.min(
            test_predictions
        )
    )
)

upper = max(
    float(
        np.max(
            y_test
        )
    ),
    float(
        np.max(
            test_predictions
        )
    )
)

plt.plot(
    [
        lower,
        upper
    ],
    [
        lower,
        upper
    ],
    linestyle="--"
)

plt.xlabel(
    "Observed pIC50"
)

plt.ylabel(
    "Predicted pIC50"
)

plt.title(
    "EGFR QSAR: "
    "Scaffold Test Set"
)

plt.tight_layout()


SCATTER_FILE = (
    FIGURE_DIR
    / "phase2_qsar_test_observed_vs_predicted.png"
)

plt.savefig(
    SCATTER_FILE,
    dpi=300
)

plt.close()


# =========================================
# Residual distribution
# =========================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    prediction_df[
        "residual"
    ],
    bins=30
)

plt.xlabel(
    "Residual "
    "(Observed - Predicted)"
)

plt.ylabel(
    "Number of compounds"
)

plt.title(
    "EGFR QSAR Test Residuals"
)

plt.tight_layout()


RESIDUAL_FILE = (
    FIGURE_DIR
    / "phase2_qsar_test_residuals.png"
)

plt.savefig(
    RESIDUAL_FILE,
    dpi=300
)

plt.close()


# =========================================
# Error analysis
# =========================================

worst_predictions = (
    prediction_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)


WORST_FILE = (
    REPORT_DIR
    / "phase2_qsar_largest_test_errors.csv"
)

worst_predictions.to_csv(
    WORST_FILE,
    index=False
)


# =========================================
# Final output summary
# =========================================

print(
    "\nSaved validation comparison:"
)

print(
    VALIDATION_REPORT
)

print(
    "\nSaved final model:"
)

print(
    MODEL_FILE
)

print(
    "\nSaved model metadata:"
)

print(
    MODEL_METADATA_FILE
)

print(
    "\nSaved test predictions:"
)

print(
    PREDICTION_FILE
)

print(
    "\nSaved largest test errors:"
)

print(
    WORST_FILE
)

print(
    "\nSaved figures:"
)

print(
    SCATTER_FILE
)

print(
    RESIDUAL_FILE
)
