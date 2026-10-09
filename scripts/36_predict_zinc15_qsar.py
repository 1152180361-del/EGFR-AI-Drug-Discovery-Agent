from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
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
    / "data/processed/external/"
      "zinc15_250k_screening_library.csv"
)

MODEL_PATH = (
    ROOT
    / "models/egfr_qsar_production_rf.joblib"
)

OUT_DIR = (
    ROOT
    / "results/zinc15_screening"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PREDICTIONS_OUT = (
    OUT_DIR
    / "zinc15_221988_qsar_predictions.csv"
)

TOP20000_OUT = (
    OUT_DIR
    / "zinc15_qsar_top20000.csv"
)

TOP5000_OUT = (
    OUT_DIR
    / "zinc15_qsar_top5000.csv"
)

TOP1000_OUT = (
    OUT_DIR
    / "zinc15_qsar_top1000.csv"
)

REPORT_OUT = (
    OUT_DIR
    / "zinc15_qsar_prediction_report.json"
)


# ============================================================
# Settings
# ============================================================

RADIUS = 2
N_BITS = 2048

BATCH_SIZE = 5000


# ============================================================
# Load screening library
# ============================================================

df = pd.read_csv(
    INPUT
)


print(
    "\n========================================"
)

print(
    "ZINC15 PRODUCTION QSAR SCREENING"
)

print(
    "========================================"
)


print(
    "\nInput molecules:",
    len(df)
)


required = [
    "screening_library_id",
    "zinc_id",
    "standardized_smiles",
]


missing = [
    x
    for x in required
    if x not in df.columns
]


if missing:

    raise RuntimeError(
        f"Missing required columns: "
        f"{missing}"
    )


# ============================================================
# Load production RF
# ============================================================

print(
    "\nLoading production RF..."
)


model = joblib.load(
    MODEL_PATH
)


print(
    "Model:",
    type(model).__name__
)


if hasattr(
    model,
    "n_estimators"
):

    print(
        "Trees:",
        model.n_estimators
    )


if hasattr(
    model,
    "n_features_in_"
):

    print(
        "Expected features:",
        model.n_features_in_
    )


    if int(
        model.n_features_in_
    ) != N_BITS:

        raise RuntimeError(
            "Model feature dimension "
            f"{model.n_features_in_} "
            f"!= expected Morgan {N_BITS}"
        )


# ============================================================
# Morgan fingerprint helper
#
# Same representation as Phase 2:
# radius = 2
# 2048 bits
# ============================================================

def smiles_to_fp_array(
    smiles,
):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )


    if mol is None:

        return None


    fp = (
        AllChem
        .GetMorganFingerprintAsBitVect(
            mol,
            radius=RADIUS,
            nBits=N_BITS,
        )
    )


    arr = np.zeros(
        (
            N_BITS,
        ),
        dtype=np.uint8,
    )


    DataStructs.ConvertToNumpyArray(
        fp,
        arr,
    )


    return arr


# ============================================================
# Batch prediction
# ============================================================

predicted = np.full(
    len(df),
    np.nan,
    dtype=np.float32,
)


valid_fp = np.zeros(
    len(df),
    dtype=bool,
)


start_time = time.time()


for start in range(
    0,
    len(df),
    BATCH_SIZE,
):

    end = min(
        start + BATCH_SIZE,
        len(df),
    )


    batch = df.iloc[
        start:end
    ]


    arrays = []

    indices = []


    for idx, smiles in zip(
        batch.index,
        batch[
            "standardized_smiles"
        ],
    ):

        fp = smiles_to_fp_array(
            smiles
        )


        if fp is None:

            continue


        arrays.append(
            fp
        )

        indices.append(
            idx
        )


    if arrays:

        X = np.stack(
            arrays,
            axis=0,
        )


        y_pred = model.predict(
            X
        )


        predicted[
            indices
        ] = (
            np.asarray(
                y_pred,
                dtype=np.float32,
            )
        )


        valid_fp[
            indices
        ] = True


    elapsed = (
        time.time()
        - start_time
    )


    print(
        f"Predicted "
        f"{end:,}/{len(df):,} "
        f"| elapsed "
        f"{elapsed/60:.1f} min",
        flush=True,
    )


# ============================================================
# Attach predictions
# ============================================================

df[
    "qsar_predicted_pIC50"
] = predicted


df[
    "qsar_prediction_valid"
] = valid_fp


failures = int(
    (~valid_fp).sum()
)


print(
    "\nPrediction failures:",
    failures
)


if failures > 0:

    print(
        "WARNING: some molecules "
        "could not be fingerprinted."
    )


# ============================================================
# Rank
#
# Higher predicted pIC50 = stronger predicted activity
# ============================================================

ranked = (
    df[
        df[
            "qsar_prediction_valid"
        ]
    ]
    .sort_values(
        "qsar_predicted_pIC50",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


ranked[
    "qsar_rank"
] = np.arange(
    1,
    len(ranked) + 1,
)


# ============================================================
# Simple activity interpretation
#
# This is NOT an experimental claim.
# pIC50 8 corresponds numerically to 10 nM.
# ============================================================

ranked[
    "predicted_pIC50_ge_8"
] = (
    ranked[
        "qsar_predicted_pIC50"
    ]
    >= 8.0
)


ranked[
    "predicted_pIC50_ge_9"
] = (
    ranked[
        "qsar_predicted_pIC50"
    ]
    >= 9.0
)


# ============================================================
# Save all
# ============================================================

ranked.to_csv(
    PREDICTIONS_OUT,
    index=False,
)


# ============================================================
# Candidate subsets
# ============================================================

top20000 = ranked.head(
    20000
).copy()


top5000 = ranked.head(
    5000
).copy()


top1000 = ranked.head(
    1000
).copy()


top20000.to_csv(
    TOP20000_OUT,
    index=False,
)


top5000.to_csv(
    TOP5000_OUT,
    index=False,
)


top1000.to_csv(
    TOP1000_OUT,
    index=False,
)


# ============================================================
# Prediction statistics
# ============================================================

scores = ranked[
    "qsar_predicted_pIC50"
]


quantiles = {
    str(q):
        float(
            scores.quantile(
                q
            )
        )

    for q in [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]
}


n_ge8 = int(
    (
        scores >= 8.0
    ).sum()
)


n_ge9 = int(
    (
        scores >= 9.0
    ).sum()
)


# ============================================================
# Report
# ============================================================

report = {
    "input_library_size":
        int(
            len(df)
        ),

    "valid_predictions":
        int(
            valid_fp.sum()
        ),

    "prediction_failures":
        int(
            failures
        ),

    "model":
        str(
            MODEL_PATH
        ),

    "representation": {
        "type":
            "Morgan fingerprint",

        "radius":
            RADIUS,

        "n_bits":
            N_BITS,
    },

    "prediction_summary": {
        "mean":
            float(
                scores.mean()
            ),

        "std":
            float(
                scores.std()
            ),

        "min":
            float(
                scores.min()
            ),

        "median":
            float(
                scores.median()
            ),

        "max":
            float(
                scores.max()
            ),

        "quantiles":
            quantiles,

        "predicted_pIC50_ge_8":
            n_ge8,

        "predicted_pIC50_ge_9":
            n_ge9,
    },

    "outputs": {
        "all_predictions":
            str(
                PREDICTIONS_OUT
            ),

        "top20000":
            str(
                TOP20000_OUT
            ),

        "top5000":
            str(
                TOP5000_OUT
            ),

        "top1000":
            str(
                TOP1000_OUT
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
    "ZINC15 QSAR SCREENING RESULTS"
)

print(
    "========================================"
)


print(
    "\nValid predictions:",
    len(ranked)
)

print(
    "Failures:",
    failures
)


print(
    "\nPredicted pIC50:"
)

print(
    "Mean:",
    f"{scores.mean():.3f}"
)

print(
    "Std:",
    f"{scores.std():.3f}"
)

print(
    "Median:",
    f"{scores.median():.3f}"
)

print(
    "Min:",
    f"{scores.min():.3f}"
)

print(
    "Max:",
    f"{scores.max():.3f}"
)


print(
    "\nSelected quantiles:"
)

for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]:

    print(
        f"{int(q*100):2d}th percentile:",
        f"{scores.quantile(q):.3f}"
    )


print(
    "\nPredicted pIC50 >= 8:",
    n_ge8
)

print(
    "Predicted pIC50 >= 9:",
    n_ge9
)


print(
    "\nTOP 20 QSAR PREDICTIONS"
)

print(
    ranked[
        [
            "qsar_rank",
            "zinc_id",
            "qsar_predicted_pIC50",
            "mwt",
            "logp",
            "standardized_smiles",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    PREDICTIONS_OUT
)

print(
    TOP20000_OUT
)

print(
    TOP5000_OUT
)

print(
    TOP1000_OUT
)

print(
    REPORT_OUT
)


print(
    "\n========================================"
)

print(
    "PHASE 4B-1 QSAR PREDICTION: PASS"
)

print(
    "========================================"
)
