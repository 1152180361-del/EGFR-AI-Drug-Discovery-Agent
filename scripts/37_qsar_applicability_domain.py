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

ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

TRAINING_PATH = (
    ROOT
    / "data/processed/egfr_qsar_modeling.csv"
)

TOP20000_PATH = (
    ROOT
    / "results/zinc15_screening/"
      "zinc15_qsar_top20000.csv"
)

MODEL_PATH = (
    ROOT
    / "models/egfr_qsar_production_rf.joblib"
)

OUT_DIR = (
    ROOT
    / "results/zinc15_screening"
)

AD_OUT = (
    OUT_DIR
    / "zinc15_top20000_qsar_AD.csv"
)

PRIORITY1000_OUT = (
    OUT_DIR
    / "zinc15_qsar_AD_priority_top1000.csv"
)

PRIORITY100_OUT = (
    OUT_DIR
    / "zinc15_qsar_AD_priority_top100.csv"
)

TRAINING_NN_OUT = (
    OUT_DIR
    / "egfr_training_LOO_nearest_similarity.csv"
)

REPORT_OUT = (
    OUT_DIR
    / "zinc15_qsar_AD_report.json"
)


# ============================================================
# Settings
# ============================================================

RADIUS = 2
N_BITS = 2048

# Predefined before examining ZINC AD results.
AD_PERCENTILE = 0.05

TREE_BATCH_SIZE = 5000


# ============================================================
# Fingerprint helper
# ============================================================

def make_fp(smiles):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    return (
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=RADIUS,
            nBits=N_BITS,
        )
    )


def fp_to_array(fp):

    arr = np.zeros(
        N_BITS,
        dtype=np.uint8,
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr,
    )

    return arr


# ============================================================
# Load
# ============================================================

training = pd.read_csv(
    TRAINING_PATH
)

queries = pd.read_csv(
    TOP20000_PATH
)

model = joblib.load(
    MODEL_PATH
)


print(
    "\n========================================"
)

print(
    "EGFR QSAR APPLICABILITY DOMAIN"
)

print(
    "========================================"
)

print(
    "\nTraining compounds:",
    len(training)
)

print(
    "External candidates:",
    len(queries)
)

print(
    "RF trees:",
    len(model.estimators_)
)


# ============================================================
# Build training fingerprints
# ============================================================

print(
    "\nBuilding training fingerprints..."
)


train_fps = []
train_arrays = []
train_rows = []


for idx, row in training.iterrows():

    fp = make_fp(
        row[
            "standardized_smiles"
        ]
    )

    if fp is None:
        continue

    train_fps.append(
        fp
    )

    train_arrays.append(
        fp_to_array(
            fp
        )
    )

    train_rows.append(
        idx
    )


if len(train_fps) != len(training):

    print(
        "WARNING:",
        len(training) - len(train_fps),
        "training molecules failed fingerprinting."
    )


train_meta = (
    training
    .loc[
        train_rows
    ]
    .reset_index(
        drop=True
    )
)


X_train = np.stack(
    train_arrays,
    axis=0,
)


print(
    "Valid training fingerprints:",
    len(train_fps)
)


# ============================================================
# Training leave-one-out nearest-neighbor similarity
# ============================================================

print(
    "\nCalculating training LOO nearest-neighbor similarities..."
)


train_nn_similarity = np.zeros(
    len(train_fps),
    dtype=np.float32,
)

train_nn_index = np.zeros(
    len(train_fps),
    dtype=np.int32,
)


start_time = time.time()


for i, fp in enumerate(
    train_fps
):

    sims = np.asarray(
        DataStructs.BulkTanimotoSimilarity(
            fp,
            train_fps,
        ),
        dtype=np.float32,
    )

    # Exclude self
    sims[i] = -1.0

    j = int(
        np.argmax(
            sims
        )
    )

    train_nn_index[i] = j

    train_nn_similarity[i] = (
        sims[j]
    )


    if (
        (i + 1) % 500 == 0
        or i + 1 == len(train_fps)
    ):

        print(
            f"  LOO similarity "
            f"{i + 1}/{len(train_fps)}",
            flush=True,
        )


# ============================================================
# Data-driven AD reference
# ============================================================

ad_threshold = float(
    np.quantile(
        train_nn_similarity,
        AD_PERCENTILE,
    )
)


training_similarity_quantiles = {
    "min":
        float(
            np.min(
                train_nn_similarity
            )
        ),

    "p01":
        float(
            np.quantile(
                train_nn_similarity,
                0.01,
            )
        ),

    "p05":
        float(
            np.quantile(
                train_nn_similarity,
                0.05,
            )
        ),

    "p10":
        float(
            np.quantile(
                train_nn_similarity,
                0.10,
            )
        ),

    "p25":
        float(
            np.quantile(
                train_nn_similarity,
                0.25,
            )
        ),

    "median":
        float(
            np.median(
                train_nn_similarity
            )
        ),

    "p75":
        float(
            np.quantile(
                train_nn_similarity,
                0.75,
            )
        ),

    "p90":
        float(
            np.quantile(
                train_nn_similarity,
                0.90,
            )
        ),

    "p95":
        float(
            np.quantile(
                train_nn_similarity,
                0.95,
            )
        ),

    "max":
        float(
            np.max(
                train_nn_similarity
            )
        ),
}


# ============================================================
# Save training nearest-neighbor table
# ============================================================

training_nn = train_meta.copy()


training_nn[
    "loo_nearest_similarity"
] = train_nn_similarity


training_nn[
    "loo_nearest_training_index"
] = train_nn_index


training_nn[
    "loo_nearest_pIC50"
] = [
    float(
        train_meta.iloc[j][
            "pIC50"
        ]
    )
    for j in train_nn_index
]


training_nn[
    "loo_nearest_smiles"
] = [
    str(
        train_meta.iloc[j][
            "standardized_smiles"
        ]
    )
    for j in train_nn_index
]


training_nn.to_csv(
    TRAINING_NN_OUT,
    index=False,
)


print(
    "\nTraining similarity reference:"
)

for key, value in (
    training_similarity_quantiles.items()
):

    print(
        f"{key:>6s}: "
        f"{value:.3f}"
    )


print(
    "\nSimilarity AD threshold "
    f"(training LOO {int(AD_PERCENTILE*100)}th percentile): "
    f"{ad_threshold:.3f}"
)


# ============================================================
# RF tree-spread internal reference
#
# This is NOT a calibrated uncertainty interval.
# It is only tree-to-tree prediction dispersion.
# ============================================================

print(
    "\nCalculating training RF tree spread..."
)


training_tree_predictions = []


for tree in model.estimators_:

    pred = tree.predict(
        X_train
    )

    training_tree_predictions.append(
        np.asarray(
            pred,
            dtype=np.float32,
        )
    )


training_tree_predictions = np.stack(
    training_tree_predictions,
    axis=0,
)


train_rf_mean = (
    training_tree_predictions
    .mean(
        axis=0
    )
)


train_rf_sd = (
    training_tree_predictions
    .std(
        axis=0,
        ddof=1,
    )
)


rf_sd_p95 = float(
    np.quantile(
        train_rf_sd,
        0.95,
    )
)


rf_sd_median = float(
    np.median(
        train_rf_sd
    )
)


print(
    "Training RF-tree SD median:",
    f"{rf_sd_median:.3f}"
)

print(
    "Training RF-tree SD 95th percentile:",
    f"{rf_sd_p95:.3f}"
)


# Free temporary matrix
del training_tree_predictions


# ============================================================
# Query fingerprints
# ============================================================

print(
    "\nBuilding Top-20,000 fingerprints..."
)


query_fps = []
query_arrays = []
valid_query_rows = []


for idx, row in queries.iterrows():

    fp = make_fp(
        row[
            "standardized_smiles"
        ]
    )

    if fp is None:
        continue

    query_fps.append(
        fp
    )

    query_arrays.append(
        fp_to_array(
            fp
        )
    )

    valid_query_rows.append(
        idx
    )


query = (
    queries
    .loc[
        valid_query_rows
    ]
    .reset_index(
        drop=True
    )
)


X_query = np.stack(
    query_arrays,
    axis=0,
)


print(
    "Valid query fingerprints:",
    len(query_fps)
)


# ============================================================
# Nearest training compound
# ============================================================

print(
    "\nCalculating external -> training nearest similarity..."
)


query_max_similarity = np.zeros(
    len(query_fps),
    dtype=np.float32,
)

query_nn_index = np.zeros(
    len(query_fps),
    dtype=np.int32,
)


start = time.time()


for i, fp in enumerate(
    query_fps
):

    sims = np.asarray(
        DataStructs.BulkTanimotoSimilarity(
            fp,
            train_fps,
        ),
        dtype=np.float32,
    )

    j = int(
        np.argmax(
            sims
        )
    )

    query_nn_index[i] = j

    query_max_similarity[i] = (
        sims[j]
    )


    if (
        (i + 1) % 500 == 0
        or i + 1 == len(query_fps)
    ):

        elapsed = (
            time.time()
            - start
        )

        print(
            f"  Similarity "
            f"{i + 1:,}/{len(query_fps):,} "
            f"| {elapsed/60:.1f} min",
            flush=True,
        )


# ============================================================
# Query RF tree spread
# ============================================================

print(
    "\nCalculating RF ensemble spread..."
)


query_rf_mean = np.zeros(
    len(query),
    dtype=np.float32,
)

query_rf_sd = np.zeros(
    len(query),
    dtype=np.float32,
)


for start_idx in range(
    0,
    len(query),
    TREE_BATCH_SIZE,
):

    end_idx = min(
        start_idx + TREE_BATCH_SIZE,
        len(query),
    )

    X_batch = X_query[
        start_idx:end_idx
    ]


    tree_preds = np.stack(
        [
            np.asarray(
                tree.predict(
                    X_batch
                ),
                dtype=np.float32,
            )

            for tree
            in model.estimators_
        ],
        axis=0,
    )


    query_rf_mean[
        start_idx:end_idx
    ] = tree_preds.mean(
        axis=0
    )


    query_rf_sd[
        start_idx:end_idx
    ] = tree_preds.std(
        axis=0,
        ddof=1,
    )


    print(
        f"  RF spread "
        f"{end_idx:,}/{len(query):,}",
        flush=True,
    )


# ============================================================
# Annotate candidates
# ============================================================

query[
    "max_training_tanimoto"
] = query_max_similarity


query[
    "nearest_training_index"
] = query_nn_index


query[
    "nearest_training_pIC50"
] = [
    float(
        train_meta.iloc[j][
            "pIC50"
        ]
    )
    for j in query_nn_index
]


query[
    "nearest_training_smiles"
] = [
    str(
        train_meta.iloc[j][
            "standardized_smiles"
        ]
    )
    for j in query_nn_index
]


if (
    "molecule_chembl_ids"
    in train_meta.columns
):

    query[
        "nearest_training_chembl_ids"
    ] = [
        str(
            train_meta.iloc[j][
                "molecule_chembl_ids"
            ]
        )
        for j in query_nn_index
    ]


query[
    "within_similarity_AD"
] = (
    query[
        "max_training_tanimoto"
    ]
    >= ad_threshold
)


query[
    "rf_tree_mean_pIC50"
] = query_rf_mean


query[
    "rf_tree_sd_pIC50"
] = query_rf_sd


query[
    "rf_tree_spread_warning"
] = (
    query[
        "rf_tree_sd_pIC50"
    ]
    > rf_sd_p95
)


# ============================================================
# Sanity check against script 36 prediction
# ============================================================

query[
    "rf_mean_minus_saved_prediction"
] = (
    query[
        "rf_tree_mean_pIC50"
    ]
    - query[
        "qsar_predicted_pIC50"
    ]
)


max_prediction_difference = float(
    np.max(
        np.abs(
            query[
                "rf_mean_minus_saved_prediction"
            ]
        )
    )
)


if max_prediction_difference > 1e-4:

    print(
        "WARNING: RF ensemble mean differs "
        "from saved model prediction."
    )


# ============================================================
# Transparent priority tiers
#
# No combined weighted score.
# ============================================================

def assign_tier(row):

    if not row[
        "within_similarity_AD"
    ]:

        return "C_OUTSIDE_AD"

    if row[
        "rf_tree_spread_warning"
    ]:

        return "B_WITHIN_AD_HIGH_SPREAD"

    return "A_WITHIN_AD_SUPPORTED"


query[
    "priority_tier"
] = query.apply(
    assign_tier,
    axis=1,
)


# ============================================================
# Sort
#
# Tier is used as support annotation.
# Within Tier A, retain QSAR pIC50 ranking.
# ============================================================

tier_order = {
    "A_WITHIN_AD_SUPPORTED":
        0,

    "B_WITHIN_AD_HIGH_SPREAD":
        1,

    "C_OUTSIDE_AD":
        2,
}


query[
    "_tier_order"
] = query[
    "priority_tier"
].map(
    tier_order
)


query = (
    query
    .sort_values(
        [
            "_tier_order",
            "qsar_predicted_pIC50",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


query[
    "AD_priority_rank"
] = np.arange(
    1,
    len(query) + 1,
)


query = query.drop(
    columns=[
        "_tier_order"
    ]
)


# ============================================================
# Save annotated Top20k
# ============================================================

query.to_csv(
    AD_OUT,
    index=False,
)


# ============================================================
# Production candidate pool
#
# We do NOT combine similarity, RF SD,
# and QSAR into an arbitrary weighted score.
#
# Primary candidate pool:
# Tier A only, ranked by frozen QSAR prediction.
# ============================================================

priority_pool = (
    query[
        query[
            "priority_tier"
        ]
        == "A_WITHIN_AD_SUPPORTED"
    ]
    .sort_values(
        "qsar_predicted_pIC50",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


priority_pool[
    "priority_pool_rank"
] = np.arange(
    1,
    len(priority_pool) + 1,
)


priority1000 = (
    priority_pool
    .head(
        1000
    )
    .copy()
)


priority100 = (
    priority_pool
    .head(
        100
    )
    .copy()
)


priority1000.to_csv(
    PRIORITY1000_OUT,
    index=False,
)


priority100.to_csv(
    PRIORITY100_OUT,
    index=False,
)


# ============================================================
# Summary
# ============================================================

tier_counts = (
    query[
        "priority_tier"
    ]
    .value_counts()
)


external_similarity_quantiles = {
    "min":
        float(
            query[
                "max_training_tanimoto"
            ].min()
        ),

    "p05":
        float(
            query[
                "max_training_tanimoto"
            ].quantile(
                0.05
            )
        ),

    "p25":
        float(
            query[
                "max_training_tanimoto"
            ].quantile(
                0.25
            )
        ),

    "median":
        float(
            query[
                "max_training_tanimoto"
            ].median()
        ),

    "p75":
        float(
            query[
                "max_training_tanimoto"
            ].quantile(
                0.75
            )
        ),

    "p95":
        float(
            query[
                "max_training_tanimoto"
            ].quantile(
                0.95
            )
        ),

    "max":
        float(
            query[
                "max_training_tanimoto"
            ].max()
        ),
}


report = {
    "training_compounds":
        int(
            len(train_fps)
        ),

    "external_candidates_analyzed":
        int(
            len(query)
        ),

    "similarity_metric":
        "Morgan radius2 2048-bit Tanimoto",

    "AD_definition": {
        "method":
            (
                "5th percentile of "
                "training leave-one-out "
                "nearest-neighbor similarity"
            ),

        "percentile":
            AD_PERCENTILE,

        "threshold":
            ad_threshold,

        "training_similarity_quantiles":
            training_similarity_quantiles,
    },

    "RF_tree_spread": {
        "meaning":
            (
                "tree-to-tree prediction "
                "dispersion; not a calibrated "
                "confidence interval"
            ),

        "training_median":
            rf_sd_median,

        "training_p95_warning_threshold":
            rf_sd_p95,
    },

    "external_similarity_quantiles":
        external_similarity_quantiles,

    "tier_counts": {
        str(k):
            int(v)
        for k, v
        in tier_counts.items()
    },

    "priority_pool_size":
        int(
            len(priority_pool)
        ),

    "max_RF_mean_difference":
        max_prediction_difference,

    "outputs": {
        "training_LOO_similarity":
            str(
                TRAINING_NN_OUT
            ),

        "annotated_top20000":
            str(
                AD_OUT
            ),

        "priority_top1000":
            str(
                PRIORITY1000_OUT
            ),

        "priority_top100":
            str(
                PRIORITY100_OUT
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
# Terminal report
# ============================================================

print(
    "\n========================================"
)

print(
    "QSAR AD / UNCERTAINTY RESULTS"
)

print(
    "========================================"
)


print(
    "\nSimilarity AD threshold:",
    f"{ad_threshold:.3f}"
)


print(
    "\nExternal max-training-similarity:"
)

for key, value in (
    external_similarity_quantiles.items()
):

    print(
        f"{key:>6s}: "
        f"{value:.3f}"
    )


print(
    "\nRF tree spread:"
)

print(
    "Training median:",
    f"{rf_sd_median:.3f}"
)

print(
    "Training p95 warning:",
    f"{rf_sd_p95:.3f}"
)


print(
    "\nPriority tiers:"
)

for key, value in (
    tier_counts.items()
):

    print(
        key,
        ":",
        value
    )


print(
    "\nTier-A priority pool:",
    len(priority_pool)
)


print(
    "\nTOP 20 SUPPORTED QSAR CANDIDATES"
)


display_cols = [
    "priority_pool_rank",
    "zinc_id",
    "qsar_predicted_pIC50",
    "max_training_tanimoto",
    "nearest_training_pIC50",
    "rf_tree_sd_pIC50",
    "mwt",
    "logp",
    "standardized_smiles",
]


print(
    priority_pool[
        display_cols
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
    AD_OUT
)

print(
    PRIORITY1000_OUT
)

print(
    PRIORITY100_OUT
)

print(
    REPORT_OUT
)


print(
    "\n========================================"
)

print(
    "PHASE 4B-2 AD ANALYSIS: PASS"
)

print(
    "========================================"
)
