from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import (
    spearmanr,
    mannwhitneyu,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

INPUT = (
    ROOT
    / "docking/3W32/benchmark/"
      "benchmark_pose_fingerprint_features.csv"
)

OUT_CSV = (
    ROOT
    / "docking/3W32/benchmark/"
      "pose_feature_discrimination_metrics.csv"
)

OUT_JSON = (
    ROOT
    / "docking/3W32/benchmark/"
      "pose_feature_discrimination_metrics.json"
)

FIG_OUT = (
    ROOT
    / "docking/3W32/benchmark/"
      "pose_feature_ROC_AUC_comparison.png"
)


# ============================================================
# Settings
# ============================================================

BOOTSTRAP_N = 2000
SEED = 42


# ============================================================
# Load
# ============================================================

df = pd.read_csv(
    INPUT
)


print(
    "Input poses:",
    len(df)
)


if len(df) != 100:

    print(
        "WARNING: expected 100 benchmark poses."
    )


y = (
    df[
        "binary_label"
    ]
    .astype(int)
    .to_numpy()
)


pic50 = (
    df[
        "pIC50"
    ]
    .astype(float)
    .to_numpy()
)


# ============================================================
# Predefined structural features
#
# direction:
# +1 = larger is expected to indicate active
# -1 = smaller is expected to indicate active
#
# No feature direction is fitted from labels.
# ============================================================

features = {
    "Vina score": {
        "column":
            "top1_vina_score",

        "direction":
            -1,
    },

    "Reference recovery": {
        "column":
            "reference_contact_recovery",

        "direction":
            +1,
    },

    "Contact Jaccard": {
        "column":
            "reference_contact_jaccard",

        "direction":
            +1,
    },

    "Key ATP coverage": {
        "column":
            "key_ATP_contact_coverage",

        "direction":
            +1,
    },

    "MET793 N proximity": {
        "column":
            "MET793_N_hetero_distance_A",

        "direction":
            -1,
    },

    "MET793 O proximity": {
        "column":
            "MET793_O_hetero_distance_A",

        "direction":
            -1,
    },

    "MET793 close count": {
        "column":
            "MET793_close_count",

        "direction":
            +1,
    },
}


# ============================================================
# Bootstrap helper
# ============================================================

rng = np.random.default_rng(
    SEED
)


active_idx = np.where(
    y == 1
)[0]


low_idx = np.where(
    y == 0
)[0]


def bootstrap_auc_ci(
    ranking_score,
):

    values = []

    for _ in range(
        BOOTSTRAP_N
    ):

        a = rng.choice(
            active_idx,
            size=len(active_idx),
            replace=True,
        )

        l = rng.choice(
            low_idx,
            size=len(low_idx),
            replace=True,
        )

        idx = np.concatenate(
            [a, l]
        )

        try:

            value = (
                roc_auc_score(
                    y[idx],
                    ranking_score[idx],
                )
            )

            values.append(
                value
            )

        except Exception:
            pass


    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


# ============================================================
# Evaluate features
# ============================================================

records = []


for feature_name, spec in (
    features.items()
):

    column = spec[
        "column"
    ]

    direction = spec[
        "direction"
    ]


    values = (
        df[
            column
        ]
        .astype(float)
        .to_numpy()
    )


    valid = (
        np.isfinite(
            values
        )
        &
        np.isfinite(
            pic50
        )
    )


    y_v = y[
        valid
    ]

    p_v = pic50[
        valid
    ]

    raw_v = values[
        valid
    ]


    # Convert to:
    # larger = predicted more active
    ranking_score = (
        direction
        * raw_v
    )


    roc_auc = float(
        roc_auc_score(
            y_v,
            ranking_score,
        )
    )


    average_precision = float(
        average_precision_score(
            y_v,
            ranking_score,
        )
    )


    # Bootstrap using full set.
    # Current dataset has no missing values
    # for these metrics, but guard anyway.
    if valid.all():

        ci_low, ci_high = (
            bootstrap_auc_ci(
                ranking_score
            )
        )

    else:

        local_rng = (
            np.random.default_rng(
                SEED
            )
        )

        a_idx = np.where(
            y_v == 1
        )[0]

        l_idx = np.where(
            y_v == 0
        )[0]

        auc_samples = []

        for _ in range(
            BOOTSTRAP_N
        ):

            a = local_rng.choice(
                a_idx,
                size=len(a_idx),
                replace=True,
            )

            l = local_rng.choice(
                l_idx,
                size=len(l_idx),
                replace=True,
            )

            idx = np.concatenate(
                [a, l]
            )

            auc_samples.append(
                roc_auc_score(
                    y_v[idx],
                    ranking_score[idx],
                )
            )


        ci_low = float(
            np.percentile(
                auc_samples,
                2.5,
            )
        )

        ci_high = float(
            np.percentile(
                auc_samples,
                97.5,
            )
        )


    # --------------------------------------------------------
    # Spearman with experimental pIC50
    # --------------------------------------------------------

    rho, rho_p = (
        spearmanr(
            ranking_score,
            p_v,
        )
    )


    # --------------------------------------------------------
    # Group comparison
    # Ranking score larger should favor ACTIVE
    # --------------------------------------------------------

    active_values = (
        ranking_score[
            y_v == 1
        ]
    )

    low_values = (
        ranking_score[
            y_v == 0
        ]
    )


    mw = mannwhitneyu(
        active_values,
        low_values,
        alternative="two-sided",
    )


    # --------------------------------------------------------
    # Original-scale means
    # --------------------------------------------------------

    active_raw = (
        raw_v[
            y_v == 1
        ]
    )

    low_raw = (
        raw_v[
            y_v == 0
        ]
    )


    records.append(
        {
            "feature":
                feature_name,

            "source_column":
                column,

            "expected_direction":
                (
                    "higher_better"
                    if direction == 1
                    else "lower_better"
                ),

            "n":
                int(
                    len(raw_v)
                ),

            "active_mean_raw":
                float(
                    np.mean(
                        active_raw
                    )
                ),

            "low_active_mean_raw":
                float(
                    np.mean(
                        low_raw
                    )
                ),

            "ROC_AUC":
                roc_auc,

            "ROC_AUC_CI_low":
                ci_low,

            "ROC_AUC_CI_high":
                ci_high,

            "Average_Precision":
                average_precision,

            "Spearman_rho":
                float(
                    rho
                ),

            "Spearman_p":
                float(
                    rho_p
                ),

            "Mann_Whitney_U":
                float(
                    mw.statistic
                ),

            "Mann_Whitney_p":
                float(
                    mw.pvalue
                ),
        }
    )


# ============================================================
# Results table
# ============================================================

results = pd.DataFrame(
    records
)


results = (
    results
    .sort_values(
        "ROC_AUC",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


results.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Interpretation
# ============================================================

def interpretation(
    auc_value,
    ci_low,
    ci_high,
):

    if (
        auc_value >= 0.70
        and ci_low > 0.50
    ):

        return (
            "potentially_useful"
        )

    if (
        auc_value >= 0.60
        and ci_low > 0.50
    ):

        return (
            "weak_but_above_random"
        )

    if (
        ci_low <= 0.50
        <= ci_high
    ):

        return (
            "not_distinguishable_from_random"
        )

    if auc_value < 0.50:

        return (
            "opposite_expected_direction"
        )

    return (
        "weak_or_inconclusive"
    )


results[
    "interpretation"
] = [
    interpretation(
        row[
            "ROC_AUC"
        ],
        row[
            "ROC_AUC_CI_low"
        ],
        row[
            "ROC_AUC_CI_high"
        ],
    )

    for _, row
    in results.iterrows()
]


results.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# Figure
# ============================================================

plot_df = (
    results
    .sort_values(
        "ROC_AUC",
        ascending=True,
    )
)


plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.barh(
    plot_df[
        "feature"
    ],
    plot_df[
        "ROC_AUC"
    ],
)


plt.axvline(
    0.5,
    linestyle="--",
)


plt.xlabel(
    "ROC-AUC"
)

plt.ylabel(
    "Structural / docking feature"
)

plt.title(
    "EGFR Retrospective Benchmark: "
    "Single-Feature Discrimination"
)

plt.xlim(
    0.0,
    1.0,
)

plt.tight_layout()

plt.savefig(
    FIG_OUT,
    dpi=300,
)

plt.close()


# ============================================================
# JSON
# ============================================================

report = {
    "n_poses":
        int(
            len(df)
        ),

    "principle":
        (
            "Each predefined feature was "
            "evaluated independently. "
            "No feature weighting, threshold "
            "optimization, or supervised "
            "combination was fitted to the "
            "benchmark labels."
        ),

    "metrics":
        results.to_dict(
            orient="records"
        ),

    "output_csv":
        str(
            OUT_CSV
        ),

    "figure":
        str(
            FIG_OUT
        ),
}


OUT_JSON.write_text(
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
    "EGFR POSE-FEATURE DISCRIMINATION"
)

print(
    "========================================"
)


display_cols = [
    "feature",
    "ROC_AUC",
    "ROC_AUC_CI_low",
    "ROC_AUC_CI_high",
    "Average_Precision",
    "Spearman_rho",
    "Spearman_p",
    "Mann_Whitney_p",
    "interpretation",
]


print(
    "\n"
    + results[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


print(
    "\n----------------------------------------"
)

print(
    "RAW GROUP MEANS"
)

print(
    "----------------------------------------"
)


for _, row in (
    results.iterrows()
):

    print(
        f"{row['feature']:22s} "
        f"ACTIVE="
        f"{row['active_mean_raw']:.3f}  "
        f"LOW="
        f"{row['low_active_mean_raw']:.3f}"
    )


print(
    "\n========================================"
)

print(
    "IMPORTANT:"
)

print(
    "No feature combination was trained "
    "or optimized on these 100 labels."
)

print(
    "========================================"
)


print(
    "\nSaved:"
)

print(
    OUT_CSV
)

print(
    OUT_JSON
)

print(
    FIG_OUT
)
