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
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    auc,
)

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

BASE = (
    ROOT
    / "docking/3W32/benchmark"
)

INPUT = (
    BASE
    / "benchmark_docking_results.csv"
)

METRICS_OUT = (
    BASE
    / "retrospective_screening_metrics.json"
)

RANKED_OUT = (
    BASE
    / "retrospective_screening_ranked.csv"
)

ROC_FIG = (
    BASE
    / "retrospective_ROC_curve.png"
)

PR_FIG = (
    BASE
    / "retrospective_PR_curve.png"
)

DIST_FIG = (
    BASE
    / "retrospective_score_distribution.png"
)

SCATTER_FIG = (
    BASE
    / "retrospective_score_vs_pIC50.png"
)


# ============================================================
# Settings
# ============================================================

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42

EF_LEVELS = [
    0.01,
    0.05,
    0.10,
]


# ============================================================
# Load
# ============================================================

df = pd.read_csv(
    INPUT
)


valid_status = [
    "PASS",
    "PASS_EXISTING",
]


df = (
    df[
        df["status"].isin(
            valid_status
        )
    ]
    .copy()
)


required = [
    "benchmark_id",
    "benchmark_label",
    "binary_label",
    "pIC50",
    "top1_vina_score",
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


df = df.dropna(
    subset=[
        "binary_label",
        "pIC50",
        "top1_vina_score",
    ]
).copy()


print(
    "Valid docked molecules:",
    len(df)
)


# ============================================================
# Ranking score
#
# More negative Vina score = better.
# Convert to a conventional:
#
# larger score = more predicted active
# ============================================================

df[
    "vina_ranking_score"
] = (
    -df[
        "top1_vina_score"
    ]
)


y_true = (
    df[
        "binary_label"
    ]
    .astype(int)
    .to_numpy()
)


y_score = (
    df[
        "vina_ranking_score"
    ]
    .to_numpy(
        dtype=float
    )
)


pic50 = (
    df[
        "pIC50"
    ]
    .to_numpy(
        dtype=float
    )
)


# ============================================================
# Class counts
# ============================================================

n_total = len(df)

n_active = int(
    y_true.sum()
)

n_inactive = (
    n_total
    - n_active
)

active_fraction = (
    n_active
    / n_total
)


print(
    "Actives:",
    n_active
)

print(
    "Low-actives:",
    n_inactive
)

print(
    "Active fraction:",
    round(
        active_fraction,
        4,
    )
)


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = float(
    roc_auc_score(
        y_true,
        y_score,
    )
)


fpr, tpr, roc_thresholds = (
    roc_curve(
        y_true,
        y_score,
    )
)


# ============================================================
# PR-AUC / Average Precision
# ============================================================

average_precision = float(
    average_precision_score(
        y_true,
        y_score,
    )
)


precision, recall, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_score,
    )
)


# sklearn recall may be decreasing;
# reverse for trapezoid integration.
pr_auc = float(
    auc(
        recall[::-1],
        precision[::-1],
    )
)


# ============================================================
# Spearman
#
# ranking score larger = stronger predicted binding
# pIC50 larger = stronger experimental activity
# ============================================================

spearman_rho, spearman_p = (
    spearmanr(
        y_score,
        pic50,
    )
)


spearman_rho = float(
    spearman_rho
)

spearman_p = float(
    spearman_p
)


# ============================================================
# Mann-Whitney U
#
# Tests whether ranking_score differs
# between ACTIVE and LOW_ACTIVE.
# ============================================================

active_scores = (
    df.loc[
        df[
            "binary_label"
        ] == 1,
        "vina_ranking_score",
    ]
    .to_numpy(
        dtype=float
    )
)


low_scores = (
    df.loc[
        df[
            "binary_label"
        ] == 0,
        "vina_ranking_score",
    ]
    .to_numpy(
        dtype=float
    )
)


mw = mannwhitneyu(
    active_scores,
    low_scores,
    alternative="two-sided",
)


mw_u = float(
    mw.statistic
)

mw_p = float(
    mw.pvalue
)


# ============================================================
# Rank molecules
# ============================================================

ranked = (
    df
    .sort_values(
        "vina_ranking_score",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


ranked[
    "rank"
] = (
    np.arange(
        1,
        len(ranked) + 1,
    )
)


ranked[
    "cumulative_actives"
] = (
    ranked[
        "binary_label"
    ]
    .cumsum()
)


ranked[
    "cumulative_active_fraction"
] = (
    ranked[
        "cumulative_actives"
    ]
    / ranked[
        "rank"
    ]
)


# ============================================================
# Enrichment factor
#
# EF(x%) =
#
# observed active fraction in top x%
# ----------------------------------
# overall active fraction
# ============================================================

ef_results = {}


for fraction in EF_LEVELS:

    n_top = max(
        1,
        int(
            np.ceil(
                n_total
                * fraction
            )
        ),
    )


    top = (
        ranked.head(
            n_top
        )
    )


    hits = int(
        top[
            "binary_label"
        ].sum()
    )


    hit_rate = (
        hits
        / n_top
    )


    ef = (
        hit_rate
        / active_fraction
    )


    key = (
        f"EF{int(fraction * 100)}%"
    )


    ef_results[
        key
    ] = {
        "fraction":
            fraction,

        "n_top":
            int(
                n_top
            ),

        "active_hits":
            hits,

        "hit_rate":
            float(
                hit_rate
            ),

        "enrichment_factor":
            float(
                ef
            ),
    }


# ============================================================
# Top-N diagnostics
# ============================================================

topn_results = {}


for n in [
    1,
    5,
    10,
    20,
]:

    n = min(
        n,
        n_total,
    )


    top = ranked.head(
        n
    )


    hits = int(
        top[
            "binary_label"
        ].sum()
    )


    topn_results[
        f"top_{n}"
    ] = {
        "active_hits":
            hits,

        "low_active_hits":
            int(
                n - hits
            ),

        "active_fraction":
            float(
                hits / n
            ),
    }


# ============================================================
# Bootstrap confidence intervals
#
# Stratified bootstrap keeps 50/50
# class balance.
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


active_indices = np.where(
    y_true == 1
)[0]


low_indices = np.where(
    y_true == 0
)[0]


bootstrap_auc = []
bootstrap_ap = []
bootstrap_spearman = []


for _ in range(
    BOOTSTRAP_N
):

    sampled_active = (
        rng.choice(
            active_indices,
            size=len(
                active_indices
            ),
            replace=True,
        )
    )


    sampled_low = (
        rng.choice(
            low_indices,
            size=len(
                low_indices
            ),
            replace=True,
        )
    )


    idx = np.concatenate(
        [
            sampled_active,
            sampled_low,
        ]
    )


    y_b = y_true[idx]
    s_b = y_score[idx]
    p_b = pic50[idx]


    try:

        bootstrap_auc.append(
            roc_auc_score(
                y_b,
                s_b,
            )
        )


        bootstrap_ap.append(
            average_precision_score(
                y_b,
                s_b,
            )
        )


        rho_b, _ = (
            spearmanr(
                s_b,
                p_b,
            )
        )


        if np.isfinite(
            rho_b
        ):

            bootstrap_spearman.append(
                rho_b
            )


    except Exception:

        continue


def ci95(
    values
):

    values = np.asarray(
        values,
        dtype=float,
    )

    return [
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
    ]


roc_auc_ci = ci95(
    bootstrap_auc
)


ap_ci = ci95(
    bootstrap_ap
)


spearman_ci = ci95(
    bootstrap_spearman
)


# ============================================================
# Class score summary
# ============================================================

active_vina = (
    df.loc[
        df[
            "binary_label"
        ] == 1,
        "top1_vina_score",
    ]
)


low_vina = (
    df.loc[
        df[
            "binary_label"
        ] == 0,
        "top1_vina_score",
    ]
)


score_summary = {
    "ACTIVE": {
        "mean":
            float(
                active_vina.mean()
            ),

        "median":
            float(
                active_vina.median()
            ),

        "std":
            float(
                active_vina.std()
            ),

        "min":
            float(
                active_vina.min()
            ),

        "max":
            float(
                active_vina.max()
            ),
    },

    "LOW_ACTIVE": {
        "mean":
            float(
                low_vina.mean()
            ),

        "median":
            float(
                low_vina.median()
            ),

        "std":
            float(
                low_vina.std()
            ),

        "min":
            float(
                low_vina.min()
            ),

        "max":
            float(
                low_vina.max()
            ),
    },
}


# ============================================================
# Figures
# ============================================================

# ROC
plt.figure(
    figsize=(
        6,
        6,
    )
)

plt.plot(
    fpr,
    tpr,
    label=(
        f"Vina ROC-AUC = "
        f"{roc_auc:.3f}"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random = 0.500",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "EGFR Retrospective Docking ROC"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ROC_FIG,
    dpi=300,
)

plt.close()


# PR
plt.figure(
    figsize=(
        6,
        6,
    )
)

plt.plot(
    recall,
    precision,
    label=(
        f"AP = "
        f"{average_precision:.3f}"
    ),
)

plt.axhline(
    active_fraction,
    linestyle="--",
    label=(
        f"Random baseline = "
        f"{active_fraction:.3f}"
    ),
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "EGFR Retrospective Docking PR"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PR_FIG,
    dpi=300,
)

plt.close()


# Score distribution
plt.figure(
    figsize=(
        7,
        5,
    )
)


plt.hist(
    active_vina,
    bins=15,
    alpha=0.55,
    label="ACTIVE",
)


plt.hist(
    low_vina,
    bins=15,
    alpha=0.55,
    label="LOW_ACTIVE",
)


plt.xlabel(
    "Top-1 Vina score (kcal/mol)"
)

plt.ylabel(
    "Count"
)

plt.title(
    "EGFR Docking Score Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    DIST_FIG,
    dpi=300,
)

plt.close()


# Score vs pIC50
plt.figure(
    figsize=(
        7,
        5,
    )
)


plt.scatter(
    df[
        "top1_vina_score"
    ],
    df[
        "pIC50"
    ],
    alpha=0.75,
)


plt.xlabel(
    "Top-1 Vina score (kcal/mol)"
)

plt.ylabel(
    "Experimental pIC50"
)

plt.title(
    "Docking Score vs Experimental Activity"
)

plt.tight_layout()

plt.savefig(
    SCATTER_FIG,
    dpi=300,
)

plt.close()


# ============================================================
# Save ranked table
# ============================================================

ranked.to_csv(
    RANKED_OUT,
    index=False,
)


# ============================================================
# Interpretation
# ============================================================

if roc_auc >= 0.70:

    discrimination = (
        "USEFUL"
    )

elif roc_auc >= 0.60:

    discrimination = (
        "WEAK_TO_MODERATE"
    )

elif roc_auc >= 0.55:

    discrimination = (
        "WEAK"
    )

elif roc_auc >= 0.45:

    discrimination = (
        "NEAR_RANDOM"
    )

else:

    discrimination = (
        "INVERTED_OR_POOR"
    )


# ============================================================
# Report
# ============================================================

report = {
    "n_total":
        int(
            n_total
        ),

    "n_active":
        int(
            n_active
        ),

    "n_low_active":
        int(
            n_inactive
        ),

    "active_fraction":
        float(
            active_fraction
        ),

    "ROC_AUC":
        roc_auc,

    "ROC_AUC_bootstrap_95CI":
        roc_auc_ci,

    "Average_Precision":
        average_precision,

    "Average_Precision_bootstrap_95CI":
        ap_ci,

    "PR_AUC":
        pr_auc,

    "Spearman_rho":
        spearman_rho,

    "Spearman_p":
        spearman_p,

    "Spearman_bootstrap_95CI":
        spearman_ci,

    "Mann_Whitney_U":
        mw_u,

    "Mann_Whitney_p":
        mw_p,

    "enrichment_factors":
        ef_results,

    "topN":
        topn_results,

    "score_summary":
        score_summary,

    "screening_discrimination":
        discrimination,

    "ranked_csv":
        str(
            RANKED_OUT
        ),

    "figures": {
        "ROC":
            str(
                ROC_FIG
            ),

        "PR":
            str(
                PR_FIG
            ),

        "distribution":
            str(
                DIST_FIG
            ),

        "score_vs_pIC50":
            str(
                SCATTER_FIG
            ),
    },
}


METRICS_OUT.write_text(
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
    "EGFR RETROSPECTIVE SCREENING PERFORMANCE"
)

print(
    "========================================"
)


print(
    "\nDataset:"
)

print(
    "Total:",
    n_total
)

print(
    "ACTIVE:",
    n_active
)

print(
    "LOW_ACTIVE:",
    n_inactive
)


print(
    "\n----------------------------------------"
)

print(
    "CLASSIFICATION / RANKING"
)

print(
    "----------------------------------------"
)


print(
    "ROC-AUC:",
    f"{roc_auc:.3f}"
)

print(
    "ROC-AUC 95% CI:",
    f"[{roc_auc_ci[0]:.3f}, "
    f"{roc_auc_ci[1]:.3f}]"
)


print(
    "Average Precision:",
    f"{average_precision:.3f}"
)

print(
    "AP 95% CI:",
    f"[{ap_ci[0]:.3f}, "
    f"{ap_ci[1]:.3f}]"
)


print(
    "PR-AUC:",
    f"{pr_auc:.3f}"
)


print(
    "\n----------------------------------------"
)

print(
    "ACTIVITY CORRELATION"
)

print(
    "----------------------------------------"
)


print(
    "Spearman rho:",
    f"{spearman_rho:.3f}"
)

print(
    "Spearman p:",
    f"{spearman_p:.4g}"
)

print(
    "Spearman 95% CI:",
    f"[{spearman_ci[0]:.3f}, "
    f"{spearman_ci[1]:.3f}]"
)


print(
    "\nMann-Whitney U:",
    f"{mw_u:.1f}"
)

print(
    "Mann-Whitney p:",
    f"{mw_p:.4g}"
)


print(
    "\n----------------------------------------"
)

print(
    "SCORE DISTRIBUTIONS"
)

print(
    "----------------------------------------"
)


for group in [
    "ACTIVE",
    "LOW_ACTIVE",
]:

    s = (
        score_summary[
            group
        ]
    )

    print(
        f"{group:10s} "
        f"mean={s['mean']:.3f} "
        f"median={s['median']:.3f} "
        f"std={s['std']:.3f}"
    )


print(
    "\n----------------------------------------"
)

print(
    "EARLY ENRICHMENT"
)

print(
    "----------------------------------------"
)


for key, values in (
    ef_results.items()
):

    print(
        f"{key:6s}: "
        f"top {values['n_top']} | "
        f"active hits="
        f"{values['active_hits']} | "
        f"EF="
        f"{values['enrichment_factor']:.3f}"
    )


print(
    "\nTop-N composition:"
)


for key, values in (
    topn_results.items()
):

    print(
        f"{key:7s}: "
        f"ACTIVE="
        f"{values['active_hits']} | "
        f"LOW_ACTIVE="
        f"{values['low_active_hits']}"
    )


print(
    "\n========================================"
)

print(
    "SCREENING DISCRIMINATION:",
    discrimination
)

print(
    "========================================"
)


print(
    "\nSaved metrics:"
)

print(
    METRICS_OUT
)

print(
    "\nSaved ranked table:"
)

print(
    RANKED_OUT
)

print(
    "\nFigures:"
)

print(
    ROC_FIG
)

print(
    PR_FIG
)

print(
    DIST_FIG
)

print(
    SCATTER_FIG
)
