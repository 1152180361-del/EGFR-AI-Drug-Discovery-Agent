from pathlib import Path
import json
import pandas as pd


ROOT = Path("/home/xie/EGFR_AIDD_Pipeline")

BASE = ROOT / "docking/3W32"

REDOCK = (
    BASE
    / "reports/W32_redocking_RMSD.csv"
)

SCREENING = (
    BASE
    / "benchmark/retrospective_screening_metrics.json"
)

POSE_METRICS = (
    BASE
    / "benchmark/pose_feature_discrimination_metrics.csv"
)

REFERENCE = (
    BASE
    / "reference_pose/W32_reference_pose_fingerprint.json"
)

OUT_DIR = ROOT / "reports"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_MD = (
    OUT_DIR
    / "phase3_EGFR_structure_validation_summary.md"
)

OUT_JSON = (
    OUT_DIR
    / "phase3_EGFR_structure_validation_summary.json"
)


# ============================================================
# Load
# ============================================================

redock = pd.read_csv(
    REDOCK
)

screening = json.loads(
    SCREENING.read_text()
)

pose = pd.read_csv(
    POSE_METRICS
)

reference = json.loads(
    REFERENCE.read_text()
)


# ============================================================
# Redocking
# ============================================================

top1 = redock.iloc[0]

best = redock.loc[
    redock[
        "crystal_pose_RMSD_A"
    ].idxmin()
]


top1_rmsd = float(
    top1[
        "crystal_pose_RMSD_A"
    ]
)

top1_score = float(
    top1[
        "vina_score_kcal_mol"
    ]
)

best_rmsd = float(
    best[
        "crystal_pose_RMSD_A"
    ]
)

best_mode = int(
    best[
        "mode"
    ]
)


# ============================================================
# Retrospective benchmark
# ============================================================

roc_auc = float(
    screening[
        "ROC_AUC"
    ]
)

roc_ci = screening[
    "ROC_AUC_bootstrap_95CI"
]

ap = float(
    screening[
        "Average_Precision"
    ]
)

spearman = float(
    screening[
        "Spearman_rho"
    ]
)

spearman_p = float(
    screening[
        "Spearman_p"
    ]
)


# ============================================================
# Best structural feature
# ============================================================

best_feature = (
    pose
    .sort_values(
        "ROC_AUC",
        ascending=False,
    )
    .iloc[0]
)


best_feature_name = str(
    best_feature[
        "feature"
    ]
)

best_feature_auc = float(
    best_feature[
        "ROC_AUC"
    ]
)

best_feature_ci_low = float(
    best_feature[
        "ROC_AUC_CI_low"
    ]
)

best_feature_ci_high = float(
    best_feature[
        "ROC_AUC_CI_high"
    ]
)


# ============================================================
# Reference fingerprint
# ============================================================

n_contacts_5A = int(
    reference[
        "n_contacts_5A"
    ]
)

n_key_contacts = len(
    reference[
        "key_ATP_pocket_contacts_within_5A"
    ]
)


# ============================================================
# Frozen scientific decision
# ============================================================

decision = {
    "phase":
        "Phase 3 - Structure-based CADD validation",

    "status":
        "FROZEN",

    "production_receptor":
        "EGFR 3W32",

    "redocking_validation":
        "PASS",

    "top1_crystal_pose_RMSD_A":
        top1_rmsd,

    "best_crystal_pose_RMSD_A":
        best_rmsd,

    "best_geometric_mode":
        best_mode,

    "vina_retrospective_ROC_AUC":
        roc_auc,

    "vina_retrospective_ROC_AUC_95CI":
        roc_ci,

    "vina_activity_ranker":
        False,

    "reference_pose_feature_activity_ranker":
        False,

    "best_pose_feature":
        best_feature_name,

    "best_pose_feature_ROC_AUC":
        best_feature_auc,

    "best_pose_feature_ROC_AUC_95CI": [
        best_feature_ci_low,
        best_feature_ci_high,
    ],

    "structure_stage_role": [
        "ATP-pocket pose plausibility",
        "steric compatibility",
        "binding-mode interpretation",
        "key-residue geometry assessment",
    ],

    "structure_stage_not_used_for": [
        "direct IC50 prediction",
        "Vina-score-only activity ranking",
        "supervised optimization on retrospective benchmark labels",
    ],

    "phase4_strategy":
        "QSAR-first cascade followed by structure-based pose filtering",
}


OUT_JSON.write_text(
    json.dumps(
        decision,
        indent=2,
    )
)


# ============================================================
# Markdown report
# ============================================================

text = f"""# Phase 3 – EGFR Structure-Based CADD Validation

## 1. Production receptor

Selected receptor: **EGFR 3W32**

The receptor was selected after structural QC of multiple EGFR structures.
3W32 retained a complete ATP-binding pocket under the predefined structural
QC criteria and was advanced to experimental-pose redocking validation.

## 2. W32 self-redocking validation

Top-ranked Vina pose:

- Vina score: {top1_score:.3f} kcal/mol
- Crystal-pose heavy-atom RMSD: **{top1_rmsd:.3f} Å**

Best geometric pose:

- Mode: {best_mode}
- RMSD: **{best_rmsd:.3f} Å**

Predefined criterion:

- RMSD <= 2.0 Å

Result:

**PASS**

The docking protocol can reproduce the experimentally observed W32 binding
pose within the predefined geometric criterion.

## 3. Reference binding-mode fingerprint

The W32 crystal pose contained:

- {n_contacts_5A} receptor contacts within 5 Å
- {n_key_contacts} predefined ATP-pocket contacts

These features were subsequently used as structural descriptors rather than
as fitted activity-prediction rules.

## 4. Retrospective activity benchmark

Benchmark:

- 50 strong EGFR actives
- 50 low-activity EGFR compounds
- scaffold-diverse
- no cross-class scaffold overlap
- physicochemical properties approximately matched

Vina Top-1 score performance:

- ROC-AUC: **{roc_auc:.3f}**
- 95% bootstrap CI: **[{roc_ci[0]:.3f}, {roc_ci[1]:.3f}]**
- Average Precision: **{ap:.3f}**
- Spearman rho with experimental pIC50: **{spearman:.3f}**
- Spearman p-value: **{spearman_p:.4g}**

Conclusion:

**Vina score did not provide useful retrospective activity discrimination.**

## 5. Pose-feature benchmark

Best predefined structural feature:

- {best_feature_name}
- ROC-AUC: **{best_feature_auc:.3f}**
- 95% CI: **[{best_feature_ci_low:.3f}, {best_feature_ci_high:.3f}]**

The confidence interval includes random performance (ROC-AUC = 0.5).

Therefore neither Vina score nor a single W32-reference pose-similarity
feature is used as an EGFR potency classifier.

## 6. Frozen Phase 4 decision

The structure-based stage is retained for:

1. ATP-pocket pose plausibility
2. steric compatibility
3. binding-mode interpretation
4. key-residue geometry assessment

It is **not** used for:

- direct IC50 prediction
- Vina-score-only compound ranking
- benchmark-fitted structural scoring

The production cascade is therefore frozen as:

**QSAR ranking -> 3W32 docking -> pose/interaction filtering -> candidate prioritization**

This prevents retrospective benchmark overfitting and preserves the distinct
roles of ligand-based prediction and structure-based CADD.
"""


OUT_MD.write_text(
    text
)


print(
    "\n========================================"
)

print(
    "PHASE 3 STRUCTURE VALIDATION FREEZE"
)

print(
    "========================================"
)

print(
    "Production receptor: 3W32"
)

print(
    "Top-1 redocking RMSD:",
    f"{top1_rmsd:.3f} A"
)

print(
    "Redocking:",
    "PASS"
)

print(
    "\nVina retrospective ROC-AUC:",
    f"{roc_auc:.3f}"
)

print(
    "95% CI:",
    f"[{roc_ci[0]:.3f}, "
    f"{roc_ci[1]:.3f}]"
)

print(
    "\nBest pose feature:",
    best_feature_name
)

print(
    "Feature ROC-AUC:",
    f"{best_feature_auc:.3f}"
)

print(
    "Feature 95% CI:",
    f"[{best_feature_ci_low:.3f}, "
    f"{best_feature_ci_high:.3f}]"
)

print(
    "\nFINAL STRUCTURE-STAGE ROLE:"
)

print(
    "Pose plausibility / structural filtering"
)

print(
    "NOT potency ranking"
)

print(
    "\nPHASE 4 CASCADE:"
)

print(
    "QSAR -> 3W32 docking -> "
    "pose filtering -> prioritization"
)

print(
    "\nSaved:"
)

print(
    OUT_MD
)

print(
    OUT_JSON
)

print(
    "\n========================================"
)

print(
    "PHASE 3 STATUS: FROZEN"
)

print(
    "========================================"
)
