from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)


print("=" * 60)
print("PHASE 5A-V2 FINAL CANDIDATE SCORING")
print("=" * 60)


# ==================================================
# Input
# ==================================================

input_file = (
    BASE /
    "results/zinc15_screening/"
    "phase4E_final_candidate_evidence_matrix.csv"
)


if not input_file.exists():
    raise FileNotFoundError(
        f"Missing input file:\n{input_file}"
    )


df = pd.read_csv(input_file)


print("\nInput candidates:", len(df))


# ==================================================
# Required columns
# ==================================================

required_cols = [

    "phase4C_candidate_id",
    "candidate_track",
    "zinc_id",

    "qsar_predicted_pIC50",
    "max_training_tanimoto",
    "rf_tree_sd_pIC50",

    "reference_contact_jaccard",
    "key_ATP_contact_coverage",

    "MET793_N_hetero_distance_A",
    "MET793_O_hetero_distance_A"

]


missing = [
    c for c in required_cols
    if c not in df.columns
]


if missing:
    raise ValueError(
        "Missing columns:\n" +
        "\n".join(missing)
    )



# ==================================================
# Normalize
# ==================================================

def minmax(x):

    x = x.astype(float)

    if x.max() == x.min():
        return pd.Series(
            np.ones(len(x)),
            index=x.index
        )

    return (
        x-x.min()
    ) / (
        x.max()-x.min()
    )



# ==================================================
# Evidence scores
# ==================================================

# QSAR predicted activity

df["score_qsar"] = minmax(
    df["qsar_predicted_pIC50"]
)



# Applicability domain

df["score_AD"] = minmax(
    df["max_training_tanimoto"]
)



# uncertainty
# lower SD better

df["score_uncertainty"] = (
    1 -
    minmax(
        df["rf_tree_sd_pIC50"]
    )
)



# Structural fingerprint

df["score_contact"] = minmax(
    df["reference_contact_jaccard"]
)



df["score_ATP"] = minmax(
    df["key_ATP_contact_coverage"]
)



# MET793 geometry

df["MET793_distance"] = (

    df["MET793_N_hetero_distance_A"]

    +

    df["MET793_O_hetero_distance_A"]

) / 2



df["score_MET793"] = (

    1 -
    minmax(
        df["MET793_distance"]
    )

)



# ==================================================
# Track confidence
# ==================================================

# IMPORTANT:
# experimental-supported candidates receive
# small confidence bonus


df["category_bonus"] = np.zeros(
    len(df),
    dtype=float
)



df.loc[
    df["candidate_track"]=="SUPPORTED_A",
    "category_bonus"
] = 0.15



df.loc[
    df["candidate_track"]=="SUPPORTED_B_HIGH_SPREAD",
    "category_bonus"
] = 0.05



df.loc[
    df["candidate_track"]=="EXPLORATORY_OOD",
    "category_bonus"
] = 0.0



# ==================================================
# Final weighted score
# ==================================================

df["final_score"] = (

    0.30 *
    df["score_qsar"]

    +

    0.20 *
    df["score_AD"]

    +

    0.20 *
    df["score_contact"]

    +

    0.15 *
    df["score_ATP"]

    +

    0.10 *
    df["score_MET793"]

    +

    0.05 *
    df["score_uncertainty"]

    +

    df["category_bonus"]

)



# ==================================================
# Rank
# ==================================================

ranked = df.sort_values(
    "final_score",
    ascending=False
)



ranked.insert(
    0,
    "final_rank",
    range(1, len(ranked)+1)
)



# ==================================================
# Save
# ==================================================

output = (

    BASE /

    "results/zinc15_screening/"

    "phase5_final_candidate_ranking_v2.csv"

)



ranked.to_csv(
    output,
    index=False
)



# ==================================================
# Display
# ==================================================

print("\n")
print("="*60)
print("TOP 15 FINAL CANDIDATES")
print("="*60)


print(

ranked[

[
"final_rank",

"phase4C_candidate_id",

"candidate_track",

"qsar_predicted_pIC50",

"max_training_tanimoto",

"reference_contact_jaccard",

"key_ATP_contact_coverage",

"MET793_distance",

"final_score"

]

]

.head(15)

.to_string(index=False)

)



print("\n")
print("="*60)
print("CATEGORY SUMMARY")
print("="*60)


print(

ranked.groupby(
    "candidate_track"
)["final_score"]

.agg(
    count="count",
    mean="mean",
    max="max"
)

)



print("\n")
print("Saved:")
print(output)



print("\n")
print("="*60)
print("PHASE 5A-V2 COMPLETE")
print("="*60)
