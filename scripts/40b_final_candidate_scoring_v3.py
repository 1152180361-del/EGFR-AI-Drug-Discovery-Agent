from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)


print("=" * 60)
print("PHASE 5A-V3 HINGE PRIORITY FINAL SCORING")
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
        f"Cannot find:\n{input_file}"
    )


df = pd.read_csv(input_file)


print("\nInput candidates:", len(df))



# ==================================================
# Check columns
# ==================================================

required = [

    "phase4C_candidate_id",
    "candidate_track",

    "qsar_predicted_pIC50",

    "max_training_tanimoto",
    "rf_tree_sd_pIC50",

    "reference_contact_jaccard",
    "key_ATP_contact_coverage",

    "MET793_N_hetero_distance_A",
    "MET793_O_hetero_distance_A"

]


missing = [
    x for x in required
    if x not in df.columns
]


if missing:

    raise ValueError(
        "Missing columns:\n"
        +
        "\n".join(missing)
    )



# ==================================================
# Normalization
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
# Evidence features
# ==================================================


# QSAR

df["score_qsar"] = minmax(
    df["qsar_predicted_pIC50"]
)



# Applicability domain

df["score_AD"] = minmax(
    df["max_training_tanimoto"]
)



# Uncertainty
# lower spread better

df["score_uncertainty"] = (

    1 -
    minmax(
        df["rf_tree_sd_pIC50"]
    )

)



# Pose fingerprint

df["score_contact"] = minmax(
    df["reference_contact_jaccard"]
)



# ATP pocket

df["score_ATP"] = minmax(
    df["key_ATP_contact_coverage"]
)



# ==================================================
# MET793 hinge geometry
# ==================================================

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
# Candidate confidence bonus
# ==================================================

# weaker bonus than V2

df["category_bonus"] = np.zeros(
    len(df),
    dtype=float
)



df.loc[

    df["candidate_track"]=="SUPPORTED_A",

    "category_bonus"

] = 0.05



df.loc[

    df["candidate_track"]=="SUPPORTED_B_HIGH_SPREAD",

    "category_bonus"

] = 0.02



# OOD no bonus



# ==================================================
# V3 weighted score
# ==================================================

df["final_score_v3"] = (

    0.25 *
    df["score_qsar"]

    +

    0.15 *
    df["score_AD"]

    +

    0.20 *
    df["score_contact"]

    +

    0.15 *
    df["score_ATP"]

    +

    0.20 *
    df["score_MET793"]

    +

    0.05 *
    df["score_uncertainty"]

    +

    df["category_bonus"]

)



# ==================================================
# Ranking
# ==================================================

ranked = df.sort_values(

    "final_score_v3",

    ascending=False

)



ranked.insert(

    0,

    "final_rank_v3",

    range(
        1,
        len(ranked)+1
    )

)



# ==================================================
# Save
# ==================================================

output = (

    BASE /

    "results/zinc15_screening/"

    "phase5_final_candidate_ranking_v3_hinge_priority.csv"

)



ranked.to_csv(

    output,

    index=False

)



# ==================================================
# Print
# ==================================================

print("\n")
print("="*60)
print("TOP 15 HINGE PRIORITY CANDIDATES")
print("="*60)



print(

ranked[

[

"final_rank_v3",

"phase4C_candidate_id",

"candidate_track",

"qsar_predicted_pIC50",

"reference_contact_jaccard",

"key_ATP_contact_coverage",

"MET793_distance",

"final_score_v3"

]

]

.head(15)

.to_string(index=False)

)



print("\n")
print("="*60)
print("TRACK SUMMARY")
print("="*60)



print(

ranked.groupby(

"candidate_track"

)["final_score_v3"]

.agg(

count="count",

mean="mean",

max="max"

)

)



print("\nSaved:")
print(output)



print("\n")
print("="*60)
print("PHASE 5A-V3 COMPLETE")
print("="*60)
