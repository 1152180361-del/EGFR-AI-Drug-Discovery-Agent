from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)


print("="*60)
print("PHASE 5A-V4 CONSENSUS CANDIDATE RANKING")
print("="*60)



# ==========================================
# Load V2 / V3
# ==========================================


v2_file = (
    BASE /
    "results/zinc15_screening/"
    "phase5_final_candidate_ranking_v2.csv"
)


v3_file = (
    BASE /
    "results/zinc15_screening/"
    "phase5_final_candidate_ranking_v3_hinge_priority.csv"
)



v2 = pd.read_csv(v2_file)

v3 = pd.read_csv(v3_file)



print("\nV2 candidates:", len(v2))
print("V3 candidates:", len(v3))



# ==========================================
# Generate rank
# ==========================================


v2["rank_v2"] = (
    v2["final_score"]
    .rank(
        ascending=False,
        method="min"
    )
)


v3["rank_v3"] = (
    v3["final_score_v3"]
    .rank(
        ascending=False,
        method="min"
    )
)



# ==========================================
# Merge
# ==========================================


cols = [

"phase4C_candidate_id",

"candidate_track",

"qsar_predicted_pIC50",

"reference_contact_jaccard",

"key_ATP_contact_coverage",

"MET793_distance"

]



cons = v2[cols].copy()



cons = cons.merge(

    v2[
        [
        "phase4C_candidate_id",
        "rank_v2",
        "final_score"
        ]
    ],

    on="phase4C_candidate_id"

)



cons = cons.merge(

    v3[
        [
        "phase4C_candidate_id",
        "rank_v3",
        "final_score_v3"
        ]
    ],

    on="phase4C_candidate_id"

)



# ==========================================
# Consensus calculation
# ==========================================


# lower rank better

cons["rank_average"] = (

    cons["rank_v2"]

    +

    cons["rank_v3"]

) / 2



# convert to score

cons["consensus_score"] = (

    1 -

    (

    cons["rank_average"]

    -

    1

    )

    /

    (

    len(cons)-1

    )

)



# stability

cons["ranking_difference"] = abs(

    cons["rank_v2"]

    -

    cons["rank_v3"]

)



# ==========================================
# Final priority
# ==========================================


def label(row):

    if row["ranking_difference"] <= 5:

        return "CONSISTENT_PRIORITY"

    elif row["rank_v3"] < row["rank_v2"]:

        return "STRUCTURE_SUPPORTED"

    else:

        return "QSAR_SUPPORTED"



cons["final_category"] = (

    cons.apply(
        label,
        axis=1
    )

)



# ==========================================
# Sort
# ==========================================


cons = cons.sort_values(

    "consensus_score",

    ascending=False

)



cons.insert(

    0,

    "consensus_rank",

    range(
        1,
        len(cons)+1
    )

)



# ==========================================
# Save
# ==========================================


out = (

BASE /

"results/zinc15_screening/"

"phase5_consensus_final_shortlist.csv"

)



cons.to_csv(

    out,

    index=False

)



# ==========================================
# Print
# ==========================================


print("\n")
print("="*60)
print("TOP 15 CONSENSUS CANDIDATES")
print("="*60)



print(

cons.head(15)

.to_string(index=False)

)



print("\nSaved:")
print(out)



print("\n")
print("="*60)
print("PHASE 5A-V4 COMPLETE")
print("="*60)
