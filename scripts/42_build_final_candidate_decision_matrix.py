from pathlib import Path
import pandas as pd


BASE = Path(
    "/home/xie/EGFR_AIDD_Pipeline"
)


print("="*70)
print("PHASE 5B-1 FINAL CANDIDATE DECISION MATRIX")
print("="*70)



input_file = (

    BASE /
    "results/zinc15_screening/"
    "phase5_consensus_final_shortlist.csv"

)



df = pd.read_csv(input_file)



print("\nCandidates:", len(df))



# ===============================
# Add biological interpretation
# ===============================


def assign_role(row):

    cid = row["phase4C_candidate_id"]


    if cid == "EGFR_CAND_003":

        return "Robust_consensus_candidate"


    elif cid == "EGFR_CAND_011":

        return "QSAR_high_activity_candidate"


    elif cid == "EGFR_CAND_020":

        return "Hinge_geometry_candidate"


    elif cid in [
        "EGFR_CAND_033",
        "EGFR_CAND_034",
        "EGFR_CAND_037"
    ]:

        return "Structure_supported_candidate"


    else:

        return "Backup_candidate"



df["candidate_role"] = df.apply(
    assign_role,
    axis=1
)



# ===============================
# Experimental priority
# ===============================


def priority(row):

    rank = row["consensus_rank"]


    if rank <= 5:

        return "Tier_1_experimental"


    elif rank <= 10:

        return "Tier_2_backup"


    else:

        return "Tier_3_reserve"



df["experimental_priority"] = df.apply(
    priority,
    axis=1
)



# ===============================
# Evidence summary
# ===============================


def evidence(row):

    return (

        f"Consensus rank={int(row['consensus_rank'])}; "

        f"V2 rank={int(row['rank_v2'])}; "

        f"V3 rank={int(row['rank_v3'])}; "

        f"MET793 distance={row['MET793_distance']:.2f} A; "

        f"ATP coverage={row['key_ATP_contact_coverage']:.2f}"

    )



df["evidence_summary"] = df.apply(
    evidence,
    axis=1
)



# ===============================
# Save
# ===============================


output = (

BASE /

"results/zinc15_screening/"

"phase5_final_candidate_decision_matrix.csv"

)



df.to_csv(
    output,
    index=False
)



# ===============================
# Display
# ===============================


print("\n")
print("="*70)
print("TOP EXPERIMENTAL CANDIDATES")
print("="*70)



print(

df[
[
"consensus_rank",
"phase4C_candidate_id",
"candidate_role",
"experimental_priority",
"qsar_predicted_pIC50",
"MET793_distance",
"evidence_summary"
]

]

.head(10)

.to_string(index=False)

)



print("\nSaved:")
print(output)



print("\n")
print("="*70)
print("PHASE 5B-1 COMPLETE")
print("="*70)
