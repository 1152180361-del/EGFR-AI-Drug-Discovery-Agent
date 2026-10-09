from pathlib import Path
import pandas as pd
from datetime import datetime


BASE = Path("/home/xie/EGFR_AIDD_Pipeline")


input_file = (
    BASE /
    "results/zinc15_screening/"
    "phase5_final_candidate_decision_matrix.csv"
)


output_file = (
    BASE /
    "reports/"
    "EGFR_AI_agent_discovery_report.md"
)


df = pd.read_csv(input_file)


top = df.head(10)


report = f"""
# EGFR AI-assisted Drug Discovery Workflow Report


Generated:
{datetime.now().strftime("%Y-%m-%d")}



# 1. Project Overview


This project established an AI-assisted EGFR inhibitor discovery workflow integrating:


- QSAR activity prediction
- Applicability domain analysis
- Scaffold diversity selection
- Molecular docking
- Pose fingerprint analysis
- Consensus candidate prioritization



The workflow represents a multi-agent decision framework where independent computational modules provide complementary evidence.



# 2. Computational Workflow


ZINC15 Library

↓

Chemical Quality Control

↓

QSAR Prediction

↓

Applicability Domain Filtering

↓

Scaffold Diversity Selection

↓

Molecular Docking

↓

Pose Fingerprint Analysis

↓

Consensus Ranking Agent

↓

Experimental Candidate Recommendation




# 3. Screening Statistics


Initial ZINC15 molecules:

250000


After chemical QC:

221988


Final AI prioritized candidates:

47




# 4. Final Recommended Candidates


"""


for _, row in top.iterrows():

    report += f"""

## {row['phase4C_candidate_id']}


Candidate role:

{row['candidate_role']}


Experimental priority:

{row['experimental_priority']}


Predicted activity:

pIC50 = {row['qsar_predicted_pIC50']:.3f}



Structural evidence:


MET793 distance:

{row['MET793_distance']:.2f} Angstrom



ATP pocket coverage:

{row['key_ATP_contact_coverage']:.3f}



Ranking evidence:


{row['evidence_summary']}



----------------------------

"""


report += """

# 5. AI Decision Logic


Final candidates were selected using multiple independent evidence sources:


1. QSAR predicted activity

2. Chemical applicability domain

3. Docking pose quality

4. ATP pocket interaction pattern

5. Hinge region geometry

6. Consensus ranking stability



No single metric was used as the final decision criterion.




# 6. Experimental Priority Recommendation



Tier 1 candidates:


1. EGFR_CAND_003

Robust consensus candidate.



2. EGFR_CAND_011

High QSAR activity candidate.



3. EGFR_CAND_020

Hinge geometry supported candidate.



4. EGFR_CAND_033

Structure supported candidate.



5. EGFR_CAND_001

QSAR supported backup candidate.





# 7. Future AI Agent Extension



The current workflow is implemented as Python orchestration.

Future development can convert each module into callable tools:


User Request

↓

LLM Planning Agent

↓

Tool Calling Layer


QSAR Tool

Docking Tool

Structure Analysis Tool


↓

Decision Agent

↓

Experiment Recommendation




"""


output_file.parent.mkdir(
    exist_ok=True
)


output_file.write_text(
    report,
    encoding="utf-8"
)



print("="*70)
print("AI AGENT DISCOVERY REPORT GENERATED")
print("="*70)

print(output_file)
