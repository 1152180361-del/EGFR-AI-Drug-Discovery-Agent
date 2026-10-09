from datetime import datetime, timezone

import pandas as pd

from .risk_engine import EGFRAgentRiskEngine
from .pose_review_evidence import render_pose_review


class EGFRReportGenerator:

    @staticmethod
    def fmt(row, column, digits=3):

        if column not in row.index:
            return "N/A"

        value = row[column]

        if pd.isna(value):
            return "N/A"

        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def find_candidate(df, candidate_id):

        result = df.loc[
            df["phase4C_candidate_id"] == candidate_id
        ]

        if len(result) != 1:
            raise ValueError(
                f"Candidate evidence missing or duplicated: "
                f"{candidate_id}"
            )

        return result.iloc[0]

    def generate(
        self,
        ranking_df,
        structure_df=None,
        risk_result=None,
        comparison_result=None,
    ):

        if ranking_df is None or ranking_df.empty:
            raise ValueError(
                "Cannot generate report with empty ranking."
            )

        if structure_df is None or structure_df.empty:
            raise ValueError(
                "Cannot generate report without candidate evidence."
            )

        best = ranking_df.iloc[0]
        candidate_id = str(best["candidate"])

        candidate = self.find_candidate(
            structure_df, candidate_id
        )

        if risk_result is None:
            risk_result = EGFRAgentRiskEngine().evaluate(
                candidate
            )

        lines = [
            "# EGFR Computational Candidate Review",
            "",
            "Generated (UTC): "
            + datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "",
            "## 1. Primary candidate",
            "",
            f"**Candidate:** {candidate_id}",
            "",
            f"**Heuristic decision score:** {best['score']}",
            "",
            f"**Decision reason:** {best['reason']}",
            "",
            "**Computational evidence category:** "
            + risk_result["confidence"],
            "",
            "This category describes rule-based evidence "
            "support, not calibrated predictive confidence "
            "or experimentally verified potency.",
            "",
            "## 2. Candidate evidence",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]

        evidence_metrics = [
            ("Candidate track", "candidate_track", None),
            ("Consensus rank", "consensus_rank", 0),
            ("QSAR predicted pIC50", "qsar_predicted_pIC50", 3),
            ("Max training Tanimoto", "max_training_tanimoto", 3),
            ("RF tree SD (pIC50)", "rf_tree_sd_pIC50", 3),
            ("ATP contact coverage", "key_ATP_contact_coverage", 3),
            ("MET793 mean proximity (A)", "MET793_distance", 2),
            ("Vina top-pose score", "top1_vina_score", 2),
        ]

        for label, column, digits in evidence_metrics:

            if digits is None:
                value = str(
                    candidate.get(column, "N/A")
                )
            else:
                value = self.fmt(
                    candidate, column, digits
                )

            lines.append(f"| {label} | {value} |")

        lines.extend([
            "",
            "## 3. Supporting evidence",
            "",
        ])

        for item in risk_result.get(
            "supporting_evidence", []
        ):
            lines.append(f"- {item}")

        lines.extend([
            "",
            "## 4. Risks and limitations",
            "",
        ])

        for item in risk_result.get(
            "risk_summary", []
        ):
            lines.append(f"- {item}")

        lines.extend([
            "",
            "## 5. Alternative candidates",
            "",
            "The alternatives are compared using existing "
            "metrics, not a validated superiority test.",
            "",
            "| Candidate | Evidence category | "
            "QSAR pIC50 | Tanimoto | RF SD | "
            "ATP coverage | MET793 proximity (A) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])

        risk_engine = EGFRAgentRiskEngine()

        if comparison_result is None:
            alternative_ids = [
                str(row["candidate"])
                for _, row in ranking_df.iloc[1:4].iterrows()
            ]
        else:
            alternative_ids = [
                str(item["candidate"])
                for item in comparison_result
            ]

        for alternative_id in alternative_ids:

            row = self.find_candidate(
                structure_df, alternative_id
            )

            alt_risk = risk_engine.evaluate(row)

            lines.append(
                "| "
                + alternative_id
                + " | "
                + alt_risk["confidence"]
                + " | "
                + self.fmt(
                    row, "qsar_predicted_pIC50"
                )
                + " | "
                + self.fmt(
                    row, "max_training_tanimoto"
                )
                + " | "
                + self.fmt(
                    row, "rf_tree_sd_pIC50"
                )
                + " | "
                + self.fmt(
                    row, "key_ATP_contact_coverage"
                )
                + " | "
                + self.fmt(
                    row, "MET793_distance", 2
                )
                + " |"
            )

        lines.extend([
            "",
            "## 6. Evidence provenance",
            "",
            "- QSAR candidate-level metrics come from the "
            "existing Phase 5 candidate table and Phase 4E "
            "evidence matrix.",
            "- The standalone QSAR tool reads an existing "
            "global ZINC prediction table; its top entries "
            "are not automatically matched to the selected "
            "EGFR_CAND identifiers.",
            "- The structure tool reads previously computed "
            "results; it does not run new docking.",
            "- MET793 distance is a proximity descriptor. "
            "Hydrogen-bond geometry and donor/acceptor "
            "compatibility require separate validation.",
            "- The consensus combines related heuristic "
            "ranking approaches, not independent "
            "experimental evidence.",
            "- Prior retrospective docking discrimination "
            "was weak; docking should be treated as "
            "structural plausibility evidence rather than "
            "an experimentally validated potency predictor.",
            "- Final compound selection requires further "
            "pose review, chemistry assessment and "
            "experimental validation.",
            "",
            "## 7. Workflow classification",
            "",
            "This workflow uses an Ollama LLM for planning "
            "and deterministic Python modules for "
            "reading precomputed evidence, prioritizing "
            "candidates, assessing risks and generating "
            "a review report.",
            "",
            "It is not an autonomous molecular-design "
            "system or a new docking/QSAR calculation "
            "for each request.",
            "",
        ])

        # Independent, non-scoring manual pose review evidence.
        # Report only candidates present in this ranking.
        review_ids = (
            ranking_df["candidate"]
            .head(10)
            .astype(str)
            .tolist()
        )

        review_text = render_pose_review(review_ids)

        review_text = review_text.replace(
            "## Manual pose review evidence",
            "### 6.1 Manual pose review evidence",
            1,
        )

        boundary = "## 7. Workflow classification"

        if lines.count(boundary) != 1:
            raise RuntimeError(
                "Report section boundary changed."
            )

        pos = lines.index(boundary)

        lines[pos:pos] = (
            review_text.splitlines() + [""]
        )

        return "\n".join(lines)
