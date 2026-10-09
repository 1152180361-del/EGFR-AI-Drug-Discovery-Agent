from pathlib import Path

import pandas as pd

from .base_tool import BaseTool


BASE = Path(__file__).resolve().parents[2]


class StructureTool(BaseTool):

    name = "structure_analysis"

    def run(self, top_n=10):

        decision_file = (
            BASE
            / "results/zinc15_screening"
            / "phase5_final_candidate_decision_matrix.csv"
        )

        evidence_file = (
            BASE
            / "results/zinc15_screening"
            / "phase4E_final_candidate_evidence_matrix.csv"
        )

        for path in [decision_file, evidence_file]:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required evidence file missing: {path}"
                )

        decision = pd.read_csv(decision_file)
        evidence = pd.read_csv(evidence_file)

        key = "phase4C_candidate_id"

        required_decision = [
            key,
            "candidate_track",
            "consensus_rank",
            "qsar_predicted_pIC50",
            "key_ATP_contact_coverage",
            "MET793_distance",
        ]

        required_evidence = [
            key,
            "max_training_tanimoto",
            "rf_tree_sd_pIC50",
        ]

        for col in required_decision:
            if col not in decision.columns:
                raise ValueError(
                    f"Missing decision column: {col}"
                )

        for col in required_evidence:
            if col not in evidence.columns:
                raise ValueError(
                    f"Missing evidence column: {col}"
                )

        if decision[key].duplicated().any():
            raise ValueError(
                "Duplicate IDs in decision matrix"
            )

        if evidence[key].duplicated().any():
            raise ValueError(
                "Duplicate IDs in Phase 4E matrix"
            )

        # Only import fields absent from the decision matrix.
        # This avoids duplicate columns with _x and _y suffixes.
        additional_fields = [
            "max_training_tanimoto",
            "rf_tree_sd_pIC50",
            "top1_vina_score",
            "reference_contact_jaccard",
            "severe_clash_pairs_lt1p5A",
            "no_severe_clash",
            "strict_QSAR_support",
            "RF_spread_warning",
        ]

        available_extra = [
            col
            for col in additional_fields
            if col in evidence.columns
            and col not in decision.columns
        ]

        evidence_subset = evidence[
            [key] + available_extra
        ].copy()

        merged = decision.merge(
            evidence_subset,
            on=key,
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        unmatched = merged["_merge"].ne("both")

        if unmatched.any():
            missing_ids = merged.loc[
                unmatched, key
            ].tolist()

            raise ValueError(
                "Unmatched candidate IDs: "
                + str(missing_ids)
            )

        merged = merged.drop(columns=["_merge"])

        numeric_fields = [
            "qsar_predicted_pIC50",
            "max_training_tanimoto",
            "rf_tree_sd_pIC50",
            "key_ATP_contact_coverage",
            "MET793_distance",
        ]

        for col in numeric_fields:
            merged[col] = pd.to_numeric(
                merged[col],
                errors="raise",
            )

            if merged[col].isna().any():
                raise ValueError(
                    f"Missing numeric evidence: {col}"
                )

        merged = merged.sort_values(
            ["consensus_rank", key],
            ascending=[True, True],
            kind="stable",
        )

        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
            or top_n < 1
        ):
            raise ValueError(
                "top_n must be a positive integer"
            )

        result = merged.head(top_n).copy()

        output_columns = [
            key,
            "zinc_id",
            "candidate_role",
            "candidate_track",
            "consensus_rank",
            "rank_v2",
            "rank_v3",
            "qsar_predicted_pIC50",
            "max_training_tanimoto",
            "rf_tree_sd_pIC50",
            "top1_vina_score",
            "reference_contact_jaccard",
            "key_ATP_contact_coverage",
            "MET793_distance",
            "severe_clash_pairs_lt1p5A",
            "no_severe_clash",
            "strict_QSAR_support",
            "RF_spread_warning",
            "experimental_priority",
        ]

        return result[
            [
                col
                for col in output_columns
                if col in result.columns
            ]
        ]
