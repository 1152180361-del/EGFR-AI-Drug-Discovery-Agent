import re

import pandas as pd


class EGFRConclusionGuard:

    ID_PATTERN = r"EGFR_CAND_\d+"

    PRIMARY_PATTERNS = [
        re.compile(
            r"\b(?:primary\s+(?:candidate|recommendation)|"
            r"recommended\s+candidate|best\s+candidate|"
            r"top[- ]ranked\s+candidate)\s*"
            r"(?:is|:|=|-)?\s*(EGFR_CAND_\d+)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:recommend|select|choose|prioritize)\s+"
            r"(EGFR_CAND_\d+)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(EGFR_CAND_\d+)\s+(?:is|was)\s+"
            r"(?:the\s+)?(?:best|top[- ]ranked|"
            r"primary|recommended)\s+candidate\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(EGFR_CAND_\d+)\s+should\s+be\s+"
            r"(?:prioritized|selected|recommended)\b",
            re.IGNORECASE,
        ),
    ]

    @staticmethod
    def _find_candidate(structure_df, candidate_id):

        rows = structure_df.loc[
            structure_df["phase4C_candidate_id"]
            == candidate_id
        ]

        if len(rows) != 1:
            raise ValueError(
                "Candidate evidence missing or duplicated: "
                + str(candidate_id)
            )

        return rows.iloc[0]

    def build_verdict(
        self,
        ranking_df,
        structure_df,
        risk_result,
    ):

        if ranking_df is None or ranking_df.empty:
            raise ValueError(
                "Final ranking is empty."
            )

        if structure_df is None or structure_df.empty:
            raise ValueError(
                "Candidate evidence is empty."
            )

        best = ranking_df.iloc[0]

        candidate_id = str(
            best["candidate"]
        )

        row = self._find_candidate(
            structure_df,
            candidate_id,
        )

        track = str(
            row["candidate_track"]
        )

        if risk_result["candidate_track"] != track:
            raise ValueError(
                "Risk category and candidate track disagree."
            )

        similarity = float(
            row["max_training_tanimoto"]
        )

        rf_sd = float(
            row["rf_tree_sd_pIC50"]
        )

        met = float(
            row["MET793_distance"]
        )

        coverage = float(
            row["key_ATP_contact_coverage"]
        )

        alternatives = []

        for _, alt in ranking_df.iloc[1:4].iterrows():

            alt_id = str(
                alt["candidate"]
            )

            alt_row = self._find_candidate(
                structure_df,
                alt_id,
            )

            alternatives.append({
                "candidate_id": alt_id,
                "candidate_track": str(
                    alt_row["candidate_track"]
                ),
                "qsar_predicted_pIC50": float(
                    alt_row["qsar_predicted_pIC50"]
                ),
                "max_training_tanimoto": float(
                    alt_row["max_training_tanimoto"]
                ),
                "MET793_distance_A": float(
                    alt_row["MET793_distance"]
                ),
            })

        return {
            "primary_candidate": candidate_id,
            "candidate_track": track,
            "computational_evidence_category": (
                risk_result["confidence"]
            ),
            "heuristic_decision_score": float(
                best["score"]
            ),
            "qsar_predicted_pIC50": float(
                row["qsar_predicted_pIC50"]
            ),
            "max_training_tanimoto": similarity,
            "rf_tree_sd_pIC50": rf_sd,
            "MET793_distance_A": met,
            "key_ATP_contact_coverage": coverage,
            "candidates_considered": int(
                len(ranking_df)
            ),
            "alternatives": alternatives,
            "decision_source": (
                "deterministic_decision_engine"
            ),
            "evidence_source": (
                "precomputed_candidate_evidence"
            ),
            "experimental_status": "not_validated",
        }

    def audit_llm_summary(
        self,
        llm_summary,
        verdict,
    ):

        text = (
            llm_summary
            if isinstance(llm_summary, str)
            else ""
        )

        primary = verdict.get(
            "primary_candidate"
        )

        claims = set()
        warnings = []

        for pattern in self.PRIMARY_PATTERNS:

            for match in pattern.finditer(text):
                claims.add(
                    match.group(1).upper()
                )

        conflicting = sorted(
            candidate_id
            for candidate_id in claims
            if candidate_id != primary
        )

        if conflicting:
            warnings.append(
                "LLM made an explicit candidate "
                "recommendation conflicting with "
                "the deterministic decision: "
                + ", ".join(conflicting)
            )

        # Ambiguous group-prioritization statements
        # require manual interpretation.
        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            text,
        )

        ambiguous = []

        for sentence in sentences:

            ids = set(
                match.upper()
                for match in re.findall(
                    self.ID_PATTERN,
                    sentence,
                    flags=re.IGNORECASE,
                )
            )

            priority_language = re.search(
                r"\b(?:prioriti[sz](?:e|ed|ation)|"
                r"top\s+candidates|"
                r"recommend(?:ed|ation)?|"
                r"experimental\s+validation)\b",
                sentence,
                flags=re.IGNORECASE,
            )

            if len(ids) >= 2 and priority_language:
                ambiguous.append(
                    sentence.strip()[:400]
                )

        if ambiguous:
            warnings.append(
                "LLM grouped multiple candidates "
                "into a prioritization statement; "
                "this is not an authoritative ranking."
            )

        if re.search(
            r"\b(?:experimentally\s+(?:validated|"
            r"confirmed|proven)|"
            r"clinically\s+(?:validated|proven)|"
            r"confirmed\s+egfr\s+inhibitor)\b",
            text,
            flags=re.IGNORECASE,
        ):
            warnings.append(
                "Possible unsupported experimental "
                "validation claim in LLM summary."
            )

        return {
            "status": "unverified_llm_commentary",
            "publish_as_scientific_conclusion": False,
            "primary_candidate_from_engine": primary,
            "explicit_primary_claims": sorted(claims),
            "conflicting_claims": conflicting,
            "ambiguous_group_statements": ambiguous,
            "warnings": warnings,
            "note": (
                "Pattern-based warnings are not "
                "exhaustive scientific fact checking. "
                "The raw LLM summary is never used "
                "as the final recommendation."
            ),
        }

    @staticmethod
    def render_markdown(verdict, audit):

        primary = verdict["primary_candidate"]

        lines = [
            "## 9. Evidence-locked final conclusion",
            "",
            f"**Primary computational candidate: {primary}**",
            "",
            "**Decision source:** Deterministic Python "
            "Decision Engine operating on existing "
            "candidate evidence.",
            "",
            "**Computational support category:** "
            + verdict["computational_evidence_category"],
            "",
            "**Key metrics:**",
            "",
            "- Predicted pIC50: "
            f"{verdict['qsar_predicted_pIC50']:.3f}",
            "- Training-set maximum Tanimoto: "
            f"{verdict['max_training_tanimoto']:.3f}",
            "- RF tree spread: "
            f"{verdict['rf_tree_sd_pIC50']:.3f}",
            "- ATP contact coverage: "
            f"{verdict['key_ATP_contact_coverage']:.3f}",
            "- MET793 proximity: "
            f"{verdict['MET793_distance_A']:.2f} A",
            "",
            "**Interpretation:** This is a "
            "heuristic computational prioritization. "
            "It is not experimental evidence of EGFR "
            "inhibition or proof of superiority.",
            "",
            "**Alternative candidates:**",
            "",
        ]

        for item in verdict["alternatives"]:
            lines.append(
                "- "
                + item["candidate_id"]
                + " ("
                + item["candidate_track"]
                + ")"
            )

        lines.extend([
            "",
            "**LLM commentary status:** "
            + audit["status"],
            "",
            "The LLM's original text is retained "
            "for audit only and is not used to "
            "override the deterministic candidate "
            "selection.",
            "",
        ])

        if audit["warnings"]:

            lines.append(
                "**LLM narrative audit warnings:**"
            )
            lines.append("")

            for warning in audit["warnings"]:
                lines.append(
                    "- " + warning
                )

            lines.append("")

        return "\n".join(lines)
