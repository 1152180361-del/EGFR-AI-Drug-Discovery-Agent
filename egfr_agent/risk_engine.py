import math


class EGFRAgentRiskEngine:

    # Operational screening thresholds.
    # These are not calibrated confidence intervals.
    AD_THRESHOLD = 0.535
    RF_SD_WARNING = 0.958

    @staticmethod
    def _number(row, field):

        if field not in row.index:
            raise ValueError(
                f"Missing evidence field: {field}"
            )

        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value for {field}"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite numeric value for {field}"
            )

        return value

    @staticmethod
    def _source_flag(row):

        field = "RF_spread_warning"

        if field not in row.index:
            raise ValueError(
                "Original RF_spread_warning flag missing."
            )

        value = str(row[field]).strip().lower()

        if value in {"true", "1", "yes"}:
            return True

        if value in {"false", "0", "no"}:
            return False

        raise ValueError(
            "Unrecognized original RF_spread_warning: "
            + str(row[field])
        )

    def evaluate(self, candidate_row):

        track = str(
            candidate_row.get(
                "candidate_track", ""
            )
        ).strip()

        if not track:
            raise ValueError(
                "Candidate track is missing."
            )

        similarity = self._number(
            candidate_row,
            "max_training_tanimoto",
        )

        rf_sd = self._number(
            candidate_row,
            "rf_tree_sd_pIC50",
        )

        met = self._number(
            candidate_row,
            "MET793_distance",
        )

        coverage = self._number(
            candidate_row,
            "key_ATP_contact_coverage",
        )

        # ==================================
        # Original Phase 4E category flag
        # ==================================

        original_flag = self._source_flag(
            candidate_row
        )

        expected_original = (
            track == "SUPPORTED_B_HIGH_SPREAD"
        )

        if original_flag != expected_original:
            raise ValueError(
                "Original RF_spread_warning disagrees "
                "with its Phase 4E definition: "
                "candidate_track == "
                "'SUPPORTED_B_HIGH_SPREAD'."
            )

        # ==================================
        # Independent numerical assessment
        # ==================================

        within_ad = (
            similarity >= self.AD_THRESHOLD
        )

        rf_sd_above = (
            rf_sd > self.RF_SD_WARNING
        )

        support = []
        risks = []

        if within_ad:
            support.append(
                "Within operational QSAR similarity "
                f"domain (Tanimoto={similarity:.3f})"
            )
        else:
            risks.append(
                "Outside operational QSAR similarity "
                f"domain (Tanimoto={similarity:.3f})"
            )

        if rf_sd_above:
            risks.append(
                "RF tree spread exceeds the operational "
                f"threshold ({rf_sd:.3f} > "
                f"{self.RF_SD_WARNING:.3f}); "
                "this spread is not a calibrated "
                "predictive uncertainty estimate"
            )
        else:
            support.append(
                "RF tree spread is within the operational "
                f"threshold (SD={rf_sd:.3f})"
            )

        if coverage >= 0.70:
            support.append(
                "Substantial reference ATP-pocket "
                f"contact coverage ({coverage:.3f})"
            )
        else:
            risks.append(
                "Limited reference ATP-pocket "
                f"contact coverage ({coverage:.3f})"
            )

        if met <= 4.0:
            support.append(
                "MET793 backbone heteroatom proximity "
                f"({met:.2f} A); hydrogen bond "
                "not confirmed"
            )
        else:
            risks.append(
                "No close MET793 backbone heteroatom "
                f"proximity ({met:.2f} A)"
            )

        clash_col = "severe_clash_pairs_lt1p5A"

        if clash_col in candidate_row.index:

            try:
                clashes = float(
                    candidate_row[clash_col]
                )
            except (TypeError, ValueError):
                clashes = float("nan")

            if math.isfinite(clashes):

                if clashes > 0:
                    risks.append(
                        "Severe heavy-atom overlaps "
                        f"below 1.5 A: {int(clashes)}"
                    )
                else:
                    support.append(
                        "No heavy-atom overlap below "
                        "the 1.5 A screening threshold"
                    )

        # ==================================
        # Computational evidence category
        # ==================================

        if track == "EXPLORATORY_OOD":

            confidence = "EXPLORATORY_OOD"

        elif track == "SUPPORTED_B_HIGH_SPREAD":

            confidence = (
                "AD_SUPPORTED_SPREAD_CAUTION"
            )

        elif (
            track == "SUPPORTED_A"
            and within_ad
            and not rf_sd_above
        ):

            confidence = (
                "AD_SUPPORTED_COMPUTATIONAL"
            )

        else:

            confidence = (
                "REQUIRES_EVIDENCE_REVIEW"
            )

        risks.append(
            "QSAR and docking evidence do not "
            "establish experimental EGFR inhibition"
        )

        return {
            "confidence": confidence,
            "candidate_track": track,
            "within_AD": bool(within_ad),

            # Preserved Phase 4E source definition.
            "RF_spread_warning_original": bool(
                original_flag
            ),

            # Independent numerical flag.
            "rf_sd_above_threshold": bool(
                rf_sd_above
            ),

            "rf_sd_operational_threshold": (
                self.RF_SD_WARNING
            ),

            "supporting_evidence": support,
            "risk_summary": risks,

            "metrics": {
                "max_training_tanimoto": similarity,
                "rf_tree_sd_pIC50": rf_sd,
                "MET793_distance_A": met,
                "ATP_contact_coverage": coverage,
            },

            "flag_definitions": {
                "RF_spread_warning_original": (
                    "True iff candidate_track is "
                    "SUPPORTED_B_HIGH_SPREAD"
                ),
                "rf_sd_above_threshold": (
                    "True iff rf_tree_sd_pIC50 > "
                    f"{self.RF_SD_WARNING}"
                ),
            },
        }
