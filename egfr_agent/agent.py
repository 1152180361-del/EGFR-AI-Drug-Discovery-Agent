import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .tool_registry import TOOLS
from .function_calling import EGFRFunctionCaller
from .decision_engine import EGFRDecisionEngine
from .risk_engine import EGFRAgentRiskEngine
from .comparison_engine import EGFRComparisonEngine
from .report_generator import EGFRReportGenerator
from .conclusion_guard import EGFRConclusionGuard
from .memory import EGFRMemory


class EGFRAgent:

    def __init__(self):

        self.tools = TOOLS

        self.function_caller = EGFRFunctionCaller(
            self.tools
        )

        self.decision_engine = EGFRDecisionEngine()
        self.risk_engine = EGFRAgentRiskEngine()
        self.comparison_engine = EGFRComparisonEngine()
        self.report_generator = EGFRReportGenerator()
        self.conclusion_guard = EGFRConclusionGuard()
        self.memory = EGFRMemory()

    @staticmethod
    def _parse_original_flags(series):

        normalized = (
            series.astype(str)
            .str.strip()
            .str.lower()
        )

        mapping = {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }

        result = normalized.map(mapping)

        if result.isna().any():
            raise ValueError(
                "Unrecognized original "
                "RF_spread_warning values."
            )

        return result.astype(bool)

    @classmethod
    def _audit_rf_definitions(cls, frame):

        required = [
            "candidate_track",
            "rf_tree_sd_pIC50",
            "RF_spread_warning",
        ]

        missing = [
            col for col in required
            if col not in frame.columns
        ]

        if missing:
            raise ValueError(
                "Missing RF audit fields: "
                + str(missing)
            )

        original = cls._parse_original_flags(
            frame["RF_spread_warning"]
        )

        expected_original = (
            frame["candidate_track"]
            == "SUPPORTED_B_HIGH_SPREAD"
        )

        invalid = (
            original != expected_original
        )

        if invalid.any():
            raise ValueError(
                "Phase 4E original RF flag does "
                "not match its source definition "
                f"for {int(invalid.sum())} rows."
            )

        rf_sd = pd.to_numeric(
            frame["rf_tree_sd_pIC50"],
            errors="raise",
        )

        if not rf_sd.notna().all():
            raise ValueError(
                "Missing RF SD measurements."
            )

        threshold = (
            EGFRAgentRiskEngine.RF_SD_WARNING
        )

        above_threshold = (
            rf_sd > threshold
        )

        return {
            "candidate_count": int(len(frame)),
            "original_category_flag_true": int(
                original.sum()
            ),
            "rf_sd_above_threshold": int(
                above_threshold.sum()
            ),
            "source_definition_valid": True,
            "rf_sd_threshold": float(threshold),
            "original_flag_rule": (
                "candidate_track == "
                "'SUPPORTED_B_HIGH_SPREAD'"
            ),
            "numeric_flag_rule": (
                "rf_tree_sd_pIC50 > "
                f"{threshold}"
            ),
        }

    @staticmethod
    def _save_trace(
        user_request,
        function_result,
        verdict=None,
        narrative_audit=None,
        rf_audit=None,
    ):

        path = Path(
            "reports/EGFR_AI_Agent_Tool_Trace.json"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        trace = {
            "generated_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "user_request": user_request,
            "model": function_result["model"],
            "mode": function_result["mode"],
            "scope": function_result.get("scope"),
            "required_tools": (
                function_result["required_tools"]
            ),
            "executed_tools": (
                function_result["executed_tools"]
            ),
            "model_requested_tools": (
                function_result.get(
                    "model_requested_tools", []
                )
            ),
            "call_log": function_result["call_log"],
            "llm_summary_status": (
                "unverified_not_used_for_final_decision"
            ),
            "llm_summary": (
                function_result["llm_summary"]
            ),
            "narrative_audit": narrative_audit,
            "trusted_decision": verdict,
            "rf_flag_definition_audit": rf_audit,
            "scientific_limitations": [
                "No new docking calculation.",
                "No new QSAR inference.",
                "No experimental validation.",
                "Candidate ranking is heuristic.",
                "RF tree SD is not calibrated "
                "predictive uncertainty.",
                "Original RF_spread_warning is "
                "a candidate-track category marker.",
                "LLM commentary cannot override "
                "the deterministic decision.",
            ],
        }

        path.write_text(
            json.dumps(
                trace,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return path

    def run(self, user_request):

        print("=" * 60)
        print("EGFR EVIDENCE-GROUNDED LLM AGENT")
        print("=" * 60)

        print("\nUSER REQUEST:")
        print(user_request)

        # ==================================
        # Native LLM function calling
        # ==================================

        function_result = (
            self.function_caller.run(
                user_request
            )
        )

        results = function_result["results"]

        structure_result = results.get(
            "structure_analysis"
        )

        if structure_result is None:

            trace_path = self._save_trace(
                user_request,
                function_result,
            )

            print(
                "\nOnly requested evidence tools "
                "were executed."
            )

            print(
                "No candidate ranking generated "
                "for this QSAR-only task."
            )

            print(
                "TOOL AUDIT SAVED:",
                trace_path,
            )

            return {
                "function_calling": function_result,
                "trace_path": str(trace_path),
            }

        structure_result = (
            structure_result.copy()
        )

        # ==================================
        # Preserve source RF category flag
        # ==================================

        rf_audit = self._audit_rf_definitions(
            structure_result
        )

        threshold = (
            self.risk_engine.RF_SD_WARNING
        )

        # New field; original remains untouched.
        structure_result[
            "rf_sd_above_threshold"
        ] = (
            pd.to_numeric(
                structure_result[
                    "rf_tree_sd_pIC50"
                ],
                errors="raise",
            )
            > threshold
        )

        print("\nRF FLAG DEFINITIONS VERIFIED")

        print(
            "Original category flag True:",
            rf_audit["original_category_flag_true"],
        )

        print(
            "RF SD above operational threshold:",
            rf_audit["rf_sd_above_threshold"],
        )

        print(
            "Original Phase 4E flags preserved."
        )

        # ==================================
        # Deterministic decision
        # ==================================

        ranking = self.decision_engine.analyze(
            structure_result
        )

        if ranking is None or ranking.empty:
            raise RuntimeError(
                "Decision Engine returned no ranking."
            )

        best = ranking.iloc[0]

        candidate_id = str(
            best["candidate"]
        )

        selected = structure_result.loc[
            structure_result[
                "phase4C_candidate_id"
            ] == candidate_id
        ]

        if len(selected) != 1:
            raise RuntimeError(
                "Selected candidate evidence "
                "missing or duplicated: "
                + candidate_id
            )

        candidate_row = selected.iloc[0]

        # ==================================
        # Risk and comparison
        # ==================================

        risk = self.risk_engine.evaluate(
            candidate_row
        )

        comparison = (
            self.comparison_engine.compare(
                ranking,
                structure_result,
            )
        )

        # ==================================
        # Evidence-locked conclusion
        # ==================================

        verdict = (
            self.conclusion_guard.build_verdict(
                ranking,
                structure_result,
                risk,
            )
        )

        narrative_audit = (
            self.conclusion_guard.audit_llm_summary(
                function_result["llm_summary"],
                verdict,
            )
        )

        if (
            verdict["primary_candidate"]
            != candidate_id
        ):
            raise RuntimeError(
                "Conclusion disagrees with "
                "Decision Engine."
            )

        print("\n" + "=" * 60)
        print("TRUSTED COMPUTATIONAL DECISION")
        print("=" * 60)

        print(
            "Primary candidate:",
            verdict["primary_candidate"],
        )

        print(
            "Evidence category:",
            verdict[
                "computational_evidence_category"
            ],
        )

        print(
            "Candidates evaluated:",
            verdict["candidates_considered"],
        )

        print(
            "LLM summary status:",
            narrative_audit["status"],
        )

        if narrative_audit["warnings"]:

            print("\nLLM NARRATIVE WARNINGS:")

            for warning in narrative_audit["warnings"]:
                print("-", warning)

        # ==================================
        # Scientific report
        # ==================================

        report = self.report_generator.generate(
            ranking,
            structure_result,
            risk,
            comparison,
        )

        report += (
            "\n## 8. Native function-calling audit\n\n"
            "- Model: `"
            + function_result["model"]
            + "`\n"
            "- Mode: `"
            + function_result["mode"]
            + "`\n"
            "- Executed tools: "
            + ", ".join(
                function_result["executed_tools"]
            )
            + "\n"
            "- Candidate ranking: deterministic "
            "Python Decision Engine\n"
            "- Candidates reviewed in this run: "
            f"{len(structure_result)}\n"
            "- Original `RF_spread_warning` definition: "
            "`candidate_track == "
            "SUPPORTED_B_HIGH_SPREAD`\n"
            "- Original category flags marked True: "
            f"{rf_audit['original_category_flag_true']}\n"
            "- Independent `rf_sd_above_threshold` "
            f"rule: `RF tree SD > {threshold}`\n"
            "- Candidate rows above numerical RF "
            "threshold: "
            f"{rf_audit['rf_sd_above_threshold']}\n"
            "- The two RF flags have different "
            "definitions and are not expected "
            "to be equal.\n"
            "- Source CSV files were not edited.\n"
            "- Raw LLM commentary is unverified "
            "and not used for candidate selection.\n"
        )

        report += "\n" + (
            self.conclusion_guard.render_markdown(
                verdict,
                narrative_audit,
            )
        )

        report_path = Path(
            "reports/EGFR_AI_Agent_Report.md"
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.write_text(
            report,
            encoding="utf-8",
        )

        # ==================================
        # Structured audit trace
        # ==================================

        trace_path = self._save_trace(
            user_request,
            function_result,
            verdict=verdict,
            narrative_audit=narrative_audit,
            rf_audit=rf_audit,
        )

        # ==================================
        # Memory
        # ==================================

        self.memory.add_record(
            user_request,
            candidate_id,
            str(best["reason"]),
        )

        print("\nREPORT SAVED:", report_path)
        print("AUDIT SAVED:", trace_path)

        print(
            "\nScientific status: computational "
            "prioritization only; experimental "
            "EGFR inhibition is unverified."
        )

        return {
            "candidate": candidate_id,
            "verdict": verdict,
            "ranking": ranking,
            "risk": risk,
            "comparison": comparison,
            "rf_audit": rf_audit,
            "narrative_audit": narrative_audit,
            "function_calling": function_result,
            "report_path": str(report_path),
            "trace_path": str(trace_path),
        }
