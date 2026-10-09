import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .llm_client import OllamaClient
from .tool_scope import EGFRToolScopeResolver
from .function_calling import EGFRFunctionCaller
from .tool_registry import TOOLS
from .risk_engine import EGFRAgentRiskEngine
from .evaluate_function_calling import CASES


class EGFRLayeredEvaluator:

    def __init__(self):
        self.scope_resolver = EGFRToolScopeResolver()
        self.llm = OllamaClient()

    @staticmethod
    def exact_match(actual, expected):
        return (
            len(actual) == len(set(actual))
            and set(actual) == set(expected)
        )

    def evaluate_raw_llm(self, request):
        """
        Probe first-round, unconstrained tool selection.

        Both tools are visible to the model.
        No tool is actually executed here.

        This is a separate diagnostic prompt,
        not the identical production prompt.
        """

        system = """
You are an EGFR computational tool selector.

Available tools:

qsar_prediction:
Read precomputed global ZINC QSAR predictions.

structure_analysis:
Read existing EGFR candidate structural evidence.

Use native tool calling.

Choose only tools needed for the user's request.
Respect explicit exclusions such as "do not"
or "without".

Do not call a structure-analysis tool merely
because the user requests ranking of global
QSAR pIC50 predictions.

Each tool has no arguments.

Do not claim that computational predictions
are experimental validation.
"""

        response = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": request,
                },
            ],
            tools=EGFRFunctionCaller.TOOL_SCHEMAS,
        )

        message = response["message"]
        calls = message.get("tool_calls") or []

        if not isinstance(calls, list):
            raise RuntimeError(
                "Invalid raw tool_calls response."
            )

        names = []

        for call in calls:
            if not isinstance(call, dict):
                names.append("INVALID_CALL")
                continue

            function = call.get("function") or {}

            if not isinstance(function, dict):
                names.append("INVALID_FUNCTION")
                continue

            names.append(
                str(function.get("name", "MISSING_NAME"))
            )

        return {
            "tools_proposed_first_round": names,
            "note": (
                "Unconstrained first-round diagnostic. "
                "No tools executed. The diagnostic "
                "system prompt differs from production."
            ),
        }

    def evaluate_scope(self, request):

        return self.scope_resolver.resolve(
            request
        )

    @staticmethod
    def inspect_evidence(results):
        """
        Verifies schema and source separation.
        Does NOT validate biological activity.
        """

        findings = []
        errors = []

        if "qsar_prediction" in results:

            frame = results["qsar_prediction"]

            required = {
                "zinc_id",
                "qsar_predicted_pIC50",
            }

            missing = required - set(frame.columns)

            if missing:
                errors.append(
                    "QSAR columns missing: "
                    + str(sorted(missing))
                )
            elif frame.empty:
                errors.append(
                    "QSAR evidence table is empty."
                )
            else:
                findings.append(
                    "Global ZINC QSAR snapshot has "
                    "compound IDs and predicted pIC50."
                )

        if "structure_analysis" in results:

            frame = results["structure_analysis"]

            required = {
                "phase4C_candidate_id",
                "candidate_track",
                "max_training_tanimoto",
                "rf_tree_sd_pIC50",
                "MET793_distance",
                "key_ATP_contact_coverage",
            }

            missing = required - set(frame.columns)

            if missing:
                errors.append(
                    "Structure columns missing: "
                    + str(sorted(missing))
                )

            elif frame.empty:
                errors.append(
                    "Structure evidence table is empty."
                )

            elif frame[
                "phase4C_candidate_id"
            ].duplicated().any():
                errors.append(
                    "Duplicated candidate identifiers."
                )

            else:
                engine = EGFRAgentRiskEngine()

                try:
                    risk = engine.evaluate(
                        frame.iloc[0]
                    )

                    findings.append(
                        "Candidate evidence supports "
                        "risk-category evaluation: "
                        + risk["confidence"]
                    )

                except Exception as exc:
                    errors.append(
                        "Risk assessment failed: "
                        + str(exc)
                    )

        if (
            "qsar_prediction" in results
            and "structure_analysis" in results
        ):

            qsar = results["qsar_prediction"]
            structure = results["structure_analysis"]

            if (
                "zinc_id" in qsar.columns
                and "zinc_id" in structure.columns
            ):
                findings.append(
                    "Both datasets contain zinc_id; "
                    "any candidate-level linkage "
                    "still requires an explicit "
                    "validated one-to-one join."
                )

            else:
                findings.append(
                    "Global QSAR and candidate tables "
                    "were NOT joined; no candidate "
                    "identity equivalence is inferred."
                )

        findings.append(
            "This audit checks evidence schemas "
            "and provenance, not experimental "
            "inhibition or model predictive accuracy."
        )

        return {
            "passed": len(errors) == 0,
            "findings": findings,
            "errors": errors,
        }

    def evaluate_case(self, case, scope_only=False):

        start = time.perf_counter()
        expected = case["expected"]
        request = case["request"]

        result = {
            "case_id": case["id"],
            "group": case["group"],
            "request": request,
            "expected_tools": expected,
            "raw_llm": None,
            "raw_first_round_exact": None,
            "scope": None,
            "scope_exact": False,
            "guarded_execution": None,
            "execution_exact": None,
            "evidence_audit": None,
            "errors": [],
        }

        # Layer 1: Rule-based authorization

        try:
            scope = self.evaluate_scope(request)

            result["scope"] = scope

            result["scope_exact"] = self.exact_match(
                scope["allowed_tools"],
                expected,
            )

        except Exception as exc:
            result["errors"].append(
                "scope: "
                + f"{type(exc).__name__}: {exc}"
            )

        if scope_only:
            result["elapsed_seconds"] = round(
                time.perf_counter() - start,
                2,
            )
            return result

        # Layer 2: First-round LLM diagnostic

        try:
            raw = self.evaluate_raw_llm(request)

            result["raw_llm"] = raw

            result["raw_first_round_exact"] = (
                self.exact_match(
                    raw["tools_proposed_first_round"],
                    expected,
                )
            )

        except Exception as exc:
            result["errors"].append(
                "raw_llm: "
                + f"{type(exc).__name__}: {exc}"
            )

        # Layer 3: Guarded native tool execution

        try:
            caller = EGFRFunctionCaller(TOOLS)

            execution = caller.run(request)

            executed = execution["executed_tools"]

            duplicate_attempts = sum(
                entry.get("status")
                == "duplicate_blocked"
                for entry in execution["call_log"]
            )

            reprompts = sum(
                entry.get("event")
                == "coverage_reprompt"
                for entry in execution["call_log"]
            )

            result["guarded_execution"] = {
                "model": execution["model"],
                "executed_tools": executed,
                "model_requested_tools": (
                    execution.get(
                        "model_requested_tools", []
                    )
                ),
                "rounds": execution["rounds"],
                "duplicate_attempts": (
                    duplicate_attempts
                ),
                "coverage_reprompts": reprompts,
            }

            result["execution_exact"] = (
                self.exact_match(
                    executed,
                    expected,
                )
            )

            # Layer 4: Source and schema integrity

            audit = self.inspect_evidence(
                execution["results"]
            )

            result["evidence_audit"] = audit

            if not audit["passed"]:
                result["errors"].extend(
                    audit["errors"]
                )

        except Exception as exc:
            result["errors"].append(
                "execution: "
                + f"{type(exc).__name__}: {exc}"
            )

        result["elapsed_seconds"] = round(
            time.perf_counter() - start,
            2,
        )

        return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--group",
        choices=[
            "baseline", "challenge", "all"
        ],
        default="challenge",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--scope-only",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "EGFR_Agent_Layered_Evaluation.json"
        ),
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    selected = [
        item
        for item in CASES
        if (
            args.group == "all"
            or item["group"] == args.group
        )
    ]

    if args.limit is not None:
        selected = selected[:args.limit]

    evaluator = EGFRLayeredEvaluator()
    records = []

    for index, case in enumerate(selected, 1):

        print(
            f"\n[{index}/{len(selected)}] "
            f"{case['id']}"
        )

        result = evaluator.evaluate_case(
            case,
            scope_only=args.scope_only,
        )

        records.append(result)

        print(
            "Expected:",
            result["expected_tools"],
        )

        if result["scope"] is not None:
            print(
                "Authorized:",
                result["scope"]["allowed_tools"],
            )

        print(
            "Scope exact:",
            result["scope_exact"],
        )

        if not args.scope_only:
            raw = result["raw_llm"]

            if raw:
                print(
                    "Raw LLM first-round proposal:",
                    raw["tools_proposed_first_round"],
                )

            print(
                "Raw LLM exact:",
                result["raw_first_round_exact"],
            )

            execution = result["guarded_execution"]

            if execution:
                print(
                    "Guarded execution:",
                    execution["executed_tools"],
                )

            print(
                "Execution exact:",
                result["execution_exact"],
            )

            audit = result["evidence_audit"]

            if audit:
                print(
                    "Evidence schema/provenance:",
                    audit["passed"],
                )

        if result["errors"]:
            print(
                "Errors:",
                result["errors"],
            )

    count = len(records)

    if count == 0:
        raise RuntimeError(
            "No evaluation cases selected."
        )

    def count_true(field):
        return sum(
            item[field] is True
            for item in records
        )

    summary = {
        "cases": count,
        "scope_exact": count_true("scope_exact"),
        "raw_first_round_exact": count_true(
            "raw_first_round_exact"
        ),
        "guarded_execution_exact": count_true(
            "execution_exact"
        ),
        "evidence_audit_pass": sum(
            item["evidence_audit"] is not None
            and item["evidence_audit"]["passed"]
            for item in records
        ),
        "cases_with_errors": sum(
            bool(item["errors"])
            for item in records
        ),
    }

    destination = Path(args.output)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "model": evaluator.llm.model,
                "scope_only": args.scope_only,
                "group": args.group,
                "summary": summary,
                "cases": records,
                "limitations": [
                    "Raw LLM proposal is a diagnostic "
                    "using a separate system prompt.",
                    "Guarded execution is influenced "
                    "by deterministic authorization "
                    "and mandatory tool coverage.",
                    "Scientific audit validates "
                    "schema and provenance only.",
                    "No experimental activity "
                    "validation is performed.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("LAYERED EVALUATION SUMMARY")
    print("=" * 60)

    print("Cases:", count)

    print(
        "Scope exact:",
        f"{summary['scope_exact']}/{count}",
    )

    if not args.scope_only:
        print(
            "Raw LLM first-round exact:",
            f"{summary['raw_first_round_exact']}/{count}",
        )

        print(
            "Guarded execution exact:",
            f"{summary['guarded_execution_exact']}/{count}",
        )

        print(
            "Evidence audit pass:",
            f"{summary['evidence_audit_pass']}/{count}",
        )

    print(
        "Cases with errors:",
        summary["cases_with_errors"],
    )

    print("Saved:", destination)


if __name__ == "__main__":
    main()
