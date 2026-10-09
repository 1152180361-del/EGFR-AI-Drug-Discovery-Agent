import argparse
import contextlib
import io
import json
import time

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .evaluate_function_calling import CASES
from .function_calling import EGFRFunctionCaller
from .tool_registry import TOOLS
from .decision_engine import EGFRDecisionEngine
from .risk_engine import EGFRAgentRiskEngine
from .comparison_engine import EGFRComparisonEngine
from .conclusion_guard import EGFRConclusionGuard
from .report_generator import EGFRReportGenerator


def evaluate_once(case, repeat_number):

    start = time.perf_counter()

    record = {
        "case_id": case["id"],
        "repeat": repeat_number,
        "expected_tools": case["expected"],
        "authorized_tools": [],
        "executed_tools": [],
        "model_requested_tools": [],
        "scope_exact": False,
        "tool_exact": False,
        "duplicate_attempts": 0,
        "coverage_reprompts": 0,
        "primary_candidate": None,
        "evidence_category": None,
        "candidate_rows_evaluated": 0,
        "narrative_warnings": [],
        "decision_checks_passed": None,
        "rf_flag_disagreements": None,
        "passed": False,
        "error": None,
    }

    try:
        caller = EGFRFunctionCaller(TOOLS)

        # Capture verbose tool tables without
        # modifying the scientific tool outputs.
        output_buffer = io.StringIO()

        with contextlib.redirect_stdout(output_buffer):
            result = caller.run(case["request"])

        executed = result["executed_tools"]
        scope = result["scope"]

        allowed = scope["allowed_tools"]

        record["authorized_tools"] = allowed
        record["executed_tools"] = executed

        record["model_requested_tools"] = (
            result.get("model_requested_tools", [])
        )

        expected = set(case["expected"])

        record["scope_exact"] = (
            set(allowed) == expected
            and len(allowed) == len(expected)
        )

        record["tool_exact"] = (
            set(executed) == expected
            and len(executed) == len(expected)
        )

        record["duplicate_attempts"] = sum(
            item.get("status") == "duplicate_blocked"
            for item in result["call_log"]
        )

        record["coverage_reprompts"] = sum(
            item.get("event") == "coverage_reprompt"
            for item in result["call_log"]
        )

        structure = result["results"].get(
            "structure_analysis"
        )

        # QSAR-only tasks must not generate
        # structure-dependent candidate rankings.
        if structure is None:
            record["decision_checks_passed"] = (
                "structure_analysis" not in expected
            )

        else:
            structure = structure.copy()

            record["candidate_rows_evaluated"] = (
                len(structure)
            )

            # Inspect RF warning definitions without
            # modifying the source data.
            sd = pd.to_numeric(
                structure["rf_tree_sd_pIC50"],
                errors="raise",
            )

            if sd.isna().any():
                raise ValueError(
                    "Missing RF tree SD values."
                )

            threshold = (
                EGFRAgentRiskEngine.RF_SD_WARNING
            )

            recalculated = sd > threshold

            if "RF_spread_warning" in structure.columns:

                stored = (
                    structure["RF_spread_warning"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes"])
                )

                record["rf_flag_disagreements"] = int(
                    (stored != recalculated).sum()
                )

            ranking = EGFRDecisionEngine().analyze(
                structure
            )

            if ranking.empty:
                raise ValueError(
                    "Empty candidate ranking."
                )

            primary = str(
                ranking.iloc[0]["candidate"]
            )

            matches = structure.loc[
                structure["phase4C_candidate_id"]
                == primary
            ]

            if len(matches) != 1:
                raise ValueError(
                    "Primary candidate evidence "
                    "is missing or duplicated."
                )

            risk = EGFRAgentRiskEngine().evaluate(
                matches.iloc[0]
            )

            comparisons = EGFRComparisonEngine().compare(
                ranking,
                structure,
            )

            guard = EGFRConclusionGuard()

            verdict = guard.build_verdict(
                ranking,
                structure,
                risk,
            )

            narrative_audit = guard.audit_llm_summary(
                result["llm_summary"],
                verdict,
            )

            # Generate the report in memory only.
            # Do not overwrite the user's main report.
            report_text = (
                EGFRReportGenerator().generate(
                    ranking,
                    structure,
                    risk,
                    comparisons,
                )
            )

            report_text += "\n" + guard.render_markdown(
                verdict,
                narrative_audit,
            )

            decision_ok = (
                primary == verdict["primary_candidate"]
                and risk["confidence"]
                == verdict[
                    "computational_evidence_category"
                ]
                and (
                    "Evidence-locked final conclusion"
                    in report_text
                )
                and (
                    f"Primary computational candidate: "
                    f"{primary}"
                    in report_text
                )
            )

            record["primary_candidate"] = primary
            record["evidence_category"] = (
                risk["confidence"]
            )

            record["narrative_warnings"] = (
                narrative_audit["warnings"]
            )

            record["decision_checks_passed"] = bool(
                decision_ok
            )

        record["passed"] = bool(
            record["scope_exact"]
            and record["tool_exact"]
            and record["decision_checks_passed"]
            and record["duplicate_attempts"] == 0
        )

    except Exception as exc:
        record["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    record["elapsed_seconds"] = round(
        time.perf_counter() - start,
        2,
    )

    return record


def summarize(records):

    total = len(records)

    grouped = defaultdict(list)

    for item in records:
        grouped[item["case_id"]].append(item)

    per_case = []

    for case_id, entries in grouped.items():

        candidates = [
            entry["primary_candidate"]
            for entry in entries
            if entry["primary_candidate"] is not None
        ]

        unique_candidates = sorted(
            set(candidates)
        )

        expected_structure = (
            "structure_analysis"
            in entries[0]["expected_tools"]
        )

        candidate_stable = (
            len(candidates) == len(entries)
            and len(unique_candidates) == 1
        ) if expected_structure else None

        per_case.append({
            "case_id": case_id,
            "repeats": len(entries),
            "passed": sum(
                entry["passed"]
                for entry in entries
            ),
            "unique_primary_candidates": (
                unique_candidates
            ),
            "primary_candidate_stable": (
                candidate_stable
            ),
            "mean_elapsed_seconds": round(
                sum(
                    entry["elapsed_seconds"]
                    for entry in entries
                ) / len(entries),
                2,
            ),
        })

    passed = sum(
        record["passed"]
        for record in records
    )

    tool_exact = sum(
        record["tool_exact"]
        for record in records
    )

    scope_exact = sum(
        record["scope_exact"]
        for record in records
    )

    cases_with_errors = sum(
        record["error"] is not None
        for record in records
    )

    return {
        "total_runs": total,
        "passed_runs": passed,
        "strict_pass_rate": round(
            passed / total, 4
        ),
        "tool_exact_runs": tool_exact,
        "tool_exact_rate": round(
            tool_exact / total, 4
        ),
        "scope_exact_runs": scope_exact,
        "runs_with_errors": cases_with_errors,
        "runs_with_reprompts": sum(
            record["coverage_reprompts"] > 0
            for record in records
        ),
        "runs_with_duplicate_attempts": sum(
            record["duplicate_attempts"] > 0
            for record in records
        ),
        "case_summaries": per_case,
    }


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
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "EGFR_Agent_Stability_Evaluation.json"
        ),
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error(
            "--repeats must be at least 1."
        )

    selected = [
        case
        for case in CASES
        if (
            args.group == "all"
            or case["group"] == args.group
        )
    ]

    if not selected:
        raise RuntimeError(
            "No evaluation cases selected."
        )

    records = []

    total_runs = len(selected) * args.repeats

    print("=" * 60)
    print("EGFR AGENT STABILITY EVALUATION")
    print("=" * 60)

    print("Group:", args.group)
    print("Repeats per case:", args.repeats)
    print("Total runs:", total_runs)

    index = 0

    for case in selected:

        for repeat in range(
            1, args.repeats + 1
        ):

            index += 1

            print(
                f"\n[{index}/{total_runs}] "
                f"{case['id']} "
                f"repeat={repeat}",
                flush=True,
            )

            result = evaluate_once(
                case,
                repeat,
            )

            records.append(result)

            print(
                "Executed:",
                result["executed_tools"],
                flush=True,
            )

            print(
                "Primary:",
                result["primary_candidate"],
                flush=True,
            )

            print(
                "Passed:",
                result["passed"],
                flush=True,
            )

            print(
                "Time:",
                result["elapsed_seconds"],
                "s",
                flush=True,
            )

            if result["error"]:
                print(
                    "Error:",
                    result["error"],
                    flush=True,
                )

    summary = summarize(records)

    destination = Path(args.output)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            {
                "generated_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "evaluation": (
                    "repeated_native_tool_calling_"
                    "and_in_memory_decision_path"
                ),
                "group": args.group,
                "repeats": args.repeats,
                "summary": summary,
                "runs": records,
                "limitations": [
                    "Small task set; results cannot be "
                    "generalized to all user requests.",
                    "Repeated Qwen tool calling is "
                    "evaluated, but the scientific "
                    "input tables remain fixed.",
                    "Candidate ranking uses a "
                    "deterministic heuristic, not "
                    "experimentally validated potency.",
                    "Report generation is checked "
                    "in memory; memory persistence "
                    "and file-write reliability "
                    "are not measured.",
                    "The test does not perform "
                    "new docking or QSAR inference.",
                ],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("STABILITY SUMMARY")
    print("=" * 60)

    print(
        "Passed:",
        f"{summary['passed_runs']}"
        f"/{summary['total_runs']}",
    )

    print(
        "Strict pass rate:",
        f"{summary['strict_pass_rate']:.1%}",
    )

    print(
        "Tool exact rate:",
        f"{summary['tool_exact_rate']:.1%}",
    )

    print(
        "Runs with errors:",
        summary["runs_with_errors"],
    )

    print(
        "Runs with reprompts:",
        summary["runs_with_reprompts"],
    )

    print(
        "Runs with duplicate attempts:",
        summary["runs_with_duplicate_attempts"],
    )

    for item in summary["case_summaries"]:
        print(
            item["case_id"],
            "| passed:",
            f"{item['passed']}/{item['repeats']}",
            "| unique primary:",
            item["unique_primary_candidates"],
            "| stable:",
            item["primary_candidate_stable"],
        )

    print("\nSaved:", destination)

    raise SystemExit(
        0
        if summary["passed_runs"] == total_runs
        else 1
    )


if __name__ == "__main__":
    main()
