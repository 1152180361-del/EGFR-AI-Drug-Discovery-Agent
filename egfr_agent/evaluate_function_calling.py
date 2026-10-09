import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .function_calling import EGFRFunctionCaller
from .tool_registry import TOOLS


# =====================================================
# Golden evaluation dataset
# Expected tools are defined independently of the LLM.
# =====================================================

CASES = [
    {
        "id": "both_en",
        "group": "baseline",
        "request": (
            "Find EGFR ATP competitive inhibitors using "
            "existing QSAR activity predictions and "
            "docking pose evidence."
        ),
        "expected": [
            "qsar_prediction",
            "structure_analysis",
        ],
    },
    {
        "id": "qsar_en",
        "group": "baseline",
        "request": (
            "Retrieve precomputed global ZINC QSAR "
            "predicted pIC50 values."
        ),
        "expected": ["qsar_prediction"],
    },
    {
        "id": "structure_en",
        "group": "baseline",
        "request": (
            "Summarize existing EGFR docking poses "
            "and MET793 backbone proximity."
        ),
        "expected": ["structure_analysis"],
    },
    {
        "id": "both_zh",
        "group": "baseline",
        "request": (
            "请读取 EGFR 的活性预测结果，并分析"
            "已有对接构象和 ATP 口袋覆盖率。"
        ),
        "expected": [
            "qsar_prediction",
            "structure_analysis",
        ],
    },
    {
        "id": "structure_zh",
        "group": "baseline",
        "request": (
            "请比较 EGFR_CAND_003 和 EGFR_CAND_020 "
            "的 ATP 口袋接触与 MET793 距离。"
        ),
        "expected": ["structure_analysis"],
    },
    {
        "id": "qsar_zh",
        "group": "baseline",
        "request": (
            "读取 ZINC 已有的 QSAR 预测 pIC50 数据。"
        ),
        "expected": ["qsar_prediction"],
    },
    {
        "id": "qsar_rank_ambiguity",
        "group": "challenge",
        "request": (
            "List global ZINC QSAR predictions "
            "ranked by pIC50."
        ),
        "expected": ["qsar_prediction"],
    },
    {
        "id": "negated_structure",
        "group": "challenge",
        "request": (
            "Retrieve global ZINC QSAR activity "
            "predictions; do not perform structure "
            "analysis or docking."
        ),
        "expected": ["qsar_prediction"],
    },
]


# =====================================================
# Offline fake tools / fake LLM
# These tests require no Ollama inference.
# =====================================================

class FakeTool:

    def __init__(self, name):
        self.name = name
        self.calls = 0

    def run(self):
        self.calls += 1

        return pd.DataFrame([
            {
                "tool": self.name,
                "value": 1.0,
            }
        ])


class FakeClient:

    model = "offline_stub"

    def __init__(self, messages):
        self.messages = list(messages)
        self.index = 0

    def chat(self, messages, tools=None):

        if self.index >= len(self.messages):
            raise AssertionError(
                "Fake LLM response sequence exhausted."
            )

        result = self.messages[self.index]
        self.index += 1

        return {
            "message": result
        }


def tool_call(name, arguments=None):

    return {
        "function": {
            "name": name,
            "arguments": (
                arguments
                if arguments is not None
                else {}
            ),
        }
    }


def assistant_message(calls=None, content=""):

    return {
        "role": "assistant",
        "content": content,
        "tool_calls": calls or [],
    }


def make_offline_caller(responses):

    registry = {
        "qsar_prediction": FakeTool(
            "qsar_prediction"
        ),
        "structure_analysis": FakeTool(
            "structure_analysis"
        ),
    }

    caller = EGFRFunctionCaller(registry)

    caller.client = FakeClient(responses)

    return caller, registry


def expect_runtime_error(operation, expected_text):

    try:
        operation()
    except RuntimeError as exc:

        if expected_text not in str(exc):
            raise AssertionError(
                f"Unexpected error: {exc}"
            ) from exc

        return

    raise AssertionError(
        "Expected RuntimeError was not raised."
    )


def test_two_calls():

    caller, registry = make_offline_caller([
        assistant_message([
            tool_call("qsar_prediction"),
            tool_call("structure_analysis"),
        ]),
        assistant_message(
            content="Tool results received."
        ),
    ])

    result = caller.run(
        "Analyze EGFR QSAR activity and docking poses."
    )

    expected = {
        "qsar_prediction",
        "structure_analysis",
    }

    assert set(result["executed_tools"]) == expected

    assert registry[
        "qsar_prediction"
    ].calls == 1

    assert registry[
        "structure_analysis"
    ].calls == 1


def test_unknown_tool_rejected():

    caller, registry = make_offline_caller([
        assistant_message([
            tool_call("delete_project_files"),
        ]),
    ])

    expect_runtime_error(
        lambda: caller.run(
            "Retrieve EGFR docking poses."
        ),
        "Rejected unapproved LLM tool",
    )

    assert all(
        tool.calls == 0
        for tool in registry.values()
    )


def test_invalid_arguments_rejected():

    caller, registry = make_offline_caller([
        assistant_message([
            tool_call(
                "structure_analysis",
                {"shell_command": "echo unsafe"},
            ),
        ]),
    ])

    expect_runtime_error(
        lambda: caller.run(
            "Retrieve EGFR docking poses."
        ),
        "Unexpected tool arguments",
    )

    assert all(
        tool.calls == 0
        for tool in registry.values()
    )


def test_duplicate_call_blocked():

    caller, registry = make_offline_caller([
        assistant_message([
            tool_call("qsar_prediction"),
            tool_call("qsar_prediction"),
        ]),
        assistant_message(
            content="Finished."
        ),
    ])

    result = caller.run(
        "Retrieve EGFR QSAR predictions."
    )

    assert registry[
        "qsar_prediction"
    ].calls == 1

    assert any(
        entry.get("status") == "duplicate_blocked"
        for entry in result["call_log"]
    )


def test_missing_tool_reprompt():

    caller, registry = make_offline_caller([
        assistant_message(
            content="I can inspect the data."
        ),
        assistant_message([
            tool_call("qsar_prediction"),
        ]),
        assistant_message(
            content="Finished after tool result."
        ),
    ])

    result = caller.run(
        "Retrieve EGFR QSAR predictions."
    )

    assert result["executed_tools"] == [
        "qsar_prediction"
    ]

    assert registry[
        "qsar_prediction"
    ].calls == 1

    assert any(
        entry.get("event") == "coverage_reprompt"
        for entry in result["call_log"]
    )


def run_offline():

    checks = [
        ("two_native_calls", test_two_calls),
        ("unknown_tool_rejected",
         test_unknown_tool_rejected),
        ("invalid_arguments_rejected",
         test_invalid_arguments_rejected),
        ("duplicate_call_blocked",
         test_duplicate_call_blocked),
        ("missing_tool_reprompt",
         test_missing_tool_reprompt),
    ]

    results = []

    for name, check in checks:

        try:
            check()
            status = "PASS"
            error = None

        except Exception as exc:
            status = "FAIL"
            error = (
                f"{type(exc).__name__}: {exc}"
            )

        results.append({
            "test": name,
            "status": status,
            "error": error,
        })

        print(
            f"[{status}] {name}"
        )

        if error:
            print("  ", error)

    passed = sum(
        entry["status"] == "PASS"
        for entry in results
    )

    print(
        f"\nOFFLINE RESULT: "
        f"{passed}/{len(results)} passed"
    )

    return passed == len(results)


# =====================================================
# Live evaluations: real Qwen + actual read-only tools
# =====================================================

def evaluate_one(case):

    start = time.perf_counter()

    caller = EGFRFunctionCaller(TOOLS)

    try:

        result = caller.run(
            case["request"]
        )

        executed = result["executed_tools"]

        expected_set = set(
            case["expected"]
        )

        executed_set = set(
            executed
        )

        missing = sorted(
            expected_set - executed_set
        )

        extra = sorted(
            executed_set - expected_set
        )

        duplicate_attempts = sum(
            entry.get("status") == "duplicate_blocked"
            for entry in result["call_log"]
        )

        exact_match = (
            not missing and not extra
        )

        clean_execution = (
            exact_match
            and duplicate_attempts == 0
        )

        return {
            "case_id": case["id"],
            "group": case["group"],
            "request": case["request"],
            "expected_tools": case["expected"],
            "executed_tools": executed,
            "missing_tools": missing,
            "extra_tools": extra,
            "exact_match": exact_match,
            "clean_execution": clean_execution,
            "duplicate_attempts": duplicate_attempts,
            "rounds": result["rounds"],
            "model": result["model"],
            "latency_seconds": round(
                time.perf_counter() - start, 2
            ),
            "error": None,
        }

    except Exception as exc:

        return {
            "case_id": case["id"],
            "group": case["group"],
            "request": case["request"],
            "expected_tools": case["expected"],
            "executed_tools": [],
            "missing_tools": case["expected"],
            "extra_tools": [],
            "exact_match": False,
            "clean_execution": False,
            "duplicate_attempts": None,
            "rounds": None,
            "model": None,
            "latency_seconds": round(
                time.perf_counter() - start, 2
            ),
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


def run_live(group, limit, output):

    cases = [
        case
        for case in CASES
        if group == "all"
        or case["group"] == group
    ]

    if limit is not None:
        cases = cases[:limit]

    if not cases:
        raise ValueError(
            "No evaluation cases selected."
        )

    records = []

    print("=" * 60)
    print("EGFR NATIVE TOOL CALLING EVALUATION")
    print("=" * 60)

    for index, case in enumerate(cases, 1):

        print(
            f"\n[{index}/{len(cases)}] "
            f"{case['id']}"
        )

        print("Request:", case["request"])
        print("Expected:", case["expected"])

        record = evaluate_one(case)
        records.append(record)

        print(
            "Executed:",
            record["executed_tools"],
        )

        print(
            "Exact match:",
            record["exact_match"],
        )

        print(
            "Duration:",
            record["latency_seconds"],
            "seconds",
        )

        if record["extra_tools"]:
            print(
                "Extra tools:",
                record["extra_tools"],
            )

        if record["missing_tools"]:
            print(
                "Missing tools:",
                record["missing_tools"],
            )

        if record["error"]:
            print(
                "Error:",
                record["error"],
            )

    total = len(records)

    exact = sum(
        bool(r["exact_match"])
        for r in records
    )

    clean = sum(
        bool(r["clean_execution"])
        for r in records
    )

    errors = sum(
        r["error"] is not None
        for r in records
    )

    summary = {
        "total_cases": total,
        "exact_tool_set_matches": exact,
        "exact_match_rate": round(
            exact / total, 4
        ),
        "clean_executions": clean,
        "clean_execution_rate": round(
            clean / total, 4
        ),
        "execution_errors": errors,
        "total_latency_seconds": round(
            sum(
                r["latency_seconds"]
                for r in records
            ),
            2,
        ),
    }

    report = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation_type": (
            "native_ollama_function_calling"
        ),
        "case_group": group,
        "note": (
            "Tests real LLM tool selection and "
            "execution against predefined labels. "
            "Does not validate predicted EGFR potency "
            "or docking accuracy."
        ),
        "summary": summary,
        "cases": records,
    }

    destination = Path(output)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        "Cases:",
        summary["total_cases"],
    )

    print(
        "Exact tool selection:",
        f"{exact}/{total}",
    )

    print(
        "Exact match rate:",
        f"{summary['exact_match_rate']:.1%}",
    )

    print(
        "Clean execution rate:",
        f"{summary['clean_execution_rate']:.1%}",
    )

    print(
        "Errors:",
        summary["execution_errors"],
    )

    print(
        "Results saved:",
        destination,
    )

    return errors == 0 and exact == total


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EGFR Agent native function calling"
        )
    )

    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
    )

    parser.add_argument(
        "--group",
        choices=["baseline", "challenge", "all"],
        default="baseline",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "EGFR_Agent_Function_Calling_Evaluation.json"
        ),
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error(
            "--limit must be a positive integer."
        )

    if args.mode == "offline":
        ok = run_offline()

    else:
        ok = run_live(
            group=args.group,
            limit=args.limit,
            output=args.output,
        )

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
