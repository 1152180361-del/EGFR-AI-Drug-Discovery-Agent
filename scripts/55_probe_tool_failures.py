import io
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from egfr_agent.agent import EGFRAgent
from egfr_agent.function_calling import EGFRFunctionCaller


QUERY = (
    "Find promising EGFR ATP competitive inhibitors. "
    "Need activity prediction, docking pose "
    "and structural analysis."
)

TOOL_NAMES = (
    "qsar_prediction",
    "structure_analysis",
)


class FakeTool:

    def __init__(self, name, failing_tool):
        self.name = name
        self.failing_tool = failing_tool

    def run(self):
        if self.name == self.failing_tool:
            raise OSError(
                f"SIMULATED_{self.name}_FAILURE"
            )

        return pd.DataFrame([{
            "zinc_id": "SIMULATED_ZINC",
            "qsar_predicted_pIC50": 7.0,
        }])


class FakeClient:

    model = "offline-fake-client"

    def __init__(self, failure):
        self.failure = failure

    def chat(self, messages, tools=None):
        if self.failure == "ollama":
            raise ConnectionError(
                "SIMULATED_OLLAMA_CONNECTION_FAILURE"
            )

        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": name,
                            "arguments": {},
                        }
                    }
                    for name in TOOL_NAMES
                ],
            }
        }


def probe(failure):
    registry = {
        name: FakeTool(name, failure)
        for name in TOOL_NAMES
    }

    client = FakeClient(failure)

    with patch(
        "egfr_agent.function_calling.OllamaClient",
        return_value=client,
    ):
        caller = EGFRFunctionCaller(registry)

    # The test only exercises the initial tool-calling
    # boundary, so other Agent dependencies are unnecessary.
    agent = EGFRAgent.__new__(EGFRAgent)
    agent.function_caller = caller

    original_dir = Path.cwd()

    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)

            reports = Path("reports")
            reports.mkdir()

            old_report = reports / (
                "EGFR_AI_Agent_Report.md"
            )
            old_trace = reports / (
                "EGFR_AI_Agent_Tool_Trace.json"
            )

            old_report.write_text(
                "PREVIOUS_SUCCESSFUL_REPORT",
                encoding="utf-8",
            )
            old_trace.write_text(
                "PREVIOUS_SUCCESSFUL_TRACE",
                encoding="utf-8",
            )

            captured = io.StringIO()
            error = None

            try:
                with redirect_stdout(captured):
                    agent.run(QUERY)
            except Exception as exc:
                error = exc

            if error is None:
                raise AssertionError(
                    f"{failure}: failure not propagated"
                )

            if failure == "ollama":
                assert isinstance(
                    error, ConnectionError
                )
            else:
                assert isinstance(error, OSError)

            report_unchanged = (
                old_report.read_text(
                    encoding="utf-8"
                ) == "PREVIOUS_SUCCESSFUL_REPORT"
            )

            trace_unchanged = (
                old_trace.read_text(
                    encoding="utf-8"
                ) == "PREVIOUS_SUCCESSFUL_TRACE"
            )

            assert report_unchanged
            assert trace_unchanged

            failure_files = list(
                reports.glob("failures/*.json")
            )

            print(f"\nSCENARIO: {failure}")
            print(
                "Exception:",
                type(error).__name__,
                str(error),
            )
            print("Failure propagated: YES")
            print("Old report unchanged: YES")
            print("Old success trace unchanged: YES")
            print(
                "Independent failure audit files:",
                len(failure_files),
            )

            if not failure_files:
                print(
                    "Audit status: MISSING "
                    "(needs implementation)"
                )

        finally:
            os.chdir(original_dir)


def main():
    print("=" * 60)
    print("PHASE 7.4.4 - OFFLINE FAILURE INJECTION")
    print("=" * 60)

    for failure in (
        "qsar_prediction",
        "structure_analysis",
        "ollama",
    ):
        probe(failure)

    print("\nOFFLINE FAILURE PROBE COMPLETE")
    print("Production data and reports were not changed.")


if __name__ == "__main__":
    main()
