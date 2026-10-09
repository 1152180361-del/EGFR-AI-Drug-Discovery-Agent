import io
import json
import os
import tempfile
import unittest

from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from egfr_agent.run_agent import main


class TestCLIFailureAudit(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_dir = Path.cwd()
        os.chdir(self.tmp.name)

        Path("reports").mkdir()

        self.report = Path(
            "reports/EGFR_AI_Agent_Report.md"
        )
        self.trace = Path(
            "reports/EGFR_AI_Agent_Tool_Trace.json"
        )

        self.report.write_text("OLD_REPORT")
        self.trace.write_text("OLD_TRACE")

    def tearDown(self):
        os.chdir(self.original_dir)
        self.tmp.cleanup()

    def test_three_execution_failures(self):
        failures = [
            OSError("SIMULATED_QSAR_FAILURE"),
            OSError("SIMULATED_STRUCTURE_FAILURE"),
            ConnectionError("SIMULATED_OLLAMA_FAILURE"),
        ]

        for index, error in enumerate(failures, 1):
            with self.subTest(error=str(error)):
                with patch(
                    "egfr_agent.run_agent.EGFRAgent"
                ) as fake:
                    fake.return_value.run.side_effect = error

                    with redirect_stderr(io.StringIO()):
                        code = main([
                            "--query",
                            "EGFR activity and structure",
                        ])

                self.assertEqual(code, 1)

                files = sorted(
                    Path("reports/failures").glob("*.json")
                )
                self.assertEqual(len(files), index)

                record = json.loads(
                    files[-1].read_text(
                        encoding="utf-8"
                    )
                )

                self.assertEqual(
                    record["status"], "failed"
                )
                self.assertEqual(
                    record["exit_code"], 1
                )
                self.assertEqual(
                    record["error_type"],
                    type(error).__name__
                )
                self.assertEqual(
                    record["failure_stage"],
                    "agent_execution"
                )
                self.assertFalse(
                    record["trusted_decision_generated"]
                )
                self.assertEqual(
                    self.report.read_text(),
                    "OLD_REPORT"
                )
                self.assertEqual(
                    self.trace.read_text(),
                    "OLD_TRACE"
                )

    def test_initialization_failure(self):
        with patch(
            "egfr_agent.run_agent.EGFRAgent",
            side_effect=RuntimeError("INIT_FAILED"),
        ):
            with redirect_stderr(io.StringIO()):
                code = main([
                    "--query", "EGFR analysis"
                ])

        self.assertEqual(code, 1)

        files = list(
            Path("reports/failures").glob("*.json")
        )
        self.assertEqual(len(files), 1)

        record = json.loads(
            files[0].read_text(encoding="utf-8")
        )
        self.assertEqual(
            record["failure_stage"],
            "agent_initialization"
        )

    def test_success_has_no_failure_record(self):
        with patch(
            "egfr_agent.run_agent.EGFRAgent"
        ) as fake:
            fake.return_value.run.return_value = {}
            code = main([
                "--query", "EGFR analysis"
            ])

        self.assertEqual(code, 0)
        self.assertFalse(
            Path("reports/failures").exists()
        )


if __name__ == "__main__":
    unittest.main()
