import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from egfr_agent.tools.qsar_tool import QSARTool


class TestQSARToolPath(unittest.TestCase):

    def test_reads_snapshot_from_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            directory = (
                root / "results/zinc15_screening"
            )
            directory.mkdir(parents=True)

            snapshot = (
                directory
                / "zinc15_221988_qsar_predictions.csv"
            )

            pd.DataFrame([
                {
                    "zinc_id": "ZINC_TEST_A",
                    "qsar_predicted_pIC50": 6.5,
                    "mwt": 300.0,
                    "logp": 2.0,
                },
                {
                    "zinc_id": "ZINC_TEST_B",
                    "qsar_predicted_pIC50": 7.8,
                    "mwt": 350.0,
                    "logp": 3.0,
                },
                {
                    "zinc_id": "ZINC_TEST_C",
                    "qsar_predicted_pIC50": 7.0,
                    "mwt": 320.0,
                    "logp": 2.5,
                },
            ]).to_csv(snapshot, index=False)

            with patch(
                "egfr_agent.tools.qsar_tool.BASE",
                root
            ):
                result = QSARTool().run(top_n=2)

            self.assertEqual(len(result), 2)
            self.assertEqual(
                result["zinc_id"].tolist(),
                ["ZINC_TEST_B", "ZINC_TEST_C"]
            )

    def test_missing_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "egfr_agent.tools.qsar_tool.BASE",
                Path(tmp)
            ):
                with self.assertRaises(
                    FileNotFoundError
                ):
                    QSARTool().run()


if __name__ == "__main__":
    unittest.main()
