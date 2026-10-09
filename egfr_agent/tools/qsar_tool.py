from pathlib import Path

import pandas as pd

from .base_tool import BaseTool


BASE = Path(__file__).resolve().parents[2]


class QSARTool(BaseTool):

    name = "qsar_prediction"

    def run(self, top_n=5):

        file = (
            BASE
            / "results/zinc15_screening"
            / "zinc15_221988_qsar_predictions.csv"
        )

        if not file.is_file():
            raise FileNotFoundError(
                f"Required QSAR snapshot missing: {file}"
            )

        df = pd.read_csv(file)

        result = (
            df.sort_values(
                "qsar_predicted_pIC50",
                ascending=False
            )
            .head(top_n)
        )

        return result[
            [
                "zinc_id",
                "qsar_predicted_pIC50",
                "mwt",
                "logp",
            ]
        ]
