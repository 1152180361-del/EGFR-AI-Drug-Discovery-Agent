from .base_cli_tool import BaseCLITool

import pandas as pd

from pathlib import Path



class QSARCLITool(BaseCLITool):


    name="qsar_cli"



    def run(self):


        script = [

            "python",

            "scripts/30_dummy_qsar.py"

        ]


        # 这里先模拟CLI调用

        # 后续替换成真实QSAR脚本


        output=self.run_command(

            script

        )


        return output
