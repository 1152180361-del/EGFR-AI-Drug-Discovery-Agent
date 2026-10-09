from .base_cli_tool import BaseCLITool


class PipelineCLITool(BaseCLITool):


    def run_python_script(
        self,
        script_path,
        timeout=3600
    ):


        command = [

            "python",

            script_path

        ]


        output = self.run_command(

            command,

            timeout

        )


        return output
