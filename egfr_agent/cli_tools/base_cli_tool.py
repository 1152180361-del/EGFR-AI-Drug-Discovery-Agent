import subprocess
import time



class BaseCLITool:


    name = "base_cli_tool"



    def run_command(
        self,
        command,
        timeout=600
    ):


        print("\nRUNNING COMMAND:")

        print(
            " ".join(command)
        )


        start=time.time()


        try:


            result = subprocess.run(

                command,

                capture_output=True,

                text=True,

                timeout=timeout

            )


            elapsed=time.time()-start



            print(
                "\nElapsed:",
                round(elapsed,2),
                "s"
            )



            if result.returncode != 0:


                raise RuntimeError(

                    result.stderr

                )


            return result.stdout



        except subprocess.TimeoutExpired:


            raise RuntimeError(

                "Command timeout"

            )
