from .pipeline_cli import PipelineCLITool



class QSARPipelineTool(PipelineCLITool):


    name = "qsar_pipeline"



    def run(self):


        print("\n")
        print("="*60)
        print("RUNNING QSAR DISCOVERY PIPELINE")
        print("="*60)



        outputs = []



        # Step 1 QSAR prediction

        print("\n[1/2] QSAR prediction")


        out1 = self.run_python_script(

            "scripts/36_predict_zinc15_qsar.py",

            timeout=7200

        )


        outputs.append(out1)



        # Step 2 Applicability domain

        print("\n[2/2] QSAR applicability domain")


        out2 = self.run_python_script(

            "scripts/37_qsar_applicability_domain.py",

            timeout=7200

        )


        outputs.append(out2)



        return "\n".join(outputs)
