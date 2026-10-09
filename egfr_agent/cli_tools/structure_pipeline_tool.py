from .pipeline_cli import PipelineCLITool



class StructurePipelineTool(PipelineCLITool):


    name="structure_pipeline"



    def run(self):


        print("\n")
        print("="*60)
        print("RUNNING STRUCTURE DISCOVERY PIPELINE")
        print("="*60)



        outputs=[]



        scripts=[

            "scripts/38_select_phase4c_scaffold_diverse_candidates.py",

            "scripts/40_final_candidate_scoring.py",

            "scripts/42_build_final_candidate_decision_matrix.py"

        ]



        for i,script in enumerate(scripts):


            print(

                f"\n[{i+1}/{len(scripts)}] {script}"

            )


            out=self.run_python_script(

                script,

                timeout=7200

            )


            outputs.append(out)



        return "\n".join(outputs)
