import pandas as pd



class EvidenceEngine:



    def integrate(
        self,
        qsar_df,
        structure_df
    ):


        print("\n")

        print("="*60)

        print("EVIDENCE FUSION")

        print("="*60)



        merged = structure_df.copy()



        # QSAR normalization

        qsar_score = (

            qsar_df
            ["qsar_predicted_pIC50"]

            .rank(
                pct=True
            )

        )



        merged["structure_score"] = (

            1 -

            merged["consensus_rank"]

            /

            merged["consensus_rank"].max()

        )



        merged["qsar_score"] = (

            qsar_score.iloc[0]

        )



        # final evidence score


        merged["agent_score"] = (

            0.4 *

            merged["qsar_score"]

            +

            0.6 *

            merged["structure_score"]

        )



        result = (

            merged

            .sort_values(

                "agent_score",

                ascending=False

            )

        )



        return result
