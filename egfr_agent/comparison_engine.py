

class EGFRComparisonEngine:



    def compare(
        self,
        ranking_df,
        structure_df
    ):


        best = ranking_df.iloc[0]


        alternatives = ranking_df.iloc[1:4]



        comparison=[]



        best_id = best["candidate"]



        best_row = structure_df[

            structure_df[
                "phase4C_candidate_id"
            ]
            ==
            best_id

        ].iloc[0]



        for _,alt in alternatives.iterrows():


            alt_row = structure_df[

                structure_df[
                    "phase4C_candidate_id"
                ]
                ==
                alt["candidate"]

            ]



            if len(alt_row)==0:

                continue



            alt_row=alt_row.iloc[0]



            result={}


            result["candidate"]=alt["candidate"]



            result["comparison"]=[]



            # QSAR reliability

            best_sim = best_row.get(
                "max_training_tanimoto",
                0
            )


            alt_sim = alt_row.get(
                "max_training_tanimoto",
                0
            )


            if best_sim > alt_sim:


                result["comparison"].append(

                    "Higher QSAR applicability confidence"

                )


            else:


                result["comparison"].append(

                    "Lower QSAR applicability confidence"

                )



            # Structure


            best_met = best_row.get(
                "MET793_distance",
                99
            )


            alt_met = alt_row.get(
                "MET793_distance",
                99
            )



            if alt_met < best_met:


                result["comparison"].append(

                    f"{alt['candidate']} has better hinge geometry"

                )

            else:


                result["comparison"].append(

                    "Primary candidate has better overall evidence"

                )



            comparison.append(result)



        return comparison
