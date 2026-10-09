
class EGFRPlanner:


    def plan(self, user_request):


        request = user_request.lower()


        plan=[]


        # -----------------------
        # Activity prediction
        # -----------------------

        if (
            "activity" in request
            or
            "qsar" in request
            or
            "inhibitor" in request
        ):

            plan.append(
                "qsar_prediction"
            )



        # -----------------------
        # Structure analysis
        # -----------------------

        if (
            "structure" in request
            or
            "docking" in request
            or
            "ATP" in user_request
        ):

            plan.append(
                "structure_analysis"
            )



        # -----------------------
        # Default discovery task
        # -----------------------

        if len(plan)==0:

            plan=[

                "qsar_prediction",

                "structure_analysis"

            ]



        return plan
