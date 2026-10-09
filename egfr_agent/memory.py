
import json

from pathlib import Path




class EGFRMemory:


    def __init__(self):


        self.file = Path(
            "reports/agent_memory.json"
        )


        if self.file.exists():

            with open(
                self.file,
                "r"
            ) as f:

                self.memory=json.load(f)


        else:

            self.memory=[]




    def add_record(
        self,
        request,
        candidate,
        reason
    ):


        record={

            "request":
            request,


            "recommended_candidate":
            candidate,


            "reason":
            reason

        }


        self.memory.append(record)


        self.save()




    def save(self):


        self.file.parent.mkdir(
            exist_ok=True
        )


        with open(
            self.file,
            "w"
        ) as f:


            json.dump(

                self.memory,

                f,

                indent=2

            )




    def history(self):


        return self.memory




    def retrieve_latest(self):


        if len(self.memory)==0:

            return None



        return self.memory[-1]
