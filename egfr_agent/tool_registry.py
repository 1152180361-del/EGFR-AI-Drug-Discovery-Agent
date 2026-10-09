
from .tools import (
    QSARTool,
    StructureTool
)



class ToolWrapper:


    def __init__(
        self,
        name,
        description,
        tool
    ):

        self.name=name

        self.description=description

        self.tool=tool



    def run(self):

        return self.tool.run()



TOOLS={


    "qsar_prediction":

    ToolWrapper(

        "qsar_prediction",

        "Predict molecular activity using QSAR model",

        QSARTool()

    ),



    "structure_analysis":

    ToolWrapper(

        "structure_analysis",

        "Analyze docking pose and structural evidence",

        StructureTool()

    )

}



def list_tools():


    return list(
        TOOLS.keys()
    )
