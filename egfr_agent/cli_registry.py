from .cli_tools import (

    QSARPipelineTool,

    StructurePipelineTool

)



CLI_TOOLS={



"qsar_pipeline":

{

"description":

"Run EGFR QSAR prediction pipeline",

"tool":

QSARPipelineTool()

},



"structure_pipeline":

{

"description":

"Run EGFR docking and structural evaluation pipeline",

"tool":

StructurePipelineTool()

}


}
