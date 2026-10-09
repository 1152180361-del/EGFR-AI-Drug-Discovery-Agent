from egfr_agent.cli_registry import CLI_TOOLS



print(

"Available CLI tools:"

)


print(

list(CLI_TOOLS.keys())

)



tool = (

CLI_TOOLS["structure_pipeline"]

["tool"]

)



output = tool.run()



print(output)
