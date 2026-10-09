from egfr_agent.cli_registry import CLI_TOOLS



tool = CLI_TOOLS["qsar_cli"]["tool"]



result = tool.run()



print("\nRESULT:")

print(result)
