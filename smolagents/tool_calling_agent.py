from smolagents import ToolCallingAgent, WebSearchTool, InferenceClientModel

agent = ToolCallingAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="together")
)

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")