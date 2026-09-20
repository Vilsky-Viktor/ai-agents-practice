from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from smolagents import CodeAgent, InferenceClientModel, Tool

load_dotenv()

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences.")