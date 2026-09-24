import gradio as gr
import os
from dotenv import load_dotenv

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, ToolCollection, MCPClient

load_dotenv()

mcp_client = MCPClient(
    {"url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse", "transport": "sse",} # This is the MCP Client we created in the previous section
)
tools = mcp_client.get_tools()

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.getenv("HF_TOKEN")
)
agent = CodeAgent(tools=[*tools], model=model)

demo = gr.ChatInterface(
    fn=lambda message, history: str(agent.run(message)),
    examples=["Prime factorization of 68"],
    title="Agent with MCP Tools",
    description="This is a simple agent that uses MCP tools to answer questions."
)

demo.launch()