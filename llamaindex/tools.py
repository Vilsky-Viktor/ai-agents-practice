from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.tools.google import GmailToolSpec
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
import chromadb

## function tool

def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)
result = tool.call("New York")
print(result)

## query tool

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

db = chromadb.PersistentClient(path="./vector_db")
chroma_collection = db.get_or_create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(llm=llm)
tool = QueryEngineTool.from_defaults(query_engine, name="search engine", description="search engine to find information in vector DB")

## tool spec

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

metadata = [(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]
print(metadata)


## tools from MCP server

async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with the MCP server's tools.",
        tools=tools,
        llm=llm,
        system_prompt="You are an agent that can work with the MCP server's tools.",
    )
    return agent

async def mcp_tools():
    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    mcp_tool = McpToolSpec(client=mcp_client)

    agent = await get_agent(mcp_tool)

    return agent

