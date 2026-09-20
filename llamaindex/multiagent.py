from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.tools import QueryEngineTool
from datasets import load_dataset
from pathlib import Path
import asyncio
from llama_index.core import SimpleDirectoryReader
import chromadb
from dotenv import load_dotenv
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

async def test():
    dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")
    
    Path("data").mkdir(parents=True, exist_ok=True)
    for i, persona in enumerate(dataset):
        with open(Path("data") / f"persona_{i}.txt", "w") as f:
            f.write(persona["persona"])


    reader = SimpleDirectoryReader(input_dir="data")
    documents = reader.load_data()

    print(len(documents))

    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection(name="alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
    )

    nodes = await pipeline.arun(documents=documents[:10])

    print(nodes)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="name",
        description="a specific description",
        return_direct=False,
    )

    # Create agent configs
    # NOTE: we can use FunctionAgent or ReActAgent here.
    # FunctionAgent works for LLMs with a function calling API.
    # ReActAgent works for any LLM.
    calculator_agent = ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant. Use your tools for any math operation.",
        tools=[add, subtract],
        llm=llm,
    )

    query_agent = ReActAgent(
        name="info_lookup",
        description="Looks up information about XYZ",
        system_prompt="Use your tool to query a RAG system to answer information about XYZ",
        tools=[query_engine_tool],
        llm=llm
    )

    # Create and run the workflow
    agent = AgentWorkflow(
        agents=[calculator_agent, query_agent], root_agent="calculator"
    )

    # Run the system
    response = await agent.run(user_msg="Can you add 5 and 3?")
    print(response)

asyncio.run(test())