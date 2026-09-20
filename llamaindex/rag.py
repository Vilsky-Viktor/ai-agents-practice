
import chromadb
import nest_asyncio
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.evaluation import FaithfulnessEvaluator
from arize.otel import register

load_dotenv()

ARIZE_API_KEY = os.getenv("PHOENIX_API_KEY")
ARIZE_SPACE_ID = os.getenv("PHOENIX_SPACE_ID")  # "Copy Space ID" button in Arize AX Space Settings

register(
    space_id=ARIZE_SPACE_ID,
    api_key=ARIZE_API_KEY,
    project_name="llamaindex-rag",
    auto_instrument=True,
)

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

    nest_asyncio.apply()  # This is needed to run the query engine
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    response = query_engine.query(
        "Respond using a persona that describes author and travel experiences?"
    )

    print(response)

    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_result = evaluator.evaluate_response(response=response)

    print(eval_result.passing)


asyncio.run(test())






