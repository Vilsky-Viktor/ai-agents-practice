import datasets
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.ensemble import EnsembleRetriever

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 3


def load_guest_documents() -> list[Document]:
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
    return [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}",
            ]),
            metadata={"name": guest["name"]},
        )
        for guest in guest_dataset
    ]


def build_hybrid_retriever(docs: list[Document], k: int = TOP_K) -> EnsembleRetriever:
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(docs)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])


guest_documents = load_guest_documents()
hybrid_retriever = build_hybrid_retriever(guest_documents)
