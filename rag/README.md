# rag

A LangGraph "gala butler" agent (Alfred) that answers questions about gala guests, with:

- **Hybrid retrieval** — BM25 keyword search + HuggingFace embeddings, fused via `EnsembleRetriever` (`rag/retriever.py`).
- **Tools** — guest-info lookup (with a suggested conversation starter), a dummy weather tool, a Hugging Face Hub stats tool, and DuckDuckGo web search as a fallback for unknown guests (`rag/tools.py`).
- **Conversation memory** — a `MemorySaver` checkpointer so the agent recalls earlier turns in the same session (`rag/app.py`).

## Running

```bash
poetry install
poetry run python -m rag.app
```

Requires a `.env` with `HF_API_TOKEN` — see `.env.example`. Running it walks through a demo conversation showing guest lookup, memory recall, web-search fallback, weather, and Hub-stats queries.
