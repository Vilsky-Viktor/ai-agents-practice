# langgraph

Two small independent [LangGraph](https://langchain-ai.github.io/langgraph/) apps.

## `main.py` — Email triage

A ReAct-style workflow that reads an email, classifies it as spam or legitimate, then either discards spam or drafts a reply and notifies the user. Traced via `langfuse.langchain.CallbackHandler`. Running it processes two example emails (one legitimate, one spam) and writes the compiled graph to `graph.png`.

## `doc_analysis.py` — Document vision agent

An "Alfred the butler" agent that can extract text from an image via GPT-4o vision and do simple arithmetic (division). Running it processes two example requests and writes the compiled graph to `graph.png`.

## Running

```bash
poetry install
poetry run python main.py
poetry run python doc_analysis.py
```

Requires a `.env` with `OPENAI_API_KEY` and Langfuse credentials — see `.env.example`.
