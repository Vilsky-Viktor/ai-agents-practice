# smolagents

Independent tutorial scripts exploring [smolagents](https://github.com/huggingface/smolagents), Hugging Face's lightweight code-first agent framework. Each file is a standalone, runnable exercise — there's no shared entry point.

## Scripts

| File | What it demonstrates |
|---|---|
| `main.py` | Basic `CodeAgent` usage: a custom `@tool`, and running search/menu/reasoning agents. |
| `tool_calling_agent.py` | `ToolCallingAgent` with `WebSearchTool`. |
| `party.py` | Custom `@tool` functions and a `Tool` subclass, with Langfuse tracing. |
| `tools.py` | Multiple tool-creation patterns (function tool, class tool, loaded Space tool). |
| `rag.py` | A `Tool` subclass wrapping a LangChain BM25 retriever (agentic RAG). |
| `lang_chain.py` | Wrapping a LangChain tool (SerpAPI search) for use in smolagents. |
| `multi_agent.py` | A manager/worker multi-agent hierarchy for a geospatial planning task. |
| `mcp_agent.py` | Connecting to an MCP server (`pubmedmcp`) and using its tools. |
| `space_tools.py` | Loading a Hugging Face Space (image generation) as a tool. |
| `vision.py` | Browser automation via Selenium/Helium, driven by a `CodeAgent`. |

## Running

```bash
poetry install
poetry run python <script>.py
```

Requires a `.env` with the relevant API keys (Hugging Face, SerpAPI, Langfuse, etc. depending on the script — see `.env.example`).
