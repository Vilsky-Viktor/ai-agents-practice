# llamaindex

Independent tutorial scripts exploring [LlamaIndex](https://www.llamaindex.ai/), covering agents, workflows, and RAG. Each file is a standalone, runnable exercise.

## Scripts

| File | What it demonstrates |
|---|---|
| `main.py` | Basic `HuggingFaceInferenceAPI` LLM usage. |
| `agents.py` | An `AgentWorkflow` with function tools and cross-call memory. |
| `agent_workflows.py` | Multiple `ReActAgent`s combined into one `AgentWorkflow` with shared state. |
| `simple_workflows.py` | The low-level `Workflow`/`step`/`Event` API (not agent-based), with a flow diagram export. |
| `tools.py` | Multiple tool-creation patterns: function tool, query-engine tool, Gmail tool spec, MCP tool spec. |
| `rag.py` | A RAG pipeline (Chroma vector store) with Arize Phoenix tracing and faithfulness evaluation. |
| `multiagent.py` | A RAG-backed query agent combined with a calculator agent in one multi-agent workflow. |

## Running

```bash
poetry install
poetry run python <script>.py
```

Requires a `.env` with the relevant API keys (Hugging Face, Arize Phoenix, etc. depending on the script — see `.env.example`). `rag.py`/`multiagent.py` download a sample dataset into `data/` and build a Chroma index in `alfred_chroma_db/` on first run.
