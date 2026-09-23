# ai-agents

A collection of standalone projects exploring different AI agent frameworks and techniques (Hugging Face Agents Course exercises, LangGraph, LlamaIndex, RAG, evaluation/observability, and fine-tuning). Each subfolder is an independent Poetry (or plain Python) project — see its own README for details.

Deployed Hugging Face Spaces: [profile](https://huggingface.co/spaces/viktor-vilskyi) · [first-agent](https://huggingface.co/spaces/viktor-vilskyi/first-agent) · [hf-certification-agent](https://huggingface.co/spaces/viktor-vilskyi/hf-certification-agent)

- **[gaia-agent/](gaia-agent/)** — LangGraph agent built for the Hugging Face Agents Course Unit 4 certification (GAIA benchmark), deployed as a Hugging Face Space ([hf-certification-agent](https://huggingface.co/spaces/viktor-vilskyi/hf-certification-agent)).
- **[smolagents/](smolagents/)** — A set of independent smolagents tutorial scripts covering tool use, RAG, MCP, multi-agent systems, vision, and browser automation.
- **[first-agent/](first-agent/)** — A smolagents agent deployed as a Gradio-based Hugging Face Space ([first-agent](https://huggingface.co/spaces/viktor-vilskyi/first-agent), Unit 1 course template).
- **[langgraph/](langgraph/)** — Two small LangGraph apps: an email-triage workflow and a document-vision agent.
- **[llamaindex/](llamaindex/)** — Independent LlamaIndex tutorial scripts covering agents, workflows, multi-agent systems, RAG, and tool creation.
- **[mcp/](mcp/)** — Model Context Protocol examples: a standalone MCP server, Gradio as an MCP server, and an MCP client agent (`tiny-agents` + Playwright) for browser automation.
- **[rag/](rag/)** — A LangGraph "gala butler" agent with hybrid (BM25 + embeddings) retrieval, web search, and conversation memory.
- **[evaluation/](evaluation/)** — Langfuse-based evaluation examples: offline dataset experiments and an online interactive chat with LLM-as-a-judge scoring.
- **[lora-training/](lora-training/)** — LoRA fine-tuning of Gemma 2 for function-calling, using TRL/PEFT.
