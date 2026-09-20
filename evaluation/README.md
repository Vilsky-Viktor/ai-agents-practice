# evaluation

Two [Langfuse](https://langfuse.com/) evaluation examples for a smolagents `CodeAgent`, both using the same model (`Qwen/Qwen2.5-Coder-32B-Instruct` via Hugging Face Inference).

## `offline.py` — Dataset experiment

Uploads a sample of the GSM8K math-word-problem dataset to Langfuse as a dataset, then runs the agent over it via `dataset.run_experiment(...)`, tracing each item automatically.

## `online.py` — Interactive chat with LLM-as-a-judge

A terminal chat loop (agent has `DuckDuckGoSearchTool`) where each turn is traced as a Langfuse span. After every response, a second LLM call scores the response's quality (1-5) as an `evaluator`-typed observation, and the user can optionally give y/n feedback with a comment — both logged as scores on the trace.

## Running

```bash
poetry install
poetry run python offline.py
poetry run python online.py
```

Requires a `.env` with `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL`, and `HF_TOKEN` — see `.env.example`.
