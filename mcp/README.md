# mcp

Examples exploring the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), using the `mcp` 1.x SDK (pinned deliberately — see Notes).

## `test.py` — A minimal MCP server

Builds a `FastMCP` server ("Weather Service") exposing a tool (`get_weather`), a resource (`weather://{location}`), and a prompt (`weather_report`).

```bash
poetry install
poetry run python test.py
```

## `gradio_app.py` — Gradio as an MCP server

A simple Gradio `Interface` (a letter-counter tool) launched with `mcp_server=True`, so Gradio exposes it as an MCP server automatically (Streamable HTTP at `/gradio_api/mcp/`).

```bash
poetry run python gradio_app.py
```

## `mcp_sentiment/app.py` — Sentiment analysis as an MCP server

A Gradio `Interface` wrapping `TextBlob` sentiment analysis (polarity, subjectivity, assessment), also launched with `mcp_server=True`. Uses the shared `mcp/` Poetry environment (no separate `pyproject.toml`).

```bash
cd mcp_sentiment
poetry run python app.py
```

## `agent.json` — MCP client via `tiny-agents`

Configures a [`tiny-agents`](https://huggingface.co/docs/huggingface_hub/en/guides/mcp) agent (Qwen 2.5-72B via Novita) connected to the [Playwright MCP server](https://github.com/microsoft/playwright-mcp) for browser automation.

```bash
poetry run tiny-agents run agent.json
```

Requires Node.js 20+ (for `npx @playwright/mcp@latest`) and a Hugging Face token with inference access. The JS `tiny-agents` CLI validates `provider` against a fixed enum that doesn't include every real HF inference provider (e.g. no `nebius`) — check with `huggingface_hub.model_info(model, expand=['inferenceProviderMapping'])` if a provider gets rejected.

## `client.py` — MCP client via smolagents

A `smolagents` `CodeAgent` that pulls tools from a remote Gradio MCP server (`MCPClient` + SSE transport) and exposes a `gr.ChatInterface` on top.

```bash
poetry run python client.py
```

## `mcp_sentiment/agent.json` / `mcp_sentiment/client.py` — MCP client via `huggingface_hub.Agent`

Connects to the locally running `mcp_sentiment/app.py` server (`http://localhost:7860/gradio_api/mcp/sse`) via `npx mcp-remote`, either through the `tiny-agents` CLI (`agent.json`) or directly in Python with `huggingface_hub.Agent` (`client.py`). Start `app.py` first, then in another shell:

```bash
cd mcp_sentiment
poetry run python client.py
```

`Agent.run()` is an **async generator**, not an awaitable — it must be consumed with `async for`, not `await`. It yields two different item types that need to be handled separately:
- `ChatCompletionStreamOutput` chunks (has `.choices`) — streamed text deltas in `chunk.choices[0].delta.content`, to be concatenated for the assistant's reply.
- `ChatCompletionInputMessage` objects (`role == "tool"`) — full tool-call results in `.content`, yielded once a tool finishes running.

```python
async def test():
    final_answer = ""
    async for chunk in agent.run("Test sentiment of this sentence"):
        if hasattr(chunk, "choices"):
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                final_answer += delta.content
        elif getattr(chunk, "role", None) == "tool":
            print(f"[tool result] {chunk.content}")

    print(final_answer)
```

## Notes

- `mcp` is pinned to `^1.30.0`, not the newer 2.x line. Gradio's `mcp_server=True` feature (used by `gradio_app.py` and `mcp_sentiment/app.py`) genuinely does not work with `mcp` 2.x — it fails at runtime with `'Server' object has no attribute 'call_tool'`, since `gradio[mcp]`'s own declared dependency caps `mcp<2.0.0` for a real reason, not just an untested constraint. All four scripts in this project share one Poetry environment, so the whole project stays on 1.x to keep everything working together.
- If experimenting with `mcp` 2.x here again: `test.py` would need `from mcp.server.fastmcp import FastMCP` changed to `from mcp.server.mcpserver import MCPServer` (renamed in 2.x), and `tiny-agents` (via `huggingface_hub`) currently breaks against 2.x with `AttributeError: 'Tool' object has no attribute 'inputSchema'` (renamed to `input_schema`) — not fixed upstream as of this writing.
- When testing Gradio startup output manually, run Python unbuffered (`python -u`) or check the log file after the process exits — buffered stdout can make a fully-working launch look like it's hanging.
