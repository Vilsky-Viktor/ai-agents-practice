# mcp

Examples exploring the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), using the `mcp` 2.x SDK.

## `test.py` — A minimal MCP server

Builds an `MCPServer` ("Weather Service") exposing a tool (`get_weather`), a resource (`weather://{location}`), and a prompt (`weather_report`).

```bash
poetry install
poetry run python test.py
```

## `gradio_app.py` — Gradio as an MCP server

A simple Gradio `Interface` (a letter-counter tool) launched with `mcp_server=True`, so Gradio exposes it as an MCP server automatically.

```bash
poetry run python gradio_app.py
```

## `agent.json` — MCP client via `tiny-agents`

Configures a [`tiny-agents`](https://huggingface.co/docs/huggingface_hub/en/guides/mcp) agent (Qwen 2.5-72B via Nebius) connected to the [Playwright MCP server](https://github.com/microsoft/playwright-mcp) for browser automation.

```bash
poetry run tiny-agents run agent.json
```

Requires Node.js 20+ (for `npx @playwright/mcp@latest`) and a Hugging Face token with inference access.

## Notes

- `mcp` is pinned to `^2.2.0` (the current major version, where `FastMCP` was renamed to `MCPServer`). `huggingface_hub`'s `tiny-agents` CLI has a known incompatibility with this version (`Tool.inputSchema` was renamed to `input_schema`) that isn't fixed upstream as of this writing — see `_mcp/mcp_client.py` in the installed `huggingface_hub` package if `tiny-agents run` fails with an `AttributeError` on `inputSchema`.
- `gradio[mcp]`'s declared dependency caps `mcp<2.0.0`, but plain `gradio` (without the `[mcp]` extra) installs fine alongside `mcp` 2.x. Runtime behavior of `mcp_server=True` against `mcp` 2.x wasn't fully confirmed (startup was slow/inconclusive in testing) — worth verifying if you rely on this.
