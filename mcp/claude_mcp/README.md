# Module 3: Slack Notification - Complete Solution

This is the complete implementation of Module 3, demonstrating how to integrate MCP Tools and Prompts for team communication via Slack.

## What This Implements

This solution extends Modules 1 and 2 with:

1. **`send_slack_notification` tool** - Sends formatted messages to Slack via webhook with proper error handling
2. **`format_ci_failure_alert` prompt** - Creates rich failure alerts with Slack markdown
3. **`format_ci_success_summary` prompt** - Creates celebration messages for successful deployments

## Setup and Usage

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Set up Slack webhook — copy `.env.example` to `.env` and fill in the URL (loaded automatically via `python-dotenv`):
   ```bash
   cp .env.example .env
   # then edit .env: SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ```

3. Start services:
   ```bash
   # Terminal 1: Webhook server
   python webhook_server.py
   
   # Terminal 2: MCP server
   poetry run python server.py
   
   # Terminal 3: Cloudflare tunnel (optional)
   cloudflared tunnel --url http://localhost:8080
   ```

## Testing

See `manual_test.md` for comprehensive testing instructions using curl commands to simulate GitHub webhook events.

## Key Learning Outcomes

This solution demonstrates all MCP primitives working together for real-world team automation.

## Notes

- Migrated from `uv` to Poetry; `mcp` is pinned to `^1.30.0` since `server.py` uses the `FastMCP` API (renamed to `MCPServer` in `mcp` 2.x — see [`mcp/README.md`](../README.md#notes) for why the sibling projects stay on 1.x too).
- `TEMPLATES_DIR` resolves to a local `templates/` folder next to `server.py`; `get_pr_templates()` auto-creates any missing template file via `create_default_template()`.
- Register with Claude Code from any directory using absolute paths:
  ```bash
  claude mcp add pr-agent-slack -- env -u VIRTUAL_ENV poetry run --directory /absolute/path/to/claude_mcp python /absolute/path/to/claude_mcp/server.py
  ```