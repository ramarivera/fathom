# fathom

CLI + MCP server wrapping the OpenAI Deep Research API.

Fathom exposes the full power of OpenAI's deep research models (`o3-deep-research`, `o4-mini-deep-research`) as both a standalone CLI tool and an MCP server — preserving **all** intermediate research artifacts (web searches, reasoning traces, code interpreter calls) that other wrappers discard.

## Features

- **Full intermediate output** — every web search, reasoning summary, MCP tool call, code interpreter call, and file search call is preserved and returned, not just the final report.
- **Web search configuration** — set user location (country, city, region). Deep Research currently accepts only `medium` search context size, so Fathom rejects unsupported values locally instead of letting the API fail.
- **Dual-transport MCP server** — stdio (for local agents like OpenCode/Claude Code) and HTTP with optional bearer token auth (for remote access).
- **Standalone CLI** — create, poll, and retrieve deep research jobs without needing an MCP client.

## Installation

```bash
cargo install fathom-mcp
```

For local development:

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
# Binary at target/release/fathom
```

## Usage

### Environment

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass `--api-key` to every command.

### CLI

```bash
# Start a research job (returns immediately with a response ID)
fathom research "What are the leading approaches to room-temperature superconductors as of 2025?"

# Start and wait for completion (5-30 minutes)
fathom research --wait "Compare Rust async runtimes: tokio vs async-std vs smol"

# Use o4-mini for faster/cheaper research
fathom research --model o4-mini "Summarize recent advances in CRISPR gene editing"

# Configure web search location
fathom research --country US --city "San Francisco" --region California "Local AI startup funding trends"

# Explicitly set the only Deep Research-supported search context size
fathom research --search-context-size medium "Comprehensive review of quantum error correction"

# Check job status
fathom status resp_abc123

# Get full results with citations, reasoning traces, and intermediate steps
fathom results resp_abc123

# Get just the report (no intermediate steps)
fathom results --no-steps resp_abc123
```

### MCP Server (stdio)

For local AI agents (OpenCode, Claude Code, etc.):

```bash
fathom serve
```

Configure in your MCP client:

```json
{
  "mcpServers": {
    "fathom": {
      "command": "fathom",
      "args": ["serve"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### MCP Server (HTTP)

For remote access with authentication:

```bash
fathom serve --transport http --bind 0.0.0.0:8080 --auth-token "my-secret-token"
```

Connect with any MCP client pointing to `http://host:8080/mcp` with `Authorization: Bearer my-secret-token`.

## MCP Tools

| Tool | Description |
|------|-------------|
| `create_research` | Start a new deep research job. Accepts query, model, web search config, and optional instructions. Returns the response ID and initial status. |
| `check_status` | Poll a research job by response ID. Returns status, progress indicators, and reasoning summaries. |
| `get_results` | Retrieve full results including the final report, all intermediate output items (web searches, reasoning, tool calls), citations, and token usage. |

## Roadmap

These features are planned but not yet implemented:

- [ ] **MCP server pass-through** — connect your own MCP servers as data sources to Deep Research
- [ ] **File search / vector store support** — attach vector stores for document-grounded research
- [ ] **Webhook support** — receive notifications when jobs complete
- [ ] **Conversation threading** — chain research jobs with `previous_response_id`
- [ ] **`max_tool_calls` parameter** — cap the number of tool invocations per job

## Architecture

```
src/
├── api/
│   ├── types.rs    # Full OpenAI Responses API types with output parsing
│   ├── client.rs   # HTTP client for create/retrieve/poll operations
│   └── mod.rs
├── mcp/
│   ├── server.rs   # FathomServer with 3 MCP tools (rmcp macros)
│   └── mod.rs
├── cli.rs          # clap-derived CLI definition
├── lib.rs          # Library re-exports
└── main.rs         # Entry point: CLI dispatch + MCP server startup
```

## License

MIT
