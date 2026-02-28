# ragdag

`ragdag` is a flat-file knowledge graph engine with three interfaces:

- A Bash CLI (`./ragdag`)
- A Python SDK (`import ragdag`)
- Server adapters (FastAPI HTTP + FastMCP)

It stores everything in a local `.ragdag/` directory (chunks, edges, config, and processing metadata) and keeps the workflow simple: ingest files, search, ask, inspect graph links, and maintain store integrity.

## Repository Layout

- `ragdag` - CLI entrypoint script
- `lib/*.sh` - Bash command implementations
- `sdk/ragdag/` - Python SDK (`RagDag`)
- `engines/` - embedding/search/LLM helpers
- `server/` - HTTP API and MCP server
- `tests/` - pytest and bats test suites
- `docs/` - project documentation

## Quick Start

### CLI

```bash
./ragdag init
./ragdag add ./docs
./ragdag search "knowledge graph" --top 5
./ragdag ask "What does this project do?" --no-llm
./ragdag graph
```

### Python SDK

```python
import ragdag

dag = ragdag.init(".")
dag.add("./docs")
results = dag.search("knowledge graph", mode="hybrid", top=5)
answer = dag.ask("What does this project do?", use_llm=False)
```

## CLI Commands

Primary commands exposed by `./ragdag`:

- `init`, `add`, `search`, `ask`
- `graph`, `neighbors`, `trace`, `relate`, `link`
- `config`, `serve`, `verify`, `repair`, `gc`, `reindex`

Run `./ragdag help` for full command and flag details.

## Server Interfaces

### HTTP API

Implemented in `server/api.py`.

- `GET /health`
- `POST /add`
- `POST /search`
- `POST /ask`
- `GET /graph`
- `GET /neighbors/{node_path}`
- `POST /link`
- `GET /trace/{node_path}`
- `POST /relate`

### MCP Server

Implemented in `server/mcp.py`.

Tools:

- `ragdag_search`
- `ragdag_ask`
- `ragdag_add`
- `ragdag_graph`
- `ragdag_neighbors`
- `ragdag_trace`

## Development

Install editable package with extras as needed:

```bash
python3 -m pip install -e ".[all]"
```

For testing workflows and commands, see `docs/TESTING.md`.
