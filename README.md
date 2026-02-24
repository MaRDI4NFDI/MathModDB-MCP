# MathModDB MCP Server

MathModDB MCP server for semantic ontology exploration and SPARQL querying.

## What it provides

- `Explore_Ontology`: semantic schema discovery (classes/properties/qualifiers).
- `SPARQL_Query`: single SPARQL query execution.
- `Batch_SPARQL_Query`: batch SPARQL query execution.

The server runs as `streamable-http` on port `8000` by default.

## Requirements

- Python `3.12+` (for local runs)
- [uv](https://docs.astral.sh/uv/) (for local runs)
- `OPENAI_API_KEY` (required)
- Docker (for containerized runs)
- Docker BuildKit enabled (required by the Dockerfile build secret)

## Run locally

```bash
uv sync --frozen
export OPENAI_API_KEY=your_key_here
uv run fastmcp run main.py --host 0.0.0.0 --port 8000 --transport streamable-http
```

## Build and run with Docker

The Docker image initializes ontology/vector data during build, so the API key must be provided at build time as a Docker secret.

### Build requirements

- Docker BuildKit enabled (`DOCKER_BUILDKIT=1`)
- `OPENAI_API_KEY` present in your shell environment

### Build image

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=OPENAI_API_KEY,env=OPENAI_API_KEY \
  -t mathmoddb-mcp .
```

### Run container

```bash
docker run --rm --name mathmoddb-mcp -p 8000:8000 -e OPENAI_API_KEY="$OPENAI_API_KEY" mathmoddb-mcp
```

You can also use the included helper script:

```bash
./start.sh
```

## Use with Claude Desktop

After the server is running at `http://127.0.0.1:8000/mcp`, add this MCP entry in your Claude Desktop config:

```json
"MathModDB": {
  "command": "npx",
  "args": [
    "mcp-remote",
    "http://127.0.0.1:8000/mcp"
  ],
  "env": {}
}
```

### Claude Desktop-side prerequisites

- Node.js / npm available locally (for `npx`)
- `mcp-remote` runnable via `npx`
