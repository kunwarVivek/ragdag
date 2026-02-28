"""MCP server for ragdag — built with FastMCP."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP, Context

# Add project root to path for SDK import
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))

import ragdag

# --- Server setup ---

mcp = FastMCP(
    "ragdag",
    instructions=(
        "ragdag is a knowledge graph engine that runs on flat files and bash. "
        "Use ragdag_search to find relevant documents, ragdag_ask to answer "
        "questions with RAG, ragdag_add to ingest new documents, ragdag_graph "
        "for knowledge graph stats, and ragdag_neighbors to explore connections."
    ),
)


def _get_dag() -> ragdag.RagDag:
    store_path = os.environ.get("RAGDAG_STORE", ".")
    return ragdag.open(store_path)


# --- Tools ---


@mcp.tool
async def ragdag_search(
    query: str,
    mode: str = "hybrid",
    domain: Optional[str] = None,
    top: int = 10,
    ctx: Context = None,
) -> str:
    """Search the ragdag knowledge base.

    Args:
        query: Search query string
        mode: Search mode — "keyword" (pure bash), "vector" (embeddings), or "hybrid" (both)
        domain: Optional domain filter to narrow search scope
        top: Number of results to return (default 10)
    """
    if ctx:
        await ctx.info(f"Searching for: {query} (mode={mode}, domain={domain})")

    dag = _get_dag()
    results = dag.search(query, mode=mode, domain=domain, top=top)

    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        preview = r.content[:200].replace("\n", " ") if r.content else ""
        parts.append(f"{i}. **{r.path}** (score: {r.score:.4f})\n   {preview}")

    return "\n\n".join(parts)


@mcp.tool
async def ragdag_ask(
    question: str,
    domain: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Ask a question and get an answer using RAG (retrieval-augmented generation).

    Searches the knowledge base, assembles relevant context, and optionally
    generates an answer using the configured LLM.

    Args:
        question: The question to answer
        domain: Optional domain filter to narrow search scope
    """
    if ctx:
        await ctx.info(f"Answering: {question}")

    dag = _get_dag()
    result = dag.ask(question, domain=domain)

    answer_text = result.answer or result.context
    if result.sources:
        sources_text = "\n".join(f"- {s}" for s in result.sources)
        return f"{answer_text}\n\n**Sources:**\n{sources_text}"
    return answer_text


@mcp.tool
async def ragdag_add(
    path: str,
    domain: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Ingest documents into the ragdag knowledge base.

    Parses, chunks, and stores documents. Optionally generates embeddings
    if an embedding provider is configured.

    Args:
        path: Path to a file or directory to ingest
        domain: Domain name for organization (or "auto" for rule-based)
    """
    if ctx:
        await ctx.info(f"Ingesting: {path} (domain={domain})")

    dag = _get_dag()
    result = dag.add(path, domain=domain)
    return json.dumps(result, indent=2)


@mcp.tool
async def ragdag_graph(
    domain: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Get knowledge graph summary statistics.

    Returns counts of domains, documents, chunks, edges, and edge type breakdown.

    Args:
        domain: Optional domain filter
    """
    dag = _get_dag()
    stats = dag.graph(domain=domain)
    return (
        f"Domains: {stats.domains}\n"
        f"Documents: {stats.documents}\n"
        f"Chunks: {stats.chunks}\n"
        f"Edges: {stats.edges}\n"
        f"Edge types: {json.dumps(stats.edge_types, indent=2)}"
    )


@mcp.tool
async def ragdag_neighbors(
    node: str,
    ctx: Context = None,
) -> str:
    """Get connected nodes for a given node in the knowledge graph.

    Shows all incoming and outgoing edges with their types.

    Args:
        node: Node path (e.g., "docs/auth/01.txt")
    """
    dag = _get_dag()
    neighbors = dag.neighbors(node)

    if not neighbors:
        return f"No neighbors found for: {node}"

    parts = []
    for n in neighbors:
        arrow = "→" if n["direction"] == "outgoing" else "←"
        parts.append(f"  {arrow} {n['node']} [{n['edge_type']}]")

    return f"Neighbors of {node}:\n" + "\n".join(parts)


@mcp.tool
async def ragdag_trace(
    node: str,
    ctx: Context = None,
) -> str:
    """Get provenance chain for a node in the knowledge graph.

    Walks backward through chunked_from/derived_via edges to show
    the full provenance from chunk to source document.

    Args:
        node: Node path (e.g., "docs/auth/01.txt")
    """
    dag = _get_dag()
    chain = dag.trace(node)

    if not chain:
        return f"No provenance found for: {node}"

    parts = []
    for i, step in enumerate(chain):
        indent = "  " * i
        if step["parent"]:
            parts.append(f"{indent}├── {step['node']} [{step['edge_type']}]")
        else:
            parts.append(f"{indent}└── {step['node']} (origin)")

    return f"Provenance of {node}:\n" + "\n".join(parts)


def run():
    """Start MCP server (stdio transport)."""
    mcp.run()


def run_http(host: str = "0.0.0.0", port: int = 8420):
    """Start MCP server (HTTP transport)."""
    mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    run()
