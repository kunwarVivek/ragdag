"""FastAPI HTTP server for ragdag."""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    print("FastAPI required: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

import ragdag

app = FastAPI(title="ragdag", version="1.0.0", description="Knowledge graph engine API")

# Global dag instance
_dag: Optional[ragdag.RagDag] = None


def get_dag() -> ragdag.RagDag:
    global _dag
    if _dag is None:
        store_path = os.environ.get("RAGDAG_STORE", ".")
        _dag = ragdag.open(store_path)
    return _dag


# Request/Response models
class AddRequest(BaseModel):
    path: str
    domain: Optional[str] = None
    embed: bool = True


class SearchRequest(BaseModel):
    query: str
    mode: Optional[str] = "hybrid"
    domain: Optional[str] = None
    top: Optional[int] = 10


class AskRequest(BaseModel):
    question: str
    domain: Optional[str] = None
    use_llm: bool = True


class LinkRequest(BaseModel):
    source: str
    target: str
    edge_type: str = "references"


# Endpoints
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/add")
def add_documents(req: AddRequest):
    dag = get_dag()
    try:
        result = dag.add(req.path, domain=req.domain, embed=req.embed)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_corpus(req: SearchRequest):
    dag = get_dag()
    results = dag.search(req.query, mode=req.mode, domain=req.domain, top=req.top)
    return [
        {"path": r.path, "score": r.score, "content": r.content, "domain": r.domain}
        for r in results
    ]


@app.post("/ask")
def ask_question(req: AskRequest):
    dag = get_dag()
    result = dag.ask(req.question, domain=req.domain, use_llm=req.use_llm)
    return {
        "answer": result.answer,
        "context": result.context,
        "sources": result.sources,
    }


@app.get("/graph")
def graph_summary(domain: Optional[str] = None):
    dag = get_dag()
    stats = dag.graph(domain=domain)
    return {
        "domains": stats.domains,
        "documents": stats.documents,
        "chunks": stats.chunks,
        "edges": stats.edges,
        "edge_types": stats.edge_types,
    }


@app.get("/neighbors/{node_path:path}")
def get_neighbors(node_path: str):
    dag = get_dag()
    return dag.neighbors(node_path)


@app.post("/link")
def create_link(req: LinkRequest):
    dag = get_dag()
    dag.link(req.source, req.target, req.edge_type)
    return {"status": "ok"}


@app.get("/trace/{node_path:path}")
def trace_provenance(node_path: str):
    dag = get_dag()
    return dag.trace(node_path)


class RelateRequest(BaseModel):
    domain: Optional[str] = None
    threshold: float = 0.8


@app.post("/relate")
def compute_relations(req: RelateRequest):
    dag = get_dag()
    try:
        dag.relate(domain=req.domain, threshold=req.threshold)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run(host: str = "0.0.0.0", port: int = 8420):
    """Start the HTTP server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
