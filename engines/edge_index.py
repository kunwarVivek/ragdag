"""Per-node edge index for O(degree) graph lookups.

Builds and maintains a JSON index (_edge_index.json) from the .edges TSV file,
enabling fast neighbor/trace/ask lookups without scanning the entire edge list.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


INDEX_FILENAME = "_edge_index.json"


def build_edge_index(store_dir: str) -> None:
    """Read .edges TSV and build _edge_index.json with bidirectional entries."""
    store = Path(store_dir)
    edges_file = store / ".edges"
    index: Dict[str, List[dict]] = {}

    if edges_file.exists():
        for line in edges_file.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            source, target, edge_type = parts[0], parts[1], parts[2]
            metadata = parts[3] if len(parts) > 3 else ""

            # Outgoing entry for source
            index.setdefault(source, []).append({
                "direction": "outgoing",
                "node": target,
                "edge_type": edge_type,
                "metadata": metadata,
            })
            # Incoming entry for target
            index.setdefault(target, []).append({
                "direction": "incoming",
                "node": source,
                "edge_type": edge_type,
                "metadata": metadata,
            })

    (store / INDEX_FILENAME).write_text(json.dumps(index, indent=2))


def load_edge_index(store_dir: str) -> Optional[dict]:
    """Load _edge_index.json if it exists, otherwise return None."""
    index_path = Path(store_dir) / INDEX_FILENAME
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text())


def lookup_edges(store_dir: str, node_path: str) -> Optional[List[dict]]:
    """Look up edges for a node. Returns None if no index file exists."""
    index = load_edge_index(store_dir)
    if index is None:
        return None
    return index.get(node_path, [])


def append_edge(store_dir: str, source: str, target: str, edge_type: str, metadata: str) -> None:
    """Add a single edge to the index (both directions). Does NOT write to .edges."""
    store = Path(store_dir)
    index_path = store / INDEX_FILENAME

    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {}

    index.setdefault(source, []).append({
        "direction": "outgoing",
        "node": target,
        "edge_type": edge_type,
        "metadata": metadata,
    })
    index.setdefault(target, []).append({
        "direction": "incoming",
        "node": source,
        "edge_type": edge_type,
        "metadata": metadata,
    })

    index_path.write_text(json.dumps(index, indent=2))
