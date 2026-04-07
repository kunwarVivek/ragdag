"""BM25 scoring over flat .txt files -- no SQLite needed."""

import math
from pathlib import Path
from typing import List, Tuple

# BM25 parameters (Okapi defaults)
K1 = 1.2   # term frequency saturation
B = 0.75   # document length normalization


def bm25_search(
    store_dir: str,
    query: str,
    domain: str = "",
    top_k: int = 50,
    synthesis_boost: float = 1.2,
    stale_penalty: float = 0.5,
) -> List[Tuple[str, float]]:
    """BM25-scored keyword search over flat .txt chunk files.

    Returns list of (relative_path, score) sorted descending.
    """
    # Fast path: use inverted index if available
    try:
        from .bm25_index import query_index
        indexed_results = query_index(store_dir, query, domain, top_k, synthesis_boost, stale_penalty)
        if indexed_results is not None:
            return indexed_results
    except ImportError:
        pass

    # Slow path: scan files (fallback when no index exists)
    store = Path(store_dir)
    search_path = store / domain if domain else store

    if not search_path.exists():
        return []

    words = [w.lower() for w in query.split() if len(w) >= 2]
    if not words:
        return []

    # Phase 1: Build corpus stats (one pass)
    docs = []  # [(rel_path, content, is_synth, is_stale)]
    for txt_file in search_path.rglob("*.txt"):
        if txt_file.name.startswith("."):
            continue
        try:
            raw = txt_file.read_text(encoding="utf-8")
        except Exception:
            continue

        is_synth = txt_file.name.startswith("_")
        is_stale = False
        content = raw

        if is_synth and raw.startswith("---\n"):
            end = raw.find("\n---\n", 4)
            if end != -1:
                frontmatter = raw[:end]
                content = raw[end + 5:]
                is_stale = "stale: true" in frontmatter

        if not content.strip():
            continue

        rel_path = str(txt_file.relative_to(store))
        docs.append((rel_path, content.lower(), is_synth, is_stale))

    if not docs:
        return []

    n = len(docs)
    avg_dl = sum(len(d[1]) for d in docs) / n

    # Phase 2: Compute IDF for each query term
    df = {}
    for word in words:
        df[word] = sum(1 for _, content, _, _ in docs if word in content)

    idf = {}
    for word in words:
        idf[word] = math.log((n - df[word] + 0.5) / (df[word] + 0.5) + 1.0)

    # Phase 3: Score each document
    results = []
    for rel_path, content, is_synth, is_stale in docs:
        dl = len(content)
        score = 0.0

        for word in words:
            if idf[word] <= 0:
                continue
            tf = content.count(word)
            if tf == 0:
                continue
            tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avg_dl))
            score += idf[word] * tf_norm

        if score > 0:
            if is_synth:
                score *= synthesis_boost * (stale_penalty if is_stale else 1.0)
            results.append((rel_path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
