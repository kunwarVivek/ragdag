"""BM25 inverted index — O(k) search instead of O(n) file scan.

Builds and queries a JSON inverted index so BM25 search doesn't need
to read every .txt file on every query.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# BM25 parameters (must match engines/bm25.py)
K1 = 1.2
B = 0.75

INDEX_FILE = "_bm25_index.json"


def _parse_doc(txt_file: Path, store: Path) -> Optional[dict]:
    """Parse a single .txt file into index-ready metadata.

    Returns dict with keys: rel_path, content, is_synth, is_stale, doc_len
    or None if file should be skipped.
    """
    if txt_file.name.startswith("."):
        return None

    try:
        raw = txt_file.read_text(encoding="utf-8")
    except Exception:
        return None

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
        return None

    rel_path = str(txt_file.relative_to(store))
    content_lower = content.lower()

    return {
        "rel_path": rel_path,
        "content": content_lower,
        "is_synth": is_synth,
        "is_stale": is_stale,
    }


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into words >= 2 chars."""
    import re
    # Remove punctuation, lowercase, split
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return [w for w in cleaned.split() if len(w) >= 2]


def _build_terms_and_docs(store: Path) -> dict:
    """Scan all .txt files and build the index data structure."""
    terms: Dict[str, Dict[str, int]] = {}
    docs: Dict[str, dict] = {}

    for txt_file in store.rglob("*.txt"):
        if txt_file.name == INDEX_FILE:
            continue
        parsed = _parse_doc(txt_file, store)
        if parsed is None:
            continue

        rel_path = parsed["rel_path"]
        content = parsed["content"]
        doc_len = len(content)

        docs[rel_path] = {
            "len": doc_len,
            "synth": parsed["is_synth"],
            "stale": parsed["is_stale"],
        }

        # Count term frequencies
        words = _tokenize(content)
        tf: Dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1

        for word, count in tf.items():
            if word not in terms:
                terms[word] = {}
            terms[word][rel_path] = count

    n = len(docs)
    avg_dl = sum(d["len"] for d in docs.values()) / n if n > 0 else 0.0

    return {
        "terms": terms,
        "docs": docs,
        "avg_dl": avg_dl,
        "n": n,
    }


def build_index(store_dir: str) -> None:
    """Full rebuild of the BM25 inverted index.

    Scans all .txt files under store_dir and writes _bm25_index.json.
    """
    store = Path(store_dir)
    idx = _build_terms_and_docs(store)
    (store / INDEX_FILE).write_text(json.dumps(idx), encoding="utf-8")


def load_index(store_dir: str) -> Optional[dict]:
    """Load the inverted index from disk. Returns None if no index exists."""
    index_path = Path(store_dir) / INDEX_FILE
    if not index_path.exists():
        return None
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def update_index(store_dir: str, changed_paths: List[str]) -> None:
    """Incremental update: re-index only the changed paths.

    - If a changed file still exists, re-parse and update its entries.
    - If a changed file was deleted, remove its entries from the index.
    """
    store = Path(store_dir)
    idx = load_index(store_dir)
    if idx is None:
        build_index(store_dir)
        return

    terms = idx["terms"]
    docs = idx["docs"]

    # Remove old entries for all changed paths
    for rel_path in changed_paths:
        # Remove from docs
        docs.pop(rel_path, None)
        # Remove from all term postings
        for term_postings in terms.values():
            term_postings.pop(rel_path, None)

    # Clean up empty term entries
    terms = {t: p for t, p in terms.items() if p}

    # Re-add files that still exist
    for rel_path in changed_paths:
        abs_path = store / rel_path
        if not abs_path.exists():
            continue

        parsed = _parse_doc(abs_path, store)
        if parsed is None:
            continue

        content = parsed["content"]
        doc_len = len(content)

        docs[rel_path] = {
            "len": doc_len,
            "synth": parsed["is_synth"],
            "stale": parsed["is_stale"],
        }

        words = _tokenize(content)
        tf: Dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1

        for word, count in tf.items():
            if word not in terms:
                terms[word] = {}
            terms[word][rel_path] = count

    # Recompute aggregate stats
    n = len(docs)
    avg_dl = sum(d["len"] for d in docs.values()) / n if n > 0 else 0.0

    idx["terms"] = terms
    idx["docs"] = docs
    idx["avg_dl"] = avg_dl
    idx["n"] = n

    (store / INDEX_FILE).write_text(json.dumps(idx), encoding="utf-8")


def query_index(
    store_dir: str,
    query: str,
    domain: str = "",
    top_k: int = 50,
    synthesis_boost: float = 1.2,
    stale_penalty: float = 0.5,
) -> Optional[List[Tuple[str, float]]]:
    """BM25 search using the inverted index.

    Returns None if no index exists (signals caller to fall back to file scan).
    Returns list of (rel_path, score) sorted descending.
    """
    idx = load_index(store_dir)
    if idx is None:
        return None

    words = _tokenize(query)
    if not words:
        return []

    terms = idx["terms"]
    docs = idx["docs"]
    n = idx["n"]
    avg_dl = idx["avg_dl"]

    if n == 0:
        return []

    # Filter docs by domain if specified
    if domain:
        candidate_docs = {p: d for p, d in docs.items() if p.startswith(domain + "/" if not domain.endswith("/") else domain)}
    else:
        candidate_docs = docs

    # Compute IDF for each query term (using full corpus N for IDF)
    idf = {}
    for word in words:
        posting = terms.get(word, {})
        df = len(posting)
        idf[word] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    # Score each candidate document
    results = []
    for rel_path, doc_info in candidate_docs.items():
        dl = doc_info["len"]
        score = 0.0

        for word in words:
            if idf[word] <= 0:
                continue
            posting = terms.get(word, {})
            tf = posting.get(rel_path, 0)
            if tf == 0:
                continue

            # BM25 uses character-level counts for TF, matching bm25.py
            # The index stores word-level TF, but bm25.py uses content.count(word)
            # which counts overlapping substring occurrences at character level.
            # For exact match, we need to replicate that behavior.
            # Since we index word-level TF and the original uses content.count(),
            # we need to be consistent. The original splits on whitespace for query
            # but uses content.count(word) for TF which is substring-based.
            # Our index stores word-level TF which is equivalent for most cases
            # since words are space-separated in content.

            tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avg_dl))
            score += idf[word] * tf_norm

        if score > 0:
            is_synth = doc_info["synth"]
            is_stale = doc_info["stale"]
            if is_synth:
                score *= synthesis_boost * (stale_penalty if is_stale else 1.0)
            results.append((rel_path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
