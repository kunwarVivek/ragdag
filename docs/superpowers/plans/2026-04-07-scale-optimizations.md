# Scale Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ragdag handle 50K+ file repos by eliminating O(n) bottlenecks in ingest, search, and graph operations.

**Architecture:** Four independent optimizations: (1) append-only embeddings.bin with in-place header updates, (2) BM25 inverted index cached to JSON at ingest time, (3) per-node edge index files for O(degree) graph lookups, (4) ANN-based relate using random projection LSH instead of O(n^2) pairwise cosine.

**Tech Stack:** Python, numpy, JSON, existing ragdag flat-file format

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `engines/embeddings_v2.py` | Append-only write + in-place header update |
| Create | `engines/bm25_index.py` | Build/load/query inverted index from JSON |
| Create | `engines/edge_index.py` | Build/load/query per-node edge index |
| Create | `engines/ann.py` | Approximate nearest neighbors via random projection LSH |
| Modify | `engines/embeddings.py` | Delegate to v2 functions, keep backward compat |
| Modify | `engines/bm25.py` | Use inverted index when available, fall back to scan |
| Modify | `engines/relate_cli.py` | Use ANN instead of O(n^2) pairwise |
| Modify | `sdk/ragdag/core.py` | Use edge index for neighbors/trace/ask, build indexes on add |
| Create | `tests/test_embeddings_v2.py` | Append-only embedding tests |
| Create | `tests/test_bm25_index.py` | Inverted index build/query tests |
| Create | `tests/test_edge_index.py` | Edge index build/query tests |
| Create | `tests/test_ann.py` | ANN accuracy and performance tests |
| Modify | `tests/test_pipeline_integration.py` | Integration tests for indexed paths |

---

### Task 1: Append-Only Embeddings

Currently `write_embeddings` loads ALL existing vectors, filters, concatenates, and rewrites the entire file. At 200K chunks × 1536 dims, that's ~1.2 GB per ingest.

Fix: Append new vectors to the end of the file. Only update the header (count field) in place. Only rewrite when removing/replacing existing paths.

**Files:**
- Create: `engines/embeddings_v2.py`
- Create: `tests/test_embeddings_v2.py`
- Modify: `engines/embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embeddings_v2.py
"""Tests for append-only embedding writes."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "engines"))

from embeddings import HEADER_SIZE, MAGIC, FORMAT_VERSION, model_hash, load_embeddings


class TestAppendOnly:

    def test_append_only_does_not_rewrite_existing_bytes(self, tmp_path):
        """Appending new vectors should not rewrite the existing binary data."""
        from embeddings_v2 import append_embeddings

        dims = 8
        # First write: create file with 3 vectors
        v1 = np.random.randn(3, dims).astype(np.float32).tolist()
        p1 = ["a/01.txt", "a/02.txt", "a/03.txt"]
        append_embeddings(str(tmp_path), v1, p1, dims, "model-a")

        # Record file content after first write
        original_bytes = (tmp_path / "embeddings.bin").read_bytes()
        original_data_section = original_bytes[HEADER_SIZE:]  # just the vectors

        # Append 2 more vectors
        v2 = np.random.randn(2, dims).astype(np.float32).tolist()
        p2 = ["b/01.txt", "b/02.txt"]
        append_embeddings(str(tmp_path), v2, p2, dims, "model-a")

        # The first 3 vectors in the file should be identical to original
        new_bytes = (tmp_path / "embeddings.bin").read_bytes()
        assert new_bytes[HEADER_SIZE:HEADER_SIZE + len(original_data_section)] == original_data_section

    def test_append_only_updates_header_count(self, tmp_path):
        """Header count field should reflect total vectors after append."""
        from embeddings_v2 import append_embeddings

        dims = 4
        v1 = [[0.1, 0.2, 0.3, 0.4]]
        append_embeddings(str(tmp_path), v1, ["a.txt"], dims, "model-a")

        v2 = [[0.5, 0.6, 0.7, 0.8]]
        append_embeddings(str(tmp_path), v2, ["b.txt"], dims, "model-a")

        vecs, d, count, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert count == 2
        assert d == dims
        np.testing.assert_allclose(vecs[0], v1[0])
        np.testing.assert_allclose(vecs[1], v2[0])

    def test_append_creates_new_file_if_missing(self, tmp_path):
        """First append should create the file from scratch."""
        from embeddings_v2 import append_embeddings

        dims = 4
        append_embeddings(str(tmp_path), [[1, 2, 3, 4]], ["x.txt"], dims, "m")

        vecs, d, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert c == 1
        assert d == dims

    def test_append_manifest_grows(self, tmp_path):
        """Manifest should grow with each append, not be rewritten."""
        from embeddings_v2 import append_embeddings
        from embeddings import _read_manifest_paths

        dims = 4
        append_embeddings(str(tmp_path), [[1, 2, 3, 4]], ["a.txt"], dims, "m")
        append_embeddings(str(tmp_path), [[5, 6, 7, 8]], ["b.txt"], dims, "m")

        paths = _read_manifest_paths(str(tmp_path / "manifest.tsv"))
        assert paths == ["a.txt", "b.txt"]

    def test_append_empty_vectors_is_noop(self, tmp_path):
        """Appending empty list should not create or modify files."""
        from embeddings_v2 import append_embeddings

        append_embeddings(str(tmp_path), [], [], 4, "m")
        assert not (tmp_path / "embeddings.bin").exists()

    def test_append_with_content_hashes(self, tmp_path):
        """Content hashes should be written to manifest on append."""
        from embeddings_v2 import append_embeddings
        from embeddings import load_manifest

        dims = 4
        append_embeddings(str(tmp_path), [[1, 2, 3, 4]], ["a.txt"], dims, "m",
                          chunk_texts=["hello"])
        append_embeddings(str(tmp_path), [[5, 6, 7, 8]], ["b.txt"], dims, "m",
                          chunk_texts=["world"])

        manifest = load_manifest(str(tmp_path / "manifest.tsv"))
        assert len(manifest) == 2
        assert manifest[0][4] != ""
        assert manifest[1][4] != ""
        assert manifest[0][4] != manifest[1][4]


class TestAppendWithReplacement:

    def test_replace_triggers_full_rewrite(self, tmp_path):
        """When replacing existing paths, must fall back to full rewrite."""
        from embeddings_v2 import append_embeddings, needs_rewrite
        from embeddings import _read_manifest_paths

        dims = 4
        append_embeddings(str(tmp_path), [[1, 2, 3, 4]], ["a.txt"], dims, "m")
        append_embeddings(str(tmp_path), [[5, 6, 7, 8]], ["b.txt"], dims, "m")

        # Check if replacing a.txt needs rewrite
        assert needs_rewrite(str(tmp_path), ["a.txt"]) is True
        # Adding new path c.txt does not need rewrite
        assert needs_rewrite(str(tmp_path), ["c.txt"]) is False

    def test_full_rewrite_still_works(self, tmp_path):
        """Full rewrite path should produce correct output."""
        from embeddings_v2 import append_embeddings
        from embeddings import load_embeddings, _read_manifest_paths

        dims = 4
        append_embeddings(str(tmp_path), [[1, 2, 3, 4]], ["a.txt"], dims, "m")
        append_embeddings(str(tmp_path), [[5, 6, 7, 8]], ["b.txt"], dims, "m")
        # Replace a.txt with new vector
        append_embeddings(str(tmp_path), [[9, 9, 9, 9]], ["a.txt"], dims, "m")

        vecs, _, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert c == 2  # b + replaced a
        paths = _read_manifest_paths(str(tmp_path / "manifest.tsv"))
        assert set(paths) == {"a.txt", "b.txt"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_embeddings_v2.py -v`
Expected: `ModuleNotFoundError: No module named 'embeddings_v2'`

- [ ] **Step 3: Implement append-only embeddings**

```python
# engines/embeddings_v2.py
"""Append-only embedding writes — O(1) ingest for new vectors."""

import struct
from pathlib import Path
from typing import List, Optional

import numpy as np

from .embeddings import (
    MAGIC, FORMAT_VERSION, HEADER_SIZE, model_hash, _content_hash,
    load_embeddings, _read_manifest_paths, load_manifest, write_embeddings,
)


def needs_rewrite(output_dir: str, new_paths: List[str]) -> bool:
    """Check if any new paths already exist in the manifest (requires rewrite)."""
    manifest_path = Path(output_dir) / "manifest.tsv"
    if not manifest_path.exists():
        return False
    existing = set(_read_manifest_paths(str(manifest_path)))
    return bool(existing & set(new_paths))


def append_embeddings(
    output_dir: str,
    vectors: List[List[float]],
    chunk_paths: List[str],
    dimensions: int,
    model_name_str: str,
    chunk_texts: Optional[List[str]] = None,
) -> None:
    """Append vectors to embeddings.bin without rewriting existing data.

    If any chunk_paths already exist in the manifest, falls back to full
    rewrite via write_embeddings() for correctness.
    """
    if not vectors:
        return

    out = Path(output_dir)
    bin_path = out / "embeddings.bin"
    manifest_path = out / "manifest.tsv"

    # If replacing existing paths, fall back to full rewrite
    if needs_rewrite(output_dir, chunk_paths):
        write_embeddings(
            output_dir, vectors, chunk_paths, dimensions,
            model_name_str, append=True, chunk_texts=chunk_texts,
        )
        return

    # Content hashes
    if chunk_texts is not None:
        content_hashes = [_content_hash(t) for t in chunk_texts]
    else:
        content_hashes = [""] * len(chunk_paths)

    arr = np.array(vectors, dtype=np.float32)

    if not bin_path.exists():
        # Create new file with header + vectors
        mhash = model_hash(model_name_str)
        count = len(vectors)

        with open(bin_path, "wb") as f:
            f.write(struct.pack("I", MAGIC))
            f.write(struct.pack("I", FORMAT_VERSION))
            f.write(struct.pack("I", dimensions))
            f.write(struct.pack("I", count))
            f.write(struct.pack("I", mhash))
            f.write(b"\x00" * 12)
            f.write(arr.tobytes())

        # Write manifest
        with open(manifest_path, "w") as f:
            f.write("# relative_chunk_path\tindex\tbyte_offset\tdimensions\tcontent_hash\n")
            for i, path in enumerate(chunk_paths):
                offset = HEADER_SIZE + i * dimensions * 4
                f.write(f"{path}\t{i}\t{offset}\t{dimensions}\t{content_hashes[i]}\n")
    else:
        # Append: read current count, append vectors, update header count
        with open(bin_path, "r+b") as f:
            # Read existing count
            f.seek(12)
            old_count = struct.unpack("I", f.read(4))[0]

            # Read existing dims to verify compatibility
            f.seek(8)
            existing_dims = struct.unpack("I", f.read(4))[0]
            if existing_dims != dimensions:
                raise ValueError(
                    f"Dimension mismatch: existing={existing_dims}, new={dimensions}"
                )

            new_count = old_count + len(vectors)

            # Append vectors at end of file
            f.seek(0, 2)  # seek to end
            f.write(arr.tobytes())

            # Update count in header
            f.seek(12)
            f.write(struct.pack("I", new_count))

        # Append to manifest
        with open(manifest_path, "a") as f:
            for i, path in enumerate(chunk_paths):
                idx = old_count + i
                offset = HEADER_SIZE + idx * dimensions * 4
                f.write(f"{path}\t{idx}\t{offset}\t{dimensions}\t{content_hashes[i]}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_embeddings_v2.py -v`
Expected: All PASS

- [ ] **Step 5: Wire into embeddings.py**

In `engines/embeddings.py`, modify `write_embeddings` to use append-only path when possible. Add at the top of the function:

```python
    # Fast path: append-only when adding new paths (no replacements)
    if append and chunk_paths:
        from .embeddings_v2 import needs_rewrite, append_embeddings
        if not needs_rewrite(output_dir, chunk_paths):
            append_embeddings(output_dir, vectors, chunk_paths, dimensions,
                              model_name_str, chunk_texts=chunk_texts)
            return
```

- [ ] **Step 6: Run all embedding tests**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_embeddings.py tests/test_embeddings_v2.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add engines/embeddings_v2.py engines/embeddings.py tests/test_embeddings_v2.py
git commit -m "feat: append-only embeddings — O(1) ingest for new vectors"
```

---

### Task 2: BM25 Inverted Index

Currently `bm25_search` reads every `.txt` file on every query via `rglob("*.txt")`. At 200K files, that's 200K file opens per search.

Fix: Build an inverted index at ingest time (`_bm25_index.json`), mapping terms → document entries. Rebuild on `reindex`.

**Files:**
- Create: `engines/bm25_index.py`
- Create: `tests/test_bm25_index.py`
- Modify: `engines/bm25.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bm25_index.py
"""Tests for BM25 inverted index."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "engines"))


class TestBuildIndex:

    def _make_store(self, tmp_path, docs: dict):
        """Create .txt files, return store_dir."""
        store = tmp_path / ".ragdag" / "domain"
        store.mkdir(parents=True)
        for name, content in docs.items():
            (store / f"{name}.txt").write_text(content)
        return str(tmp_path / ".ragdag")

    def test_build_creates_index_file(self, tmp_path):
        from bm25_index import build_index

        store_dir = self._make_store(tmp_path, {"01": "hello world"})
        build_index(store_dir)
        assert (Path(store_dir) / "_bm25_index.json").exists()

    def test_build_index_contains_terms(self, tmp_path):
        from bm25_index import build_index, load_index

        store_dir = self._make_store(tmp_path, {"01": "python programming language"})
        build_index(store_dir)
        index = load_index(store_dir)
        assert "python" in index["terms"]
        assert "programming" in index["terms"]

    def test_build_index_tracks_doc_lengths(self, tmp_path):
        from bm25_index import build_index, load_index

        store_dir = self._make_store(tmp_path, {
            "01": "short",
            "02": "this is a much longer document with more words",
        })
        build_index(store_dir)
        index = load_index(store_dir)
        assert len(index["docs"]) == 2
        assert index["docs"]["domain/02.txt"]["len"] > index["docs"]["domain/01.txt"]["len"]

    def test_build_index_stores_term_frequency(self, tmp_path):
        from bm25_index import build_index, load_index

        store_dir = self._make_store(tmp_path, {"01": "python python python java"})
        build_index(store_dir)
        index = load_index(store_dir)
        # "python" appears 3 times in doc 01
        assert index["terms"]["python"]["domain/01.txt"] == 3

    def test_build_index_skips_short_words(self, tmp_path):
        from bm25_index import build_index, load_index

        store_dir = self._make_store(tmp_path, {"01": "a b cd efgh"})
        build_index(store_dir)
        index = load_index(store_dir)
        assert "a" not in index["terms"]
        assert "b" not in index["terms"]
        assert "cd" in index["terms"]

    def test_synthesis_frontmatter_stripped(self, tmp_path):
        from bm25_index import build_index, load_index

        store = tmp_path / ".ragdag" / "domain"
        store.mkdir(parents=True)
        (store / "_summary.txt").write_text(
            "---\ntype: summary\nstale: false\n---\npython overview"
        )
        store_dir = str(tmp_path / ".ragdag")
        build_index(store_dir)
        index = load_index(store_dir)
        # "type" and "summary" from frontmatter should not be indexed
        assert "python" in index["terms"]
        assert index["docs"]["domain/_summary.txt"]["synth"] is True


class TestQueryIndex:

    def _make_indexed_store(self, tmp_path, docs: dict):
        from bm25_index import build_index
        store = tmp_path / ".ragdag" / "domain"
        store.mkdir(parents=True)
        for name, content in docs.items():
            (store / f"{name}.txt").write_text(content)
        store_dir = str(tmp_path / ".ragdag")
        build_index(store_dir)
        return store_dir

    def test_query_returns_scored_results(self, tmp_path):
        from bm25_index import query_index

        store_dir = self._make_indexed_store(tmp_path, {
            "01": "python programming",
            "02": "java programming",
        })
        results = query_index(store_dir, "python")
        assert len(results) == 1
        assert results[0][0].endswith("01.txt")
        assert results[0][1] > 0

    def test_query_idf_prefers_rare_terms(self, tmp_path):
        from bm25_index import query_index

        store_dir = self._make_indexed_store(tmp_path, {
            "01": "common word common word",
            "02": "common word common word",
            "03": "common word rare_term",
        })
        results = query_index(store_dir, "rare_term")
        assert results[0][0].endswith("03.txt")

    def test_query_empty_returns_empty(self, tmp_path):
        from bm25_index import query_index

        store_dir = self._make_indexed_store(tmp_path, {"01": "hello"})
        assert query_index(store_dir, "nonexistent") == []

    def test_query_no_index_returns_none(self, tmp_path):
        from bm25_index import query_index

        store_dir = str(tmp_path)
        result = query_index(store_dir, "test")
        assert result is None  # signals caller to fall back to file scan


class TestIncrementalUpdate:

    def test_update_adds_new_doc(self, tmp_path):
        from bm25_index import build_index, update_index, load_index

        store = tmp_path / ".ragdag" / "domain"
        store.mkdir(parents=True)
        (store / "01.txt").write_text("python programming")
        store_dir = str(tmp_path / ".ragdag")
        build_index(store_dir)

        # Add new doc and update incrementally
        (store / "02.txt").write_text("java programming")
        update_index(store_dir, ["domain/02.txt"])

        index = load_index(store_dir)
        assert "domain/02.txt" in index["docs"]
        assert "java" in index["terms"]

    def test_update_replaces_existing_doc(self, tmp_path):
        from bm25_index import build_index, update_index, load_index

        store = tmp_path / ".ragdag" / "domain"
        store.mkdir(parents=True)
        (store / "01.txt").write_text("python programming")
        store_dir = str(tmp_path / ".ragdag")
        build_index(store_dir)

        # Modify doc and update
        (store / "01.txt").write_text("java programming")
        update_index(store_dir, ["domain/01.txt"])

        index = load_index(store_dir)
        assert "java" in index["terms"]
        # python should be gone (only doc had it)
        assert "python" not in index["terms"] or "domain/01.txt" not in index["terms"].get("python", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_bm25_index.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement BM25 inverted index**

```python
# engines/bm25_index.py
"""BM25 inverted index — build at ingest, query at search time."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math


def _tokenize(text: str) -> List[str]:
    """Lowercase and split, filtering words < 2 chars."""
    return [w for w in text.lower().split() if len(w) >= 2]


def _strip_frontmatter(text: str) -> Tuple[str, bool, bool]:
    """Strip YAML frontmatter from synthesis nodes.

    Returns (content, is_synth_from_name, is_stale).
    """
    is_stale = False
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            frontmatter = text[:end]
            is_stale = "stale: true" in frontmatter
            return text[end + 5:], True, is_stale
    return text, False, False


def build_index(store_dir: str) -> None:
    """Build a full BM25 inverted index from all .txt files in the store."""
    store = Path(store_dir)
    index = {"terms": {}, "docs": {}, "avg_dl": 0.0, "n": 0}

    total_len = 0
    for txt_file in store.rglob("*.txt"):
        if txt_file.name.startswith("."):
            continue
        try:
            raw = txt_file.read_text(encoding="utf-8")
        except Exception:
            continue

        is_synth = txt_file.name.startswith("_")
        content, _, is_stale = _strip_frontmatter(raw) if is_synth else (raw, False, False)

        if not content.strip():
            continue

        rel_path = str(txt_file.relative_to(store))
        words = _tokenize(content)
        doc_len = len(content)

        index["docs"][rel_path] = {
            "len": doc_len,
            "synth": is_synth,
            "stale": is_stale,
        }

        # Count term frequencies
        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1

        for word, count in tf.items():
            if word not in index["terms"]:
                index["terms"][word] = {}
            index["terms"][word][rel_path] = count

        total_len += doc_len

    n = len(index["docs"])
    index["n"] = n
    index["avg_dl"] = total_len / n if n > 0 else 0.0

    with open(store / "_bm25_index.json", "w") as f:
        json.dump(index, f)


def load_index(store_dir: str) -> Optional[Dict]:
    """Load the inverted index. Returns None if not found."""
    idx_path = Path(store_dir) / "_bm25_index.json"
    if not idx_path.exists():
        return None
    with open(idx_path) as f:
        return json.load(f)


def update_index(store_dir: str, changed_paths: List[str]) -> None:
    """Incrementally update the index for changed/new documents."""
    store = Path(store_dir)
    index = load_index(store_dir)
    if index is None:
        build_index(store_dir)
        return

    # Remove old entries for changed paths
    for rel_path in changed_paths:
        if rel_path in index["docs"]:
            del index["docs"][rel_path]
        # Remove from term postings
        for term in list(index["terms"]):
            if rel_path in index["terms"][term]:
                del index["terms"][term][rel_path]
                if not index["terms"][term]:
                    del index["terms"][term]

    # Re-index changed paths
    for rel_path in changed_paths:
        full_path = store / rel_path
        if not full_path.exists():
            continue
        try:
            raw = full_path.read_text(encoding="utf-8")
        except Exception:
            continue

        is_synth = full_path.name.startswith("_")
        content, _, is_stale = _strip_frontmatter(raw) if is_synth else (raw, False, False)

        if not content.strip():
            continue

        words = _tokenize(content)
        doc_len = len(content)

        index["docs"][rel_path] = {
            "len": doc_len,
            "synth": is_synth,
            "stale": is_stale,
        }

        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1

        for word, count in tf.items():
            if word not in index["terms"]:
                index["terms"][word] = {}
            index["terms"][word][rel_path] = count

    # Recalculate stats
    n = len(index["docs"])
    index["n"] = n
    index["avg_dl"] = sum(d["len"] for d in index["docs"].values()) / n if n > 0 else 0.0

    with open(store / "_bm25_index.json", "w") as f:
        json.dump(index, f)


def query_index(
    store_dir: str,
    query: str,
    domain: str = "",
    top_k: int = 50,
    synthesis_boost: float = 1.2,
    stale_penalty: float = 0.5,
) -> Optional[List[Tuple[str, float]]]:
    """BM25 search using the inverted index. Returns None if no index exists."""
    K1 = 1.2
    B = 0.75

    index = load_index(store_dir)
    if index is None:
        return None

    words = _tokenize(query)
    if not words:
        return []

    n = index["n"]
    avg_dl = index["avg_dl"]
    if n == 0:
        return []

    # Filter docs by domain if specified
    if domain:
        docs = {p: d for p, d in index["docs"].items() if p.startswith(domain + "/")}
    else:
        docs = index["docs"]

    if not docs:
        return []

    # Compute IDF for query terms
    idf = {}
    for word in words:
        df = len(index["terms"].get(word, {}))
        idf[word] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    # Score documents
    scores = {}
    for word in words:
        if idf[word] <= 0:
            continue
        postings = index["terms"].get(word, {})
        for doc_path, tf in postings.items():
            if doc_path not in docs:
                continue
            dl = docs[doc_path]["len"]
            tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avg_dl))
            scores[doc_path] = scores.get(doc_path, 0.0) + idf[word] * tf_norm

    # Apply synthesis boost
    results = []
    for path, score in scores.items():
        doc_info = docs[path]
        if doc_info.get("synth"):
            score *= synthesis_boost * (stale_penalty if doc_info.get("stale") else 1.0)
        results.append((path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_bm25_index.py -v`
Expected: All PASS

- [ ] **Step 5: Wire into bm25.py**

At the top of `bm25_search()` in `engines/bm25.py`, add a fast path:

```python
    # Fast path: use inverted index if available
    from .bm25_index import query_index
    indexed_results = query_index(store_dir, query, domain, top_k, synthesis_boost, stale_penalty)
    if indexed_results is not None:
        return indexed_results

    # Slow path: scan files (fallback when no index exists)
```

- [ ] **Step 6: Run all BM25 tests**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_bm25.py tests/test_bm25_index.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add engines/bm25_index.py engines/bm25.py tests/test_bm25_index.py
git commit -m "feat: BM25 inverted index — O(k) search instead of O(n) file scan"
```

---

### Task 3: Edge Index

Currently every `neighbors()`, `trace()`, and `ask()` graph expansion scans the full `.edges` TSV file linearly. At 500K edges, that's a ~30MB parse per operation.

Fix: Build per-node edge index files in `_edge_index/` mapping each node to its edges. Rebuild from `.edges` on demand.

**Files:**
- Create: `engines/edge_index.py`
- Create: `tests/test_edge_index.py`
- Modify: `sdk/ragdag/core.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_edge_index.py
"""Tests for per-node edge index."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "engines"))


class TestBuildEdgeIndex:

    def _make_edges(self, tmp_path, edges: list):
        """Create .edges file from list of (src, tgt, type, meta) tuples."""
        store = tmp_path / ".ragdag"
        store.mkdir(parents=True)
        with open(store / ".edges", "w") as f:
            for src, tgt, etype, meta in edges:
                f.write(f"{src}\t{tgt}\t{etype}\t{meta}\n")
        return str(store)

    def test_build_creates_index_file(self, tmp_path):
        from edge_index import build_edge_index

        store_dir = self._make_edges(tmp_path, [
            ("a.txt", "b.txt", "related_to", ""),
        ])
        build_edge_index(store_dir)
        assert (Path(store_dir) / "_edge_index.json").exists()

    def test_lookup_outgoing(self, tmp_path):
        from edge_index import build_edge_index, lookup_edges

        store_dir = self._make_edges(tmp_path, [
            ("a.txt", "b.txt", "related_to", "sim=0.9"),
            ("a.txt", "c.txt", "derived_from", ""),
            ("d.txt", "a.txt", "references", ""),
        ])
        build_edge_index(store_dir)

        edges = lookup_edges(store_dir, "a.txt")
        outgoing = [e for e in edges if e["direction"] == "outgoing"]
        incoming = [e for e in edges if e["direction"] == "incoming"]
        assert len(outgoing) == 2
        assert len(incoming) == 1

    def test_lookup_nonexistent_node(self, tmp_path):
        from edge_index import build_edge_index, lookup_edges

        store_dir = self._make_edges(tmp_path, [
            ("a.txt", "b.txt", "related_to", ""),
        ])
        build_edge_index(store_dir)

        edges = lookup_edges(store_dir, "z.txt")
        assert edges == []

    def test_lookup_no_index_returns_none(self, tmp_path):
        from edge_index import lookup_edges

        store_dir = str(tmp_path)
        result = lookup_edges(store_dir, "a.txt")
        assert result is None

    def test_append_edge(self, tmp_path):
        from edge_index import build_edge_index, append_edge, lookup_edges

        store_dir = self._make_edges(tmp_path, [
            ("a.txt", "b.txt", "related_to", ""),
        ])
        build_edge_index(store_dir)

        append_edge(store_dir, "c.txt", "a.txt", "references", "")
        edges = lookup_edges(store_dir, "a.txt")
        incoming = [e for e in edges if e["direction"] == "incoming"]
        assert any(e["node"] == "c.txt" for e in incoming)

    def test_rebuild_from_edges_file(self, tmp_path):
        from edge_index import build_edge_index, lookup_edges

        store_dir = self._make_edges(tmp_path, [
            ("a.txt", "b.txt", "related_to", ""),
            ("b.txt", "c.txt", "chunked_from", ""),
            ("a.txt", "c.txt", "derived_from", ""),
        ])
        build_edge_index(store_dir)

        b_edges = lookup_edges(store_dir, "b.txt")
        assert len(b_edges) == 2  # incoming from a, outgoing to c
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_edge_index.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement edge index**

```python
# engines/edge_index.py
"""Per-node edge index for O(degree) graph lookups."""

import json
from pathlib import Path
from typing import Dict, List, Optional


def build_edge_index(store_dir: str) -> None:
    """Build edge index from .edges file."""
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
            source, target, etype = parts[0], parts[1], parts[2]
            metadata = parts[3] if len(parts) > 3 else ""

            # Outgoing from source
            index.setdefault(source, []).append({
                "direction": "outgoing",
                "node": target,
                "edge_type": etype,
                "metadata": metadata,
            })
            # Incoming to target
            index.setdefault(target, []).append({
                "direction": "incoming",
                "node": source,
                "edge_type": etype,
                "metadata": metadata,
            })

    with open(store / "_edge_index.json", "w") as f:
        json.dump(index, f)


def load_edge_index(store_dir: str) -> Optional[Dict]:
    """Load the edge index. Returns None if not found."""
    idx_path = Path(store_dir) / "_edge_index.json"
    if not idx_path.exists():
        return None
    with open(idx_path) as f:
        return json.load(f)


def lookup_edges(store_dir: str, node_path: str) -> Optional[List[dict]]:
    """Look up all edges for a node. Returns None if no index exists."""
    index = load_edge_index(store_dir)
    if index is None:
        return None
    return index.get(node_path, [])


def append_edge(
    store_dir: str,
    source: str, target: str, edge_type: str, metadata: str = ""
) -> None:
    """Append a single edge to the index (and .edges file)."""
    store = Path(store_dir)
    idx_path = store / "_edge_index.json"

    if not idx_path.exists():
        return  # no index to update

    index = load_edge_index(store_dir)

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

    with open(idx_path, "w") as f:
        json.dump(index, f)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_edge_index.py -v`
Expected: All PASS

- [ ] **Step 5: Wire into SDK core.py**

Replace `neighbors()` to use edge index:

```python
    def neighbors(self, node_path: str) -> List[dict]:
        """Get connected nodes."""
        # Fast path: use edge index
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.edge_index import lookup_edges
            result = lookup_edges(str(self._store), node_path)
            if result is not None:
                return result
        except ImportError:
            pass

        # Slow path: scan .edges file
        edges_file = self._edges_path()
        # ... (keep existing code as fallback)
```

Same pattern for `trace()` and the graph expansion in `ask()`.

- [ ] **Step 6: Run all tests**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_edge_index.py tests/test_sdk_integration.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add engines/edge_index.py sdk/ragdag/core.py tests/test_edge_index.py
git commit -m "feat: per-node edge index — O(degree) graph lookups"
```

---

### Task 4: ANN-based Relate

Currently `relate_cli.py` computes the full O(n^2) pairwise cosine similarity matrix. At 200K chunks, that's 40 billion comparisons — impossible.

Fix: Use random projection LSH to bucket vectors, then only compute exact similarity within buckets. Reduces to O(n × bucket_size).

**Files:**
- Create: `engines/ann.py`
- Create: `tests/test_ann.py`
- Modify: `engines/relate_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ann.py
"""Tests for approximate nearest neighbors via random projection LSH."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "engines"))


class TestRandomProjectionLSH:

    def test_similar_vectors_same_bucket(self):
        """Very similar vectors should land in the same bucket most of the time."""
        from ann import find_neighbors_ann

        np.random.seed(42)
        base = np.random.randn(128).astype(np.float32)
        # Create 10 vectors very close to base
        vectors = np.array([base + np.random.randn(128) * 0.01 for _ in range(10)],
                           dtype=np.float32)
        paths = [f"doc_{i}.txt" for i in range(10)]

        pairs = find_neighbors_ann(vectors, paths, threshold=0.95)
        # Most pairs should be found since vectors are nearly identical
        assert len(pairs) > 30  # 10 choose 2 = 45 total possible

    def test_dissimilar_vectors_different_buckets(self):
        """Random vectors should produce few matches at high threshold."""
        from ann import find_neighbors_ann

        np.random.seed(42)
        vectors = np.random.randn(50, 128).astype(np.float32)
        paths = [f"doc_{i}.txt" for i in range(50)]

        pairs = find_neighbors_ann(vectors, paths, threshold=0.95)
        # Random vectors are unlikely to have cosine > 0.95
        assert len(pairs) < 10

    def test_returns_path_pairs_with_similarity(self):
        """Each result should be (path_i, path_j, similarity)."""
        from ann import find_neighbors_ann

        np.random.seed(42)
        base = np.random.randn(32).astype(np.float32)
        vectors = np.array([base, base * 1.01, np.random.randn(32)],
                           dtype=np.float32)
        paths = ["a.txt", "b.txt", "c.txt"]

        pairs = find_neighbors_ann(vectors, paths, threshold=0.9)
        for p in pairs:
            assert len(p) == 3  # (path_i, path_j, similarity)
            assert isinstance(p[2], float)
            assert p[2] >= 0.9

    def test_empty_vectors(self):
        from ann import find_neighbors_ann

        pairs = find_neighbors_ann(np.array([]).reshape(0, 32), [], threshold=0.8)
        assert pairs == []

    def test_single_vector(self):
        from ann import find_neighbors_ann

        vectors = np.random.randn(1, 32).astype(np.float32)
        pairs = find_neighbors_ann(vectors, ["a.txt"], threshold=0.8)
        assert pairs == []

    def test_threshold_controls_results(self):
        """Lower threshold should return more pairs."""
        from ann import find_neighbors_ann

        np.random.seed(42)
        vectors = np.random.randn(20, 64).astype(np.float32)
        paths = [f"doc_{i}.txt" for i in range(20)]

        pairs_high = find_neighbors_ann(vectors, paths, threshold=0.9)
        pairs_low = find_neighbors_ann(vectors, paths, threshold=0.5)
        assert len(pairs_low) >= len(pairs_high)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_ann.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ANN**

```python
# engines/ann.py
"""Approximate Nearest Neighbors via random projection LSH.

Uses multiple random hyperplanes to hash vectors into buckets,
then computes exact cosine similarity only within buckets.
Reduces relate from O(n^2) to O(n * avg_bucket_size).
"""

from typing import List, Tuple

import numpy as np


def find_neighbors_ann(
    vectors: np.ndarray,
    paths: List[str],
    threshold: float = 0.8,
    n_tables: int = 10,
    n_bits: int = 16,
    seed: int = 42,
) -> List[Tuple[str, str, float]]:
    """Find pairs of vectors with cosine similarity >= threshold.

    Args:
        vectors: (n, dims) array of float32 vectors.
        paths: Corresponding chunk paths.
        threshold: Minimum cosine similarity.
        n_tables: Number of hash tables (more = better recall, slower).
        n_bits: Bits per hash (more = smaller buckets, less recall).
        seed: Random seed for reproducibility.

    Returns:
        List of (path_i, path_j, similarity) tuples.
    """
    n = len(vectors)
    if n < 2:
        return []

    dims = vectors.shape[1]
    rng = np.random.RandomState(seed)

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    normed = vectors / norms

    # Generate random hyperplanes for each table
    candidate_pairs = set()

    for _ in range(n_tables):
        # Random projection: sign(vectors @ random_matrix) → hash bits
        planes = rng.randn(dims, n_bits).astype(np.float32)
        projections = normed @ planes
        hashes = (projections > 0).astype(np.uint8)

        # Convert to hashable keys
        buckets = {}
        for i in range(n):
            key = hashes[i].tobytes()
            buckets.setdefault(key, []).append(i)

        # Collect candidate pairs from same bucket
        for indices in buckets.values():
            if len(indices) < 2:
                continue
            for a_idx in range(len(indices)):
                for b_idx in range(a_idx + 1, len(indices)):
                    i, j = indices[a_idx], indices[b_idx]
                    if i < j:
                        candidate_pairs.add((i, j))
                    else:
                        candidate_pairs.add((j, i))

    # Compute exact similarity for candidates only
    results = []
    for i, j in candidate_pairs:
        sim = float(normed[i] @ normed[j])
        if sim >= threshold:
            results.append((paths[i], paths[j], sim))

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_ann.py -v`
Expected: All PASS

- [ ] **Step 5: Wire into relate_cli.py**

In `engines/relate_cli.py`, replace the O(n^2) pairwise block with:

```python
        # Use ANN for large corpora, exact for small
        if count > 1000:
            from .ann import find_neighbors_ann
            pairs = find_neighbors_ann(vectors, [m[0] for m in manifest], args.threshold)
            for path_i, path_j, sim in pairs:
                adjacency.setdefault(path_i, []).append(path_j)
                adjacency.setdefault(path_j, []).append(path_i)
                if (path_i, path_j) not in existing_edges:
                    new_edges.append(f"{path_i}\t{path_j}\trelated_to\tsimilarity={sim:.4f}")
                    existing_edges.add((path_i, path_j))
                    existing_edges.add((path_j, path_i))
                    added += 1
        else:
            # Original exact pairwise for small corpora
            # ... (keep existing code)
```

- [ ] **Step 6: Run all relate and ANN tests**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/test_ann.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add engines/ann.py engines/relate_cli.py tests/test_ann.py
git commit -m "feat: ANN-based relate using random projection LSH (replaces O(n^2) pairwise)"
```

---

### Task 5: Wire Index Building into Ingest Pipeline

Connect all indexes to the `add()` pipeline so they stay up to date.

**Files:**
- Modify: `sdk/ragdag/core.py` (add method)
- Modify: `lib/add.sh` (bash ingest)

- [ ] **Step 1: Write integration test**

```python
# Append to tests/test_pipeline_integration.py

class TestIndexBuildOnIngest:

    def test_add_builds_bm25_index(self, tmp_path):
        """After dag.add(), _bm25_index.json should exist."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nSome searchable content here.\n")
        dag.add(str(doc))

        assert (tmp_path / ".ragdag" / "_bm25_index.json").exists()

    def test_add_builds_edge_index(self, tmp_path):
        """After dag.add(), _edge_index.json should exist."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nSome content.\n")
        dag.add(str(doc))

        assert (tmp_path / ".ragdag" / "_edge_index.json").exists()

    def test_search_uses_index_after_add(self, tmp_path):
        """Search after add should use the BM25 index (not file scan)."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nUnique searchable keyword.\n")
        dag.add(str(doc))

        results = dag.search("searchable keyword", mode="keyword")
        assert len(results) >= 1
```

- [ ] **Step 2: Wire index building into SDK add()**

In `sdk/ragdag/core.py`, find the `add()` method. After chunks are written and edges are created, add:

```python
        # Build/update indexes
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.bm25_index import update_index
            from engines.edge_index import build_edge_index

            # Update BM25 index with new chunks
            new_chunk_paths = [str(p.relative_to(self._store)) for p in chunk_files]
            update_index(str(self._store), new_chunk_paths)

            # Rebuild edge index (append-safe)
            build_edge_index(str(self._store))
        except ImportError:
            pass  # engines not available
```

- [ ] **Step 3: Add reindex commands**

In `sdk/ragdag/core.py`, add a `reindex()` method:

```python
    def reindex(self, what: str = "all") -> None:
        """Rebuild indexes. what: 'bm25', 'edges', or 'all'."""
        sys.path.insert(0, str(self._ragdag_dir))
        if what in ("bm25", "all"):
            from engines.bm25_index import build_index
            build_index(str(self._store))
        if what in ("edges", "all"):
            from engines.edge_index import build_edge_index
            build_edge_index(str(self._store))
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/vivek/jet/ragdag && python3 -m pytest tests/ -q`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add sdk/ragdag/core.py tests/test_pipeline_integration.py
git commit -m "feat: build BM25 and edge indexes on ingest, add reindex()"
```

---

## Verification Plan

After all tasks:

1. **Append-only embeddings**: Add 100 docs, measure time. Add 1 more doc, verify time is constant (not proportional to existing count).
2. **BM25 index**: Search with and without `_bm25_index.json`. Verify results are identical. Verify search with index doesn't open .txt files.
3. **Edge index**: Call `neighbors()` with and without `_edge_index.json`. Verify results are identical. Verify indexed path doesn't scan `.edges`.
4. **ANN relate**: Run `relate` on a domain with >1000 chunks. Verify it completes in reasonable time (vs hanging on O(n^2)).
5. **Full pipeline**: `python3 -m pytest tests/ -q` — all tests pass.
6. **Backward compat**: Delete all index files. All operations should fall back to file scanning gracefully.
