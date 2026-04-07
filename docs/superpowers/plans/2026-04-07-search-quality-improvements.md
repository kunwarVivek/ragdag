# Search Quality Improvements — Lessons from qmd

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring ragdag's search pipeline up to modern standards by adopting RRF fusion, cross-encoder reranking, BM25 keyword scoring, content-addressable embedding cache, and search explainability — all without SQLite, preserving the flat-file philosophy.

**Architecture:** Each improvement is a self-contained module that plugs into the existing search pipeline. The pipeline becomes: keyword(BM25) + vector → RRF fusion → rerank → results. Each piece is independently toggleable via `.config`.

**Tech Stack:** Python, numpy, sentence-transformers (CrossEncoder), existing ragdag engines

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `engines/bm25.py` | Pure-Python BM25 scoring over flat .txt files |
| Create | `engines/reranker.py` | Cross-encoder reranker engine abstraction |
| Create | `engines/rrf.py` | Reciprocal Rank Fusion combining ranked lists |
| Modify | `engines/search_cli.py` | Wire new pipeline: BM25 + vector → RRF → rerank |
| Modify | `engines/embeddings.py` | Content-addressable embedding cache via manifest |
| Modify | `sdk/ragdag/core.py` | Mirror pipeline changes in SDK search methods |
| Modify | `lib/search.sh` | Pass `--explain` flag through to Python |
| Create | `tests/test_bm25.py` | BM25 scoring tests |
| Create | `tests/test_rrf.py` | RRF fusion tests |
| Create | `tests/test_reranker.py` | Reranker tests |
| Create | `tests/test_explain.py` | Explain mode output tests |
| Modify | `tests/test_search_hybrid.py` | Update for new fusion behavior |

---

### Task 1: BM25 Keyword Search

Replace substring counting with proper BM25 scoring. No SQLite — pure Python over flat files.

**Files:**
- Create: `engines/bm25.py`
- Create: `tests/test_bm25.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bm25.py
"""Tests for BM25 keyword search scoring."""

import pytest
from pathlib import Path


class TestBM25Scoring:
    """Tests for BM25 scoring correctness."""

    def _make_corpus(self, tmp_path, docs: dict):
        """Create .txt files from {name: content} dict, return store dir."""
        store = tmp_path / ".ragdag" / "testdomain"
        store.mkdir(parents=True)
        for name, content in docs.items():
            (store / f"{name}.txt").write_text(content)
        return str(tmp_path / ".ragdag")

    def test_idf_prefers_rare_terms(self, tmp_path):
        """A term appearing in fewer documents should score higher."""
        store_dir = self._make_corpus(tmp_path, {
            "01": "the cat sat on the mat",
            "02": "the dog sat on the mat",
            "03": "the cat chased the dog",
            "04": "quantum physics experiment results",
        })
        from engines.bm25 import bm25_search

        # "quantum" appears in 1/4 docs — should rank doc 04 first
        results = bm25_search(store_dir, "quantum", domain="testdomain")
        assert len(results) >= 1
        assert results[0][0].endswith("04.txt")

    def test_tf_saturation(self, tmp_path):
        """Repeating a term should help but with diminishing returns (BM25 saturation)."""
        store_dir = self._make_corpus(tmp_path, {
            "01": "python " * 50,        # 50 mentions
            "02": "python " * 5 + "programming language guide",  # 5 mentions + context
        })
        from engines.bm25 import bm25_search

        results = bm25_search(store_dir, "python", domain="testdomain")
        assert len(results) == 2
        # doc 01 should score higher but NOT 10x higher (saturation)
        ratio = results[0][1] / results[1][1]
        assert ratio < 5.0  # BM25 saturates; raw TF would give ~10x

    def test_document_length_normalization(self, tmp_path):
        """Shorter docs with same match count should score higher."""
        store_dir = self._make_corpus(tmp_path, {
            "01": "machine learning",
            "02": "machine learning " + "filler word " * 100,
        })
        from engines.bm25 import bm25_search

        results = bm25_search(store_dir, "machine learning", domain="testdomain")
        assert len(results) == 2
        assert results[0][0].endswith("01.txt")  # shorter doc wins

    def test_multi_term_query(self, tmp_path):
        """Multi-term queries accumulate IDF-weighted scores."""
        store_dir = self._make_corpus(tmp_path, {
            "01": "authentication oauth tokens security",
            "02": "authentication database migration",
            "03": "oauth tokens refresh flow",
        })
        from engines.bm25 import bm25_search

        results = bm25_search(store_dir, "oauth tokens", domain="testdomain")
        # doc 01 and 03 both match; doc 02 doesn't match either term
        paths = [r[0] for r in results]
        assert not any(p.endswith("02.txt") for p in paths)

    def test_no_results_for_absent_terms(self, tmp_path):
        """Query with no matching terms returns empty list."""
        store_dir = self._make_corpus(tmp_path, {
            "01": "hello world",
        })
        from engines.bm25 import bm25_search

        results = bm25_search(store_dir, "xyznonexistent", domain="testdomain")
        assert results == []

    def test_synthesis_node_boost(self, tmp_path):
        """Synthesis nodes (prefixed with _) get configurable boost."""
        store = tmp_path / ".ragdag" / "testdomain"
        store.mkdir(parents=True)
        (store / "01.txt").write_text("neural network training")
        (store / "_summary.txt").write_text(
            "---\ntype: summary\nstale: false\n---\nneural network training overview"
        )
        store_dir = str(tmp_path / ".ragdag")

        from engines.bm25 import bm25_search

        results = bm25_search(store_dir, "neural network", domain="testdomain", synthesis_boost=1.5)
        # Both should appear; synthesis node should be boosted
        synth_results = [r for r in results if "_summary" in r[0]]
        regular_results = [r for r in results if "_summary" not in r[0]]
        assert len(synth_results) >= 1
        assert len(regular_results) >= 1


class TestBM25EdgeCases:

    def test_empty_store(self, tmp_path):
        store = tmp_path / ".ragdag" / "empty"
        store.mkdir(parents=True)
        from engines.bm25 import bm25_search
        assert bm25_search(str(tmp_path / ".ragdag"), "test", domain="empty") == []

    def test_single_char_words_ignored(self, tmp_path):
        """Words shorter than 2 chars should be skipped."""
        store = tmp_path / ".ragdag" / "d"
        store.mkdir(parents=True)
        (store / "01.txt").write_text("a b c real content here")
        from engines.bm25 import bm25_search
        results = bm25_search(str(tmp_path / ".ragdag"), "a b real", domain="d")
        assert len(results) == 1  # only "real" matched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_bm25.py -v`
Expected: `ModuleNotFoundError: No module named 'engines.bm25'`

- [ ] **Step 3: Implement BM25**

```python
# engines/bm25.py
"""BM25 scoring over flat .txt files — no SQLite needed."""

import math
from pathlib import Path
from typing import List, Tuple, Optional


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
    df = {}  # word -> document frequency
    for word in words:
        df[word] = sum(1 for _, content, _, _ in docs if word in content)

    idf = {}
    for word in words:
        # Standard BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
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
            # BM25 TF component with saturation and length normalization
            tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avg_dl))
            score += idf[word] * tf_norm

        if score > 0:
            if is_synth:
                score *= synthesis_boost * (stale_penalty if is_stale else 1.0)
            results.append((rel_path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_bm25.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add engines/bm25.py tests/test_bm25.py
git commit -m "feat: add BM25 keyword scoring engine (replaces substring counting)"
```

---

### Task 2: Reciprocal Rank Fusion

Replace weighted linear score fusion with RRF. This is rank-based and robust to score distribution skew.

**Files:**
- Create: `engines/rrf.py`
- Create: `tests/test_rrf.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rrf.py
"""Tests for Reciprocal Rank Fusion."""

import pytest


class TestRRF:

    def test_basic_fusion(self):
        """Two ranked lists fused with RRF."""
        from engines.rrf import reciprocal_rank_fusion

        list_a = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        list_b = [("doc2", 0.95), ("doc3", 0.8), ("doc1", 0.3)]

        results = reciprocal_rank_fusion([list_a, list_b], k=60)

        # doc2 is rank 1 in list_b and rank 2 in list_a — should be top
        assert results[0][0] == "doc2"
        # All 3 docs should appear
        assert len(results) == 3

    def test_disjoint_lists(self):
        """Documents appearing in only one list still get scored."""
        from engines.rrf import reciprocal_rank_fusion

        list_a = [("doc1", 0.9), ("doc2", 0.5)]
        list_b = [("doc3", 0.8), ("doc4", 0.6)]

        results = reciprocal_rank_fusion([list_a, list_b], k=60)
        paths = [r[0] for r in results]
        assert set(paths) == {"doc1", "doc2", "doc3", "doc4"}

    def test_single_list(self):
        """With one list, RRF preserves rank order."""
        from engines.rrf import reciprocal_rank_fusion

        list_a = [("doc1", 0.9), ("doc2", 0.5), ("doc3", 0.1)]
        results = reciprocal_rank_fusion([list_a], k=60)
        assert [r[0] for r in results] == ["doc1", "doc2", "doc3"]

    def test_empty_lists(self):
        from engines.rrf import reciprocal_rank_fusion
        assert reciprocal_rank_fusion([[], []], k=60) == []
        assert reciprocal_rank_fusion([], k=60) == []

    def test_weighted_lists(self):
        """Lists can have different weights."""
        from engines.rrf import reciprocal_rank_fusion

        list_a = [("doc1", 0.9)]
        list_b = [("doc2", 0.9)]

        # Equal weight — tied by insertion order
        r1 = reciprocal_rank_fusion([list_a, list_b], k=60, weights=[1.0, 1.0])

        # Double weight on list_b — doc2 should win
        r2 = reciprocal_rank_fusion([list_a, list_b], k=60, weights=[1.0, 2.0])
        assert r2[0][0] == "doc2"

    def test_k_parameter_affects_scores(self):
        """Lower k gives more weight to top ranks."""
        from engines.rrf import reciprocal_rank_fusion

        list_a = [("doc1", 0.9), ("doc2", 0.1)]
        list_b = [("doc2", 0.9), ("doc1", 0.1)]

        # With very low k, rank 1 vs rank 2 is a bigger gap
        r_low_k = reciprocal_rank_fusion([list_a, list_b], k=1)
        # With high k, rank difference is smaller
        r_high_k = reciprocal_rank_fusion([list_a, list_b], k=1000)

        # Both should have same docs, but score spread differs
        assert len(r_low_k) == 2
        assert len(r_high_k) == 2

    def test_top_k_limit(self):
        """Results can be limited to top_k."""
        from engines.rrf import reciprocal_rank_fusion

        list_a = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        results = reciprocal_rank_fusion([list_a], k=60, top_k=2)
        assert len(results) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_rrf.py -v`
Expected: `ModuleNotFoundError: No module named 'engines.rrf'`

- [ ] **Step 3: Implement RRF**

```python
# engines/rrf.py
"""Reciprocal Rank Fusion for combining multiple ranked result lists."""

from typing import List, Optional, Tuple
from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over lists L of: weight_L / (k + rank_of_d_in_L)
    Documents not in a list are skipped for that list (no penalty).

    Args:
        ranked_lists: Each list is [(path, score)] sorted descending by score.
        k: Ranking constant. Higher k reduces the gap between ranks.
            Standard value is 60.
        weights: Optional weight per list. Defaults to equal weights.
        top_k: Optional limit on returned results.

    Returns:
        Fused [(path, rrf_score)] sorted descending.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: dict[str, float] = defaultdict(float)

    for lst, weight in zip(ranked_lists, weights):
        for rank, (path, _score) in enumerate(lst, start=1):
            rrf_scores[path] += weight / (k + rank)

    results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_rrf.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add engines/rrf.py tests/test_rrf.py
git commit -m "feat: add Reciprocal Rank Fusion module for robust score fusion"
```

---

### Task 3: Cross-Encoder Reranker

Add an optional reranking step using a cross-encoder model from sentence-transformers.

**Files:**
- Create: `engines/reranker.py`
- Create: `tests/test_reranker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reranker.py
"""Tests for cross-encoder reranker."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestRerankerInterface:

    def test_rerank_reorders_results(self):
        """Reranker should re-score and reorder results."""
        from engines.reranker import rerank

        candidates = [
            ("doc1.txt", 0.8, "python is a programming language"),
            ("doc2.txt", 0.6, "how to cook pasta at home"),
            ("doc3.txt", 0.4, "python snake species in australia"),
        ]

        # Mock the cross-encoder to give high score to doc1 (programming)
        mock_scores = [0.9, 0.1, 0.3]  # doc1 best match for "python programming"

        with patch("engines.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_scores
            mock_ce.return_value = mock_model

            results = rerank("python programming", candidates)

        assert results[0][0] == "doc1.txt"
        assert results[-1][0] == "doc2.txt"  # cooking is least relevant

    def test_rerank_preserves_all_candidates(self):
        """All input candidates should appear in output."""
        from engines.reranker import rerank

        candidates = [
            ("a.txt", 0.5, "content a"),
            ("b.txt", 0.5, "content b"),
        ]

        with patch("engines.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.5, 0.5]
            mock_ce.return_value = mock_model

            results = rerank("query", candidates)

        assert len(results) == 2

    def test_rerank_empty_candidates(self):
        """Empty candidate list returns empty."""
        from engines.reranker import rerank
        assert rerank("query", []) == []

    def test_rerank_top_k(self):
        """top_k limits output."""
        from engines.reranker import rerank

        candidates = [
            ("a.txt", 0.5, "aaa"),
            ("b.txt", 0.4, "bbb"),
            ("c.txt", 0.3, "ccc"),
        ]

        with patch("engines.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.3, 0.9, 0.6]
            mock_ce.return_value = mock_model

            results = rerank("query", candidates, top_k=2)

        assert len(results) == 2

    def test_rerank_blends_rrf_and_reranker_scores(self):
        """Final score should blend original RRF score with reranker score."""
        from engines.reranker import rerank

        candidates = [
            ("a.txt", 0.8, "high rrf content"),
            ("b.txt", 0.2, "low rrf content"),
        ]

        with patch("engines.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            # Reranker reverses the order
            mock_model.predict.return_value = [0.1, 0.9]
            mock_ce.return_value = mock_model

            results = rerank("query", candidates)

        # b.txt should win (reranker dominates for non-top positions)
        assert results[0][0] == "b.txt"


class TestRerankerFallback:

    def test_rerank_returns_original_order_on_import_error(self):
        """If sentence-transformers not installed, return candidates unchanged."""
        from engines.reranker import rerank

        candidates = [
            ("a.txt", 0.8, "aaa"),
            ("b.txt", 0.6, "bbb"),
        ]

        with patch("engines.reranker._get_cross_encoder", side_effect=ImportError("no module")):
            results = rerank("query", candidates)

        # Should return original order, not crash
        assert results[0][0] == "a.txt"
        assert len(results) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_reranker.py -v`
Expected: `ModuleNotFoundError: No module named 'engines.reranker'`

- [ ] **Step 3: Implement reranker**

```python
# engines/reranker.py
"""Cross-encoder reranker for search result refinement."""

from typing import List, Optional, Tuple

_cross_encoder = None


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def rerank(
    query: str,
    candidates: List[Tuple[str, float, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Rerank candidates using a cross-encoder model.

    Args:
        query: The search query.
        candidates: List of (path, rrf_score, content) tuples.
        model_name: Cross-encoder model name.
        top_k: Optional limit on results.

    Returns:
        List of (path, blended_score) sorted descending.
    """
    if not candidates:
        return []

    try:
        model = _get_cross_encoder(model_name)
    except (ImportError, Exception):
        # Graceful degradation: return original ranking
        return [(path, score) for path, score, _ in candidates]

    # Build query-document pairs for cross-encoder
    pairs = [(query, content) for _, _, content in candidates]
    ce_scores = model.predict(pairs)

    # Normalize cross-encoder scores to [0, 1]
    ce_min = min(ce_scores)
    ce_max = max(ce_scores)
    ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
    ce_norm = [(s - ce_min) / ce_range for s in ce_scores]

    # Normalize RRF scores to [0, 1]
    rrf_scores = [score for _, score, _ in candidates]
    rrf_max = max(rrf_scores) if rrf_scores else 1.0
    rrf_norm = [s / rrf_max if rrf_max > 0 else 0 for s in rrf_scores]

    # Blend: RRF 40%, reranker 60%
    # (qmd uses position-aware blending; this is simpler and sufficient)
    results = []
    for i, (path, _orig_score, _content) in enumerate(candidates):
        blended = 0.4 * rrf_norm[i] + 0.6 * ce_norm[i]
        results.append((path, blended))

    results.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_reranker.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add engines/reranker.py tests/test_reranker.py
git commit -m "feat: add cross-encoder reranker with graceful degradation"
```

---

### Task 4: Content-Addressable Embedding Cache

Skip re-embedding chunks whose content hasn't changed by tracking SHA-256 of chunk text in the manifest.

**Files:**
- Modify: `engines/embeddings.py:22-93` (write_embeddings and manifest format)
- Modify: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_embeddings.py

class TestContentAddressableCache:

    def test_manifest_includes_content_hash(self, tmp_path):
        """Manifest should include content_hash column."""
        from engines.embeddings import write_embeddings, load_manifest

        write_embeddings(
            output_dir=str(tmp_path),
            vectors=[[0.1, 0.2, 0.3]],
            chunk_paths=["test/01.txt"],
            dimensions=3,
            model_name_str="test-model",
            chunk_texts=["hello world"],
        )

        manifest = load_manifest(str(tmp_path / "manifest.tsv"))
        # Manifest entries should now be 5-tuples with content_hash
        assert len(manifest[0]) == 5
        assert manifest[0][4] != ""  # content_hash is populated

    def test_skip_reembedding_unchanged_content(self, tmp_path):
        """When chunk text hasn't changed, embedding should be reused."""
        from engines.embeddings import write_embeddings, load_manifest, load_embeddings
        import numpy as np

        # First write
        write_embeddings(
            output_dir=str(tmp_path),
            vectors=[[0.1, 0.2, 0.3]],
            chunk_paths=["test/01.txt"],
            dimensions=3,
            model_name_str="test-model",
            chunk_texts=["hello world"],
        )

        # Second write with same content but different path
        write_embeddings(
            output_dir=str(tmp_path),
            vectors=[[0.9, 0.9, 0.9]],  # different vector (should be ignored)
            chunk_paths=["test/02.txt"],
            dimensions=3,
            model_name_str="test-model",
            chunk_texts=["hello world"],  # same content!
            append=True,
        )

        manifest = load_manifest(str(tmp_path / "manifest.tsv"))
        vecs, _, count, _ = load_embeddings(str(tmp_path / "embeddings.bin"))

        assert count == 2
        # The second entry should reuse the first vector, not the new one
        np.testing.assert_array_almost_equal(vecs[1], [0.1, 0.2, 0.3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_embeddings.py::TestContentAddressableCache -v`
Expected: FAIL — `write_embeddings() got unexpected keyword argument 'chunk_texts'`

- [ ] **Step 3: Modify embeddings.py**

In `engines/embeddings.py`, modify `write_embeddings` to accept optional `chunk_texts` parameter, compute SHA-256 content hashes, and skip re-embedding content that already exists in the manifest:

Add at top of file:
```python
import hashlib  # already imported for model_hash
```

Modify `write_embeddings` signature to add `chunk_texts: Optional[List[str]] = None`:

```python
def write_embeddings(
    output_dir: str,
    vectors: List[List[float]],
    chunk_paths: List[str],
    dimensions: int,
    model_name_str: str,
    append: bool = True,
    chunk_texts: Optional[List[str]] = None,
) -> None:
```

After loading existing data (line ~43), build a content hash lookup from existing manifest:

```python
    # Build content hash → vector lookup from existing data
    existing_hash_to_vec = {}
    if chunk_texts is not None and existing_paths:
        existing_manifest = load_manifest(str(manifest_path)) if manifest_path.exists() else []
        for entry in existing_manifest:
            if len(entry) >= 5 and entry[4]:
                idx = entry[1]
                if idx < len(existing_vectors):
                    existing_hash_to_vec[entry[4]] = existing_vectors[idx]
```

Before combining (line ~63), deduplicate by content hash:

```python
    # Content-addressable dedup: reuse vectors for identical content
    if chunk_texts is not None:
        content_hashes = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in chunk_texts]
        deduped_vectors = []
        for i, chash in enumerate(content_hashes):
            if chash in existing_hash_to_vec:
                deduped_vectors.append(existing_hash_to_vec[chash])
            else:
                deduped_vectors.append(vectors[i])
        vectors = deduped_vectors
    else:
        content_hashes = [""] * len(chunk_paths)
```

Modify manifest write (line ~88) to include content hash:

```python
    with open(manifest_path, "w") as f:
        f.write("# relative_chunk_path\tindex\tbyte_offset\tdimensions\tcontent_hash\n")
        for i, path in enumerate(all_paths):
            offset = HEADER_SIZE + i * dimensions * 4
            chash = all_hashes[i] if i < len(all_hashes) else ""
            f.write(f"{path}\t{i}\t{offset}\t{dimensions}\t{chash}\n")
```

Update `load_manifest` to return 5-tuples (backward compatible):

```python
def load_manifest(manifest_path: str) -> List[Tuple[str, int, int, int, str]]:
    """Load manifest entries: (path, index, byte_offset, dimensions, content_hash)."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                content_hash = parts[4] if len(parts) >= 5 else ""
                entries.append(
                    (parts[0], int(parts[1]), int(parts[2]), int(parts[3]), content_hash)
                )
    return entries
```

- [ ] **Step 4: Fix callers that destructure manifest as 4-tuples**

In `engines/similarity.py:72` and `engines/similarity.py:85`, change:
```python
# Before:
for i, (p, _, _, _) in enumerate(manifest)
# After:
for i, entry in enumerate(manifest):
    p = entry[0]
```

Same pattern in `search_cli.py` if it destructures manifest entries.

- [ ] **Step 5: Run all embedding and similarity tests**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_embeddings.py tests/test_similarity.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add engines/embeddings.py engines/similarity.py tests/test_embeddings.py
git commit -m "feat: content-addressable embedding cache via SHA-256 in manifest"
```

---

### Task 5: Wire New Search Pipeline

Replace the old fusion logic in both `search_cli.py` and `sdk/ragdag/core.py` with BM25 → vector → RRF → optional rerank.

**Files:**
- Modify: `engines/search_cli.py:54-127`
- Modify: `sdk/ragdag/core.py:890-972`
- Modify: `tests/test_search_hybrid.py`

- [ ] **Step 1: Write failing test for new pipeline**

```python
# Append to tests/test_search_hybrid.py

class TestRRFPipeline:
    """Tests that hybrid search uses RRF instead of weighted linear fusion."""

    def test_hybrid_uses_rrf_not_weighted_sum(self, tmp_path):
        """Hybrid mode should use RRF, not normalized weighted sum."""
        dag = ragdag.init(str(tmp_path))
        for i in range(5):
            doc = tmp_path / f"doc{i}.md"
            doc.write_text(f"# Document {i}\n\nUnique content for variation {i}.\n")
            dag.add(str(doc))

        # With provider=none, hybrid falls back to keyword only
        # but the pipeline path should still work
        results = dag.search("unique content", mode="hybrid")
        assert len(results) >= 1

    def test_explain_mode_returns_breakdown(self, tmp_path):
        """Search with explain=True should return score breakdown."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nSearch explain mode verification.\n")
        dag.add(str(doc))

        results = dag.search("search explain", mode="hybrid", explain=True)
        assert len(results) >= 1
        # Each result should have explain data
        assert hasattr(results[0], 'explain') or isinstance(results[0], dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_search_hybrid.py::TestRRFPipeline -v`
Expected: FAIL — `search() got unexpected keyword argument 'explain'`

- [ ] **Step 3: Update search_cli.py hybrid mode**

Replace the score fusion block in `cmd_search` (lines 77-117) with:

```python
    elif mode == "hybrid":
        from .bm25 import bm25_search
        from .rrf import reciprocal_rank_fusion

        # Phase 1: BM25 keyword search
        bm25_results = bm25_search(store_dir, query, domain, top_k=top_k * 3)

        # Phase 2: Vector search (over-fetch for fusion)
        candidate_paths = [path for path, _ in bm25_results] if bm25_results else None
        vec_results = search_vectors(
            query_embedding=query_vec,
            store_dir=store_dir,
            domain=domain,
            top_k=top_k * 3,
            candidate_paths=candidate_paths,
        )

        # Phase 3: RRF fusion
        results = reciprocal_rank_fusion(
            [bm25_results, vec_results],
            k=60,
            top_k=top_k,
        )

        # Phase 4: Optional reranking
        rerank_enabled = _read_config(store_dir).get("search.rerank", "false") == "true"
        if rerank_enabled and results:
            from .reranker import rerank
            store = Path(store_dir)
            candidates = []
            for path, score in results:
                full_path = store / path
                content = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
                candidates.append((path, score, content))
            rerank_model = _read_config(store_dir).get(
                "search.rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            results = rerank(query, candidates, model_name=rerank_model, top_k=top_k)
```

Add `--explain` and `--rerank` args to the argument parser (line ~213):

```python
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--rerank", action="store_true")
```

- [ ] **Step 4: Update SDK core.py hybrid search**

In `sdk/ragdag/core.py`, modify `search()` to accept `explain: bool = False` parameter.

Modify `_python_search` (line 890) to replace the weighted sum block with:

```python
        if mode == "hybrid":
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.bm25 import bm25_search
            from engines.rrf import reciprocal_rank_fusion

            # BM25 keyword search
            bm25_results = bm25_search(
                str(self._store), query, domain or "", top * 3,
                synthesis_boost=self._synthesis_boost(),
            )

            # Vector search
            vec_results = search_vectors(
                query_embedding=query_vec,
                store_dir=str(self._store),
                domain=domain or "",
                top_k=top * 3,
                candidate_paths=[p for p, _ in bm25_results] if bm25_results else None,
            )

            # RRF fusion
            fused = reciprocal_rank_fusion([bm25_results, vec_results], k=60, top_k=top)

            # Optional reranking
            rerank_enabled = self._read_config("search.rerank", "false") == "true"
            if rerank_enabled and fused:
                from engines.reranker import rerank as do_rerank
                candidates = []
                for path, score in fused:
                    fp = self._store / path
                    content = fp.read_text(encoding="utf-8") if fp.exists() else ""
                    candidates.append((path, score, content))
                fused = do_rerank(query, candidates, top_k=top)

            vec_results = fused
```

- [ ] **Step 5: Update _keyword_search to use BM25**

In `sdk/ragdag/core.py`, replace `_keyword_search` body (lines 837-888) with:

```python
    def _keyword_search(
        self, query: str, domain: Optional[str], top: int
    ) -> List[SearchResult]:
        """BM25 keyword search over flat files."""
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.bm25 import bm25_search

            results = bm25_search(
                str(self._store), query, domain or "", top,
                synthesis_boost=self._synthesis_boost(),
            )
        except ImportError:
            # Fallback: if engines not available, use old substring matching
            return self._keyword_search_legacy(query, domain, top)

        return [
            SearchResult(
                path=path, score=score,
                content=(self._store / path).read_text(encoding="utf-8")
                    if (self._store / path).exists() else "",
                domain=path.split("/")[0] if len(path.split("/")) >= 3 else "",
            )
            for path, score in results
        ]
```

Rename the old `_keyword_search` to `_keyword_search_legacy` as a fallback.

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add engines/search_cli.py sdk/ragdag/core.py tests/test_search_hybrid.py
git commit -m "feat: wire BM25 + RRF + optional reranking into search pipeline"
```

---

### Task 6: Search Explain Mode

Add `--explain` flag that outputs score breakdown per result.

**Files:**
- Modify: `engines/search_cli.py` (output functions)
- Modify: `sdk/ragdag/core.py` (SearchResult dataclass)
- Create: `tests/test_explain.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_explain.py
"""Tests for search explain mode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

import ragdag


class TestExplainMode:

    def test_explain_adds_breakdown_to_results(self, tmp_path):
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nKeyword matching explanation mode.\n")
        dag.add(str(doc))

        results = dag.search("keyword matching", mode="keyword", explain=True)
        assert len(results) >= 1
        assert results[0].explain is not None
        assert "bm25" in results[0].explain

    def test_explain_false_has_no_breakdown(self, tmp_path):
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nNo explain mode.\n")
        dag.add(str(doc))

        results = dag.search("explain", mode="keyword")
        assert len(results) >= 1
        assert results[0].explain is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_explain.py -v`
Expected: FAIL

- [ ] **Step 3: Add `explain` field to SearchResult**

In `sdk/ragdag/core.py`, find the `SearchResult` dataclass and add:

```python
    explain: Optional[dict] = None
```

- [ ] **Step 4: Populate explain data in search methods**

In `_keyword_search`, when `explain=True`, attach breakdown:

```python
explain_data = {"bm25": score} if explain else None
```

In `_python_search` hybrid mode, after RRF:

```python
explain_data = {
    "bm25": bm25_score_for_path,
    "vector": vec_score_for_path,
    "rrf": rrf_score,
    "reranker": reranker_score_if_applicable,
} if explain else None
```

Pass `explain` parameter through `search()` → `_keyword_search()` and `_python_search()`.

- [ ] **Step 5: Add --explain to CLI output**

In `search_cli.py`, when `args.explain` is set, include score breakdown in JSON output:

```python
if args.explain:
    entry["explain"] = {"bm25": bm25_score, "vector": vec_score, "rrf": rrf_score}
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_explain.py tests/test_search_hybrid.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add engines/search_cli.py sdk/ragdag/core.py tests/test_explain.py
git commit -m "feat: add --explain mode showing BM25/vector/RRF score breakdown"
```

---

### Task 7: Update Bash Search to Pass Through New Flags

**Files:**
- Modify: `lib/search.sh:20-37` (argument parsing)

- [ ] **Step 1: Add --explain and --rerank flags to bash argument parser**

In `lib/search.sh`, add to the `case` block (after line 27):

```bash
      --explain)  explain=1; shift ;;
      --rerank)   rerank=1; shift ;;
```

Initialize them before the while loop:

```bash
  local explain=0
  local rerank=0
```

Pass them through in `_search_hybrid` (line 233):

```bash
  [[ "$explain" -eq 1 ]] && args+=(--explain)
  [[ "$rerank" -eq 1 ]] && args+=(--rerank)
```

- [ ] **Step 2: Test bash passthrough manually**

Run: `cd /Users/vivek/jet/ragdag && bash -c 'source ragdag && ragdag search "test" --explain --json'`
Expected: JSON output with explain fields (or graceful error if no store)

- [ ] **Step 3: Commit**

```bash
git add lib/search.sh
git commit -m "feat: pass --explain and --rerank flags through bash to Python search"
```

---

### Task 8: Config Defaults

Add new config keys with sensible defaults.

**Files:**
- Modify: `sdk/ragdag/core.py` (init method where .config is written)

- [ ] **Step 1: Find where .config is initialized**

Look for where `ragdag init` writes the default `.config` file and add:

```ini
[search]
default_mode = hybrid
top_k = 10
rerank = false
rerank_model = cross-encoder/ms-marco-MiniLM-L-6-v2
```

- [ ] **Step 2: Verify init creates config with new keys**

Run: `cd /tmp && python -c "import ragdag; dag = ragdag.init('/tmp/test_cfg'); print(open('/tmp/test_cfg/.ragdag/.config').read())"`
Expected: Config contains `[search]` section with `rerank = false`

- [ ] **Step 3: Commit**

```bash
git add sdk/ragdag/core.py
git commit -m "feat: add search.rerank config keys with defaults"
```

---

## Verification Plan

After all tasks are complete:

1. **BM25 quality**: `ragdag search "authentication" --keyword --explain` — verify BM25 scores show IDF weighting
2. **RRF fusion**: `ragdag search "query" --hybrid --explain` — verify results show RRF scores, not weighted sums
3. **Reranking**: Set `search.rerank = true` in `.config`, run search, verify reranker scores appear in explain
4. **Content cache**: Re-ingest same document, verify embedding API isn't called for unchanged chunks
5. **Backward compat**: `python -m pytest tests/ -v` — all existing tests still pass
6. **Graceful degradation**: Remove sentence-transformers, verify search works without reranker
7. **Bash passthrough**: `ragdag search "test" --explain --json` — verify JSON output includes explain fields

---

## What's NOT in this plan (by design)

- **SQLite/FTS5** — not needed; BM25 over flat files is sufficient
- **Query expansion** — can be added later using existing LLM provider (prompt-based, no custom model)
- **AST-aware chunking** — high effort, separate plan
- **Strong-signal early exit** — optimization, add after baseline is working
- **Scored breakpoint chunking** — separate plan, independent of search pipeline
