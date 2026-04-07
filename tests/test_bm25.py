"""Tests for BM25 keyword scoring engine over flat .txt files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engines.bm25 import bm25_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_store(tmp_path, files: dict) -> str:
    """Create a temp store with {relative_path: content} mapping."""
    store = tmp_path / ".ragdag"
    for rel, content in files.items():
        _write(store / rel, content)
    return str(store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBM25Basic:
    """Core BM25 scoring behavior."""

    def test_empty_store_returns_empty(self, tmp_path):
        """Empty store directory yields no results."""
        store = tmp_path / ".ragdag"
        store.mkdir()
        assert bm25_search(str(store), "anything") == []

    def test_nonexistent_store_returns_empty(self, tmp_path):
        """Non-existent store path yields no results."""
        assert bm25_search(str(tmp_path / "nope"), "anything") == []

    def test_no_results_for_absent_terms(self, tmp_path):
        """Query terms not present in any doc yield no results."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "the quick brown fox",
            "d1/02.txt": "jumps over the lazy dog",
        })
        assert bm25_search(store, "elephant giraffe") == []

    def test_single_char_words_ignored(self, tmp_path):
        """Words with length < 2 are filtered from the query."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "a b c d e f g",
        })
        # All single-char words -- should return nothing
        assert bm25_search(store, "a b c") == []

    def test_mixed_single_and_multi_char(self, tmp_path):
        """Only words >= 2 chars participate in search."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "the fox jumped high",
        })
        results = bm25_search(store, "a fox")
        assert len(results) == 1
        assert "d1/01.txt" in results[0][0]


class TestBM25IDF:
    """IDF should prefer documents with rare terms."""

    def test_idf_prefers_rare_term(self, tmp_path):
        """A term appearing in only 1 of 4 docs should rank that doc first."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "common common common common",
            "d2/01.txt": "common common common common",
            "d3/01.txt": "common common common common",
            "d4/01.txt": "common rare unique special",
        })
        results = bm25_search(store, "rare")
        assert len(results) == 1
        assert "d4/01.txt" in results[0][0]

    def test_idf_weights_discriminative_term(self, tmp_path):
        """When querying 'common rare', the doc with 'rare' should rank first
        because 'rare' has much higher IDF weight."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "common word common word common word",
            "d2/01.txt": "common word common word common word",
            "d3/01.txt": "common word common word common word",
            "d4/01.txt": "common rare word",
        })
        results = bm25_search(store, "common rare")
        assert len(results) >= 1
        # d4 should be ranked first because 'rare' has high IDF
        assert "d4/01.txt" in results[0][0]


class TestBM25TFSaturation:
    """TF saturation: repeated terms should not dominate linearly."""

    def test_tf_saturation(self, tmp_path):
        """50 repeats should NOT score 10x higher than 5 repeats."""
        store = _build_store(tmp_path, {
            "d1/01.txt": " ".join(["target"] * 5),
            "d2/01.txt": " ".join(["target"] * 50),
        })
        results = bm25_search(store, "target")
        assert len(results) == 2

        scores = {r[0]: r[1] for r in results}
        s5 = scores[[k for k in scores if "d1" in k][0]]
        s50 = scores[[k for k in scores if "d2" in k][0]]

        # BM25 saturates -- ratio should be well below 10x
        assert s50 / s5 < 5.0, f"Expected saturation, got ratio {s50/s5:.2f}"
        # But higher count should still score higher
        assert s50 > s5


class TestBM25LengthNorm:
    """Document length normalization."""

    def test_shorter_doc_scores_higher(self, tmp_path):
        """With the same match count, a shorter doc should score higher."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "target word",
            "d2/01.txt": "target " + " ".join(["padding"] * 200),
        })
        results = bm25_search(store, "target")
        assert len(results) == 2
        # Shorter doc (d1) should rank first
        assert "d1/01.txt" in results[0][0]


class TestBM25MultiTerm:
    """Multi-term queries accumulate IDF-weighted scores."""

    def test_multi_term_accumulation(self, tmp_path):
        """Doc matching more query terms should score higher."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "alpha content here",
            "d2/01.txt": "alpha beta content here",
            "d3/01.txt": "alpha beta gamma content here",
        })
        results = bm25_search(store, "alpha beta gamma")
        assert len(results) == 3
        # d3 matches all three terms, should be first
        assert "d3/01.txt" in results[0][0]
        # d2 matches two terms, should be second
        assert "d2/01.txt" in results[1][0]


class TestBM25Synthesis:
    """Synthesis nodes (_prefix) get boosted; stale ones get penalized."""

    def test_synthesis_boost(self, tmp_path):
        """Synthesis nodes (starting with _) should score higher than identical
        regular chunks."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "knowledge graph patterns",
            "d1/_summary.txt": "---\ntype: summary\nstale: false\n---\nknowledge graph patterns",
        })
        results = bm25_search(store, "knowledge graph", synthesis_boost=1.5)
        assert len(results) == 2
        # Synthesis node should rank first due to boost
        assert "_summary.txt" in results[0][0]

    def test_stale_synthesis_penalized(self, tmp_path):
        """Stale synthesis nodes get boost * stale_penalty, which should
        reduce their score below a fresh identical chunk."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "knowledge graph patterns extra words for length",
            "d1/_summary.txt": "---\ntype: summary\nstale: true\n---\nknowledge graph patterns extra words for length",
        })
        results = bm25_search(
            store, "knowledge graph",
            synthesis_boost=1.2,
            stale_penalty=0.3,
        )
        assert len(results) == 2
        # Stale synthesis (1.2 * 0.3 = 0.36 multiplier) should rank BELOW regular
        assert "01.txt" in results[0][0]
        assert "_summary.txt" in results[1][0]

    def test_fresh_synthesis_beats_stale(self, tmp_path):
        """Fresh synthesis node should outscore a stale one."""
        store = _build_store(tmp_path, {
            "d1/_fresh.txt": "---\ntype: summary\nstale: false\n---\nspecial term here",
            "d1/_stale.txt": "---\ntype: summary\nstale: true\n---\nspecial term here",
        })
        results = bm25_search(store, "special term")
        assert len(results) == 2
        assert "_fresh.txt" in results[0][0]

    def test_frontmatter_stripped_from_scoring(self, tmp_path):
        """YAML frontmatter should not contribute to BM25 scoring."""
        store = _build_store(tmp_path, {
            "d1/_summary.txt": "---\ntype: summary\nstale: false\n---\nactual content here",
            "d1/01.txt": "type summary stale false",
        })
        # Search for frontmatter metadata keywords
        results = bm25_search(store, "type summary stale")
        # The regular chunk (d1/01.txt) should match; synthesis should also
        # match because 'summary' is not in frontmatter content... but 'type'
        # and 'stale' should NOT match in the synthesis node if stripped correctly
        scores = {r[0]: r[1] for r in results}
        synth_score = scores.get("d1/_summary.txt", 0)
        regular_score = scores.get("d1/01.txt", 0)
        # Regular doc has all three terms; synthesis has only partial match
        assert regular_score > synth_score


class TestBM25Domain:
    """Domain filtering."""

    def test_domain_filter(self, tmp_path):
        """When domain is specified, only files under that domain are searched."""
        store = _build_store(tmp_path, {
            "physics/doc1/01.txt": "quantum entanglement theory",
            "biology/doc1/01.txt": "quantum mechanics in biology",
        })
        results = bm25_search(store, "quantum", domain="physics")
        assert len(results) == 1
        assert "physics" in results[0][0]

    def test_no_domain_searches_all(self, tmp_path):
        """Without domain filter, all domains are searched."""
        store = _build_store(tmp_path, {
            "physics/doc1/01.txt": "quantum theory",
            "biology/doc1/01.txt": "quantum biology",
        })
        results = bm25_search(store, "quantum")
        assert len(results) == 2


class TestBM25TopK:
    """Top-K limiting."""

    def test_top_k_limits_results(self, tmp_path):
        """Results are limited to top_k."""
        files = {f"d{i}/01.txt": f"target word doc{i}" for i in range(10)}
        store = _build_store(tmp_path, files)
        results = bm25_search(store, "target", top_k=3)
        assert len(results) == 3
