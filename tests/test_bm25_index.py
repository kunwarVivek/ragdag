"""Tests for BM25 inverted index — O(k) search instead of O(n) file scan."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engines.bm25_index import build_index, load_index, update_index, query_index


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


INDEX_FILE = "_bm25_index.json"


# ---------------------------------------------------------------------------
# build_index tests
# ---------------------------------------------------------------------------

class TestBuildIndex:
    """build_index creates _bm25_index.json with correct structure."""

    def test_creates_index_file(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "hello world",
        })
        build_index(store)
        assert (Path(store) / INDEX_FILE).exists()

    def test_index_contains_terms(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "python coding rocks",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert "python" in idx["terms"]
        assert "coding" in idx["terms"]
        assert "rocks" in idx["terms"]

    def test_term_frequencies_correct(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "python python python java",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert idx["terms"]["python"]["d1/01.txt"] == 3
        assert idx["terms"]["java"]["d1/01.txt"] == 1

    def test_doc_lengths_tracked(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "hello world",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert "d1/01.txt" in idx["docs"]
        assert idx["docs"]["d1/01.txt"]["len"] > 0

    def test_short_words_excluded(self, tmp_path):
        """Words with length < 2 are not indexed."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "a b c python is great",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert "a" not in idx["terms"]
        assert "b" not in idx["terms"]
        assert "c" not in idx["terms"]
        assert "python" in idx["terms"]

    def test_avg_dl_and_n(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "hello world",
            "d2/01.txt": "foo bar baz",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert idx["n"] == 2
        assert idx["avg_dl"] > 0

    def test_synthesis_frontmatter_stripped(self, tmp_path):
        """Frontmatter is stripped; only body is indexed."""
        store = _build_store(tmp_path, {
            "d1/_summary.txt": "---\ntype: summary\nstale: false\n---\nactual content here",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        # frontmatter keywords should NOT be indexed
        assert "d1/_summary.txt" not in idx["terms"].get("type", {})
        # body keywords should be indexed
        assert "actual" in idx["terms"]
        assert "content" in idx["terms"]

    def test_synthesis_flags_tracked(self, tmp_path):
        """Synthesis nodes have synth=True, stale flag from frontmatter."""
        store = _build_store(tmp_path, {
            "d1/_summary.txt": "---\ntype: summary\nstale: true\n---\ncontent here",
            "d1/01.txt": "regular content",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert idx["docs"]["d1/_summary.txt"]["synth"] is True
        assert idx["docs"]["d1/_summary.txt"]["stale"] is True
        assert idx["docs"]["d1/01.txt"]["synth"] is False
        assert idx["docs"]["d1/01.txt"]["stale"] is False

    def test_hidden_files_skipped(self, tmp_path):
        """Files starting with . are skipped."""
        store = _build_store(tmp_path, {
            "d1/.hidden.txt": "secret data",
            "d1/01.txt": "visible data",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert "d1/.hidden.txt" not in idx["docs"]
        assert "d1/01.txt" in idx["docs"]

    def test_multiple_docs_same_term(self, tmp_path):
        """Term postings list includes all docs containing the term."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "python rocks",
            "d2/01.txt": "python rules",
            "d3/01.txt": "java only",
        })
        build_index(store)
        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert len(idx["terms"]["python"]) == 2
        assert "d1/01.txt" in idx["terms"]["python"]
        assert "d2/01.txt" in idx["terms"]["python"]


# ---------------------------------------------------------------------------
# load_index tests
# ---------------------------------------------------------------------------

class TestLoadIndex:

    def test_returns_none_when_no_index(self, tmp_path):
        store = tmp_path / ".ragdag"
        store.mkdir()
        assert load_index(str(store)) is None

    def test_returns_dict_when_index_exists(self, tmp_path):
        store = _build_store(tmp_path, {"d1/01.txt": "hello"})
        build_index(store)
        idx = load_index(store)
        assert isinstance(idx, dict)
        assert "terms" in idx


# ---------------------------------------------------------------------------
# update_index tests
# ---------------------------------------------------------------------------

class TestUpdateIndex:

    def test_adds_new_docs(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "original content",
        })
        build_index(store)

        # Add a new file
        _write(Path(store) / "d1/02.txt", "brand new document")
        update_index(store, ["d1/02.txt"])

        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert "d1/02.txt" in idx["docs"]
        assert "brand" in idx["terms"]
        assert "d1/02.txt" in idx["terms"]["brand"]

    def test_replaces_modified_docs(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "python programming language",
        })
        build_index(store)

        # Modify the file
        (Path(store) / "d1/01.txt").write_text("rust systems language")
        update_index(store, ["d1/01.txt"])

        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        # Old terms removed for this doc
        assert "d1/01.txt" not in idx["terms"].get("python", {})
        # New terms added
        assert "rust" in idx["terms"]
        assert "d1/01.txt" in idx["terms"]["rust"]

    def test_removes_deleted_docs(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "content to delete",
            "d1/02.txt": "content to keep",
        })
        build_index(store)

        # Delete the file
        (Path(store) / "d1/01.txt").unlink()
        update_index(store, ["d1/01.txt"])

        idx = json.loads((Path(store) / INDEX_FILE).read_text())
        assert "d1/01.txt" not in idx["docs"]
        # Term entries for deleted doc are cleaned up
        for term_docs in idx["terms"].values():
            assert "d1/01.txt" not in term_docs

    def test_updates_avg_dl_and_n(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "short",
        })
        build_index(store)
        idx1 = json.loads((Path(store) / INDEX_FILE).read_text())

        _write(Path(store) / "d1/02.txt", "a much longer document with many words")
        update_index(store, ["d1/02.txt"])

        idx2 = json.loads((Path(store) / INDEX_FILE).read_text())
        assert idx2["n"] == 2
        assert idx2["avg_dl"] != idx1["avg_dl"]


# ---------------------------------------------------------------------------
# query_index tests
# ---------------------------------------------------------------------------

class TestQueryIndex:

    def test_returns_none_when_no_index(self, tmp_path):
        """No index file => None (caller falls back to file scan)."""
        store = tmp_path / ".ragdag"
        store.mkdir()
        result = query_index(str(store), "anything")
        assert result is None

    def test_empty_query_returns_empty(self, tmp_path):
        store = _build_store(tmp_path, {"d1/01.txt": "hello world"})
        build_index(store)
        assert query_index(store, "") == []

    def test_single_char_query_returns_empty(self, tmp_path):
        store = _build_store(tmp_path, {"d1/01.txt": "hello world"})
        build_index(store)
        assert query_index(store, "a b c") == []

    def test_basic_search_returns_results(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "python programming language",
            "d1/02.txt": "java enterprise framework",
        })
        build_index(store)
        results = query_index(store, "python")
        assert len(results) == 1
        assert "d1/01.txt" in results[0][0]
        assert results[0][1] > 0

    def test_idf_prefers_rare_terms(self, tmp_path):
        """Rare term should give higher score."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "common common common common",
            "d2/01.txt": "common common common common",
            "d3/01.txt": "common common common common",
            "d4/01.txt": "common rare unique special",
        })
        build_index(store)
        results = query_index(store, "rare")
        assert len(results) == 1
        assert "d4/01.txt" in results[0][0]

    def test_domain_filtering(self, tmp_path):
        store = _build_store(tmp_path, {
            "physics/doc1/01.txt": "quantum entanglement theory",
            "biology/doc1/01.txt": "quantum mechanics in biology",
        })
        build_index(store)
        results = query_index(store, "quantum", domain="physics")
        assert len(results) == 1
        assert "physics" in results[0][0]

    def test_synthesis_boost(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "knowledge graph patterns",
            "d1/_summary.txt": "---\ntype: summary\nstale: false\n---\nknowledge graph patterns",
        })
        build_index(store)
        results = query_index(store, "knowledge graph", synthesis_boost=1.5)
        assert len(results) == 2
        assert "_summary.txt" in results[0][0]

    def test_stale_penalty(self, tmp_path):
        store = _build_store(tmp_path, {
            "d1/01.txt": "knowledge graph patterns extra words for length",
            "d1/_summary.txt": "---\ntype: summary\nstale: true\n---\nknowledge graph patterns extra words for length",
        })
        build_index(store)
        results = query_index(store, "knowledge graph", synthesis_boost=1.2, stale_penalty=0.3)
        assert len(results) == 2
        assert "01.txt" in results[0][0]

    def test_top_k_limits(self, tmp_path):
        files = {f"d{i}/01.txt": f"target word doc{i}" for i in range(10)}
        store = _build_store(tmp_path, files)
        build_index(store)
        results = query_index(store, "target", top_k=3)
        assert len(results) == 3

    def test_multi_term_accumulation(self, tmp_path):
        """Doc matching more query terms should score higher."""
        store = _build_store(tmp_path, {
            "d1/01.txt": "alpha content here",
            "d2/01.txt": "alpha beta content here",
            "d3/01.txt": "alpha beta gamma content here",
        })
        build_index(store)
        results = query_index(store, "alpha beta gamma")
        assert len(results) == 3
        assert "d3/01.txt" in results[0][0]
        assert "d2/01.txt" in results[1][0]

    def test_scores_match_file_scan(self, tmp_path):
        """Index-based query should produce same rankings as file-scan BM25."""
        from engines.bm25 import bm25_search

        store = _build_store(tmp_path, {
            "d1/01.txt": "python programming language tutorial",
            "d2/01.txt": "java enterprise framework spring",
            "d3/01.txt": "python data science machine learning",
            "d1/_summary.txt": "---\ntype: summary\nstale: false\n---\npython overview guide",
        })
        query = "python programming"

        # File scan results
        scan_results = bm25_search(store, query)

        # Index results
        build_index(store)
        index_results = query_index(store, query)

        assert index_results is not None
        # Same number of results
        assert len(index_results) == len(scan_results)
        # Same ranking order
        scan_paths = [r[0] for r in scan_results]
        index_paths = [r[0] for r in index_results]
        assert scan_paths == index_paths
        # Scores should be very close (float precision)
        for (sp, ss), (ip, is_) in zip(scan_results, index_results):
            assert abs(ss - is_) < 0.01, f"Score mismatch for {sp}: scan={ss}, index={is_}"
