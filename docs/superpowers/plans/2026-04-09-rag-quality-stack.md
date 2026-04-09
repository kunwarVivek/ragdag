# RAG Quality Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four bottom-up quality layers to ragdag — chunk provenance metadata, proposition chunking, HyDE query transformation, and corrective RAG (CRAG) — each behind config toggles for backward compatibility.

**Architecture:** Layer 1 (provenance) adds YAML frontmatter to all chunk files. Layer 2 (propositions) adds an LLM-based chunking strategy that decomposes text into atomic statements. Layer 3 (HyDE) transforms queries into hypothetical answers before vector embedding. Layer 4 (CRAG) adds a relevance gate + retry loop in `ask()` to prevent hallucination from weak context.

**Tech Stack:** Python 3.13, pytest, existing `engines/llm.py` for LLM calls, existing `engines/synthesis.py` frontmatter utilities.

---

### Task 1: ChunkMeta dataclass and frontmatter utilities

**Files:**
- Modify: `sdk/ragdag/core.py:1-9` (imports)
- Modify: `sdk/ragdag/core.py:12-18` (add ChunkMeta dataclass after SearchResult)
- Modify: `engines/synthesis.py:17-35` (generalize `write_synthesis_node` to also handle chunk provenance)
- Test: `tests/test_provenance.py` (create)

- [ ] **Step 1: Write failing tests for ChunkMeta and frontmatter round-trip**

Create `tests/test_provenance.py`:

```python
"""Tests for chunk provenance metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ragdag.core import ChunkMeta


class TestChunkMeta:
    def test_dataclass_fields(self):
        meta = ChunkMeta(
            source="/path/to/file.md",
            heading="## Deploy > ### Docker",
            position=3,
            total=7,
            strategy="heading",
            hash="a1b2c3d4",
        )
        assert meta.source == "/path/to/file.md"
        assert meta.heading == "## Deploy > ### Docker"
        assert meta.position == 3
        assert meta.total == 7
        assert meta.strategy == "heading"
        assert meta.hash == "a1b2c3d4"

    def test_defaults(self):
        meta = ChunkMeta(source="/f.md", position=1, total=1, strategy="heading", hash="abc")
        assert meta.heading == ""


class TestChunkFrontmatter:
    def test_write_and_read_chunk_frontmatter(self, tmp_path):
        from engines.synthesis import write_chunk_node, read_frontmatter, read_body

        chunk_file = tmp_path / "01.txt"
        meta = ChunkMeta(
            source="/src/readme.md",
            heading="## Overview",
            position=1,
            total=3,
            strategy="heading",
            hash="deadbeef",
        )
        write_chunk_node(chunk_file, "This is the chunk content.", meta)

        fm = read_frontmatter(chunk_file)
        assert fm is not None
        assert fm["type"] == "chunk"
        assert fm["source"] == "/src/readme.md"
        assert fm["heading"] == "## Overview"
        assert fm["position"] == "1"
        assert fm["total"] == "3"
        assert fm["strategy"] == "heading"
        assert fm["hash"] == "deadbeef"

        body = read_body(chunk_file)
        assert body == "This is the chunk content."

    def test_read_body_no_frontmatter(self, tmp_path):
        """Backward compat: bare chunk files still read correctly."""
        from engines.synthesis import read_body

        chunk_file = tmp_path / "01.txt"
        chunk_file.write_text("plain chunk text")
        assert read_body(chunk_file) == "plain chunk text"

    def test_read_frontmatter_none_for_bare_file(self, tmp_path):
        from engines.synthesis import read_frontmatter

        chunk_file = tmp_path / "01.txt"
        chunk_file.write_text("no frontmatter here")
        assert read_frontmatter(chunk_file) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_provenance.py -v`
Expected: FAIL — `ChunkMeta` not defined, `write_chunk_node` not defined

- [ ] **Step 3: Add ChunkMeta dataclass to core.py**

In `sdk/ragdag/core.py`, after the `SearchResult` dataclass (line 18), add:

```python
@dataclass
class ChunkMeta:
    source: str
    position: int
    total: int
    strategy: str
    hash: str
    heading: str = ""
```

- [ ] **Step 4: Add write_chunk_node to synthesis.py**

In `engines/synthesis.py`, after `write_synthesis_node` (line 35), add:

```python
def write_chunk_node(
    path: Path,
    content: str,
    meta,  # ChunkMeta from core — accept duck-typed to avoid circular import
) -> None:
    """Write a chunk file with provenance frontmatter."""
    frontmatter = (
        "---\n"
        f"type: chunk\n"
        f"source: {meta.source}\n"
        f"heading: {meta.heading}\n"
        f"position: {meta.position}\n"
        f"total: {meta.total}\n"
        f"strategy: {meta.strategy}\n"
        f"hash: {meta.hash}\n"
        "---\n"
    )
    path.write_text(frontmatter + content, encoding="utf-8")
```

- [ ] **Step 5: Update read_frontmatter to parse chunk fields**

In `engines/synthesis.py`, update `read_frontmatter()` (line 38-70). The current code returns `None` unless `"type" in result` — this already works for chunk nodes since we write `type: chunk`. The existing key-value parser handles our new fields. No code change needed — just verify.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_provenance.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 7: Run full test suite for regression**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest --tb=short -q`
Expected: All existing tests still pass

- [ ] **Step 8: Commit**

```bash
cd /Users/vivek/jet/ragdag
git add sdk/ragdag/core.py engines/synthesis.py tests/test_provenance.py
git commit -m "feat: add ChunkMeta dataclass and write_chunk_node for provenance"
```

---

### Task 2: Wire provenance into the ingest pipeline

**Files:**
- Modify: `sdk/ragdag/core.py:184-227` (chunk writing in `add()`)
- Modify: `sdk/ragdag/core.py:405-416` (`_chunk_text` return type)
- Modify: `sdk/ragdag/core.py:418-457` (`_chunk_heading` — track heading breadcrumbs)
- Test: `tests/test_provenance.py` (extend)

- [ ] **Step 1: Write failing tests for provenance in ingested chunks**

Append to `tests/test_provenance.py`:

```python
class TestIngestProvenance:
    def test_add_writes_frontmatter_to_chunks(self, tmp_path):
        """After add(), each chunk .txt has provenance frontmatter."""
        import ragdag
        from engines.synthesis import read_frontmatter, read_body

        dag = ragdag.init(str(tmp_path))
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Introduction\nHello world.\n\n# Details\nMore info here.")
        dag.add(str(md_file))

        store = tmp_path / ".ragdag"
        chunk_files = sorted(store.rglob("*.txt"))
        # Filter to only numbered chunk files (not .config, .edges, etc.)
        chunk_files = [f for f in chunk_files if f.name[0].isdigit()]
        assert len(chunk_files) >= 1

        for cf in chunk_files:
            fm = read_frontmatter(cf)
            assert fm is not None, f"No frontmatter in {cf.name}"
            assert fm["type"] == "chunk"
            assert fm["source"] == str(md_file.resolve())
            assert fm["strategy"] == "heading"
            assert "position" in fm
            assert "total" in fm
            assert "hash" in fm

    def test_add_heading_breadcrumb(self, tmp_path):
        """Heading strategy captures the heading text in frontmatter."""
        import ragdag
        from engines.synthesis import read_frontmatter

        dag = ragdag.init(str(tmp_path))
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Chapter 1\nContent under chapter 1.\n\n# Chapter 2\nContent under chapter 2.")
        dag.add(str(md_file))

        store = tmp_path / ".ragdag"
        chunk_files = sorted(
            f for f in store.rglob("*.txt") if f.name[0].isdigit()
        )
        assert len(chunk_files) >= 2

        fm1 = read_frontmatter(chunk_files[0])
        assert "Chapter 1" in fm1.get("heading", "")

    def test_provenance_position_and_total(self, tmp_path):
        """position and total reflect chunk ordering."""
        import ragdag
        from engines.synthesis import read_frontmatter

        dag = ragdag.init(str(tmp_path))
        md_file = tmp_path / "multi.md"
        md_file.write_text("# A\nFirst\n\n# B\nSecond\n\n# C\nThird")
        dag.add(str(md_file))

        store = tmp_path / ".ragdag"
        chunk_files = sorted(
            f for f in store.rglob("*.txt") if f.name[0].isdigit()
        )
        assert len(chunk_files) == 3

        fm1 = read_frontmatter(chunk_files[0])
        fm3 = read_frontmatter(chunk_files[2])
        assert fm1["position"] == "1"
        assert fm1["total"] == "3"
        assert fm3["position"] == "3"
        assert fm3["total"] == "3"

    def test_bm25_strips_frontmatter(self, tmp_path):
        """BM25 search should strip frontmatter and only search body content."""
        import ragdag

        dag = ragdag.init(str(tmp_path))
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Deployment\nUse docker-compose to deploy the application.")
        dag.add(str(md_file))

        results = dag.search("deploy", mode="keyword")
        assert len(results) >= 1
        # Content should not contain frontmatter markers
        for r in results:
            assert "---\ntype: chunk" not in r.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_provenance.py::TestIngestProvenance -v`
Expected: FAIL — chunks don't have frontmatter yet

- [ ] **Step 3: Modify _chunk_text to return List of (text, heading) tuples**

We need heading info from the chunker. Rather than changing the return type of `_chunk_text` (which would break many callers), we add a parallel method. In `sdk/ragdag/core.py`, after `_chunk_text` (line 416), add:

```python
    def _chunk_text_with_meta(
        self, text: str, chunk_size: int, overlap: int, strategy: str = "heading"
    ) -> List[tuple]:
        """Split text into chunks, returning (text, heading) tuples."""
        if strategy == "heading":
            return self._chunk_heading_with_meta(text, chunk_size, overlap)
        else:
            # Other strategies don't have heading info
            plain_chunks = self._chunk_text(text, chunk_size, overlap, strategy)
            return [(c, "") for c in plain_chunks]
```

- [ ] **Step 4: Add _chunk_heading_with_meta**

In `sdk/ragdag/core.py`, after `_chunk_heading` (line 457), add:

```python
    def _chunk_heading_with_meta(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[tuple]:
        """Split on markdown headings, returning (text, heading) tuples."""
        lines = text.split("\n")
        chunks = []
        buffer = []
        buffer_len = 0
        current_heading = ""

        for line in lines:
            is_header = line.startswith("#")

            if is_header:
                if buffer_len > 0:
                    chunk_text = "\n".join(buffer)
                    if chunk_text.strip():
                        chunks.append((chunk_text, current_heading))
                    if overlap > 0:
                        buffer = [chunk_text[-overlap:], line]
                    else:
                        buffer = [line]
                    buffer_len = sum(len(b) for b in buffer)
                current_heading = line.strip()
                if not buffer:
                    buffer = [line]
                    buffer_len = len(line) + 1
                continue

            buffer.append(line)
            buffer_len += len(line) + 1

            if buffer_len >= chunk_size:
                chunk_text = "\n".join(buffer)
                if chunk_text.strip():
                    chunks.append((chunk_text, current_heading))
                if overlap > 0:
                    buffer = [chunk_text[-overlap:]]
                else:
                    buffer = []
                buffer_len = sum(len(b) for b in buffer)

        if buffer:
            chunk_text = "\n".join(buffer)
            if chunk_text.strip():
                chunks.append((chunk_text, current_heading))

        return chunks
```

- [ ] **Step 5: Modify add() to write provenance frontmatter**

In `sdk/ragdag/core.py`, replace the chunk writing section in `add()` (lines 196-226) with:

```python
            # Chunk (with metadata)
            chunk_tuples = self._chunk_text_with_meta(text, chunk_size, chunk_overlap, strategy)
            if not chunk_tuples:
                chunk_tuples = [(text, "")]
            chunks = [t[0] for t in chunk_tuples]

            # Determine doc name and domain
            doc_name = _sanitize(file.stem) or "document"
            file_domain = domain or ""
            if file_domain == "auto":
                file_domain = self._apply_domain_rules(str(abs_path)) or "unsorted"

            # Store chunks
            if file_domain:
                target_dir = self._store / file_domain / doc_name
            else:
                target_dir = self._store / doc_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Mark existing synthesis nodes as stale before replacing chunks
            if target_dir.exists():
                self._mark_synthesis_stale(target_dir)

            # Remove old raw chunks (preserve _ synthesis nodes)
            for old in target_dir.glob("*.txt"):
                if not old.name.startswith("_"):
                    old.unlink()

            # Write new chunks with provenance
            total_in_doc = len(chunk_tuples)
            for i, (chunk_text, heading) in enumerate(chunk_tuples, 1):
                chunk_file = target_dir / f"{i:02d}.txt"
                meta = ChunkMeta(
                    source=str(abs_path),
                    heading=heading,
                    position=i,
                    total=total_in_doc,
                    strategy=strategy,
                    hash=content_hash,
                )
                sys.path.insert(0, str(self._ragdag_dir))
                from engines.synthesis import write_chunk_node
                write_chunk_node(chunk_file, chunk_text, meta)
```

- [ ] **Step 6: Update BM25 and search to strip chunk frontmatter**

The BM25 engine (`engines/bm25.py:46-69`) already strips frontmatter from synthesis nodes starting with `---\n`. The same logic works for chunk frontmatter — no change needed in `bm25.py`.

The legacy keyword search in `core.py` (lines 923-928) also already strips frontmatter. Verify no changes needed.

For `_keyword_search` at `core.py:893-902`, content is read raw — add frontmatter stripping:

```python
        results = []
        for path, score in bm25_results:
            full_path = self._store / path
            content = ""
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                # Strip frontmatter (provenance or synthesis)
                if content.startswith("---\n"):
                    end = content.find("\n---\n", 4)
                    if end != -1:
                        content = content[end + 5:]
```

Similarly, in `_python_search` at line 1055, add the same stripping:

```python
        return [
            SearchResult(
                path=path,
                score=score,
                content=self._strip_frontmatter((self._store / path).read_text(encoding="utf-8")) if (self._store / path).exists() else "",
                domain=path.split("/")[0] if len(path.split("/")) >= 3 else "",
                explain=explain_map.get(path) if explain else None,
            )
            for path, score in final_results
        ]
```

Add helper method to `RagDag`:

```python
    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Strip YAML frontmatter from chunk or synthesis node text."""
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                return text[end + 5:]
        return text
```

- [ ] **Step 7: Run provenance tests**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_provenance.py -v`
Expected: PASS (all tests)

- [ ] **Step 8: Run full test suite for regression**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest --tb=short -q`
Expected: All tests pass. Watch for tests that compare raw chunk content — they may need updating since content now includes frontmatter that gets stripped.

- [ ] **Step 9: Commit**

```bash
cd /Users/vivek/jet/ragdag
git add sdk/ragdag/core.py engines/synthesis.py tests/test_provenance.py
git commit -m "feat: wire chunk provenance metadata into ingest pipeline"
```

---

### Task 3: Proposition chunking strategy

**Files:**
- Modify: `sdk/ragdag/core.py:405-416` (add proposition to `_chunk_text`)
- Modify: `sdk/ragdag/core.py` (add `_chunk_proposition` method)
- Test: `tests/test_propositions.py` (create)

- [ ] **Step 1: Write failing tests for proposition chunking**

Create `tests/test_propositions.py`:

```python
"""Tests for proposition chunking strategy."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from ragdag.core import RagDag


class TestPropositionChunking:
    def test_proposition_with_llm_mock(self, tmp_path):
        """Proposition strategy decomposes text into atomic statements via LLM."""
        dag = ragdag.init(str(tmp_path))

        # Mock LLM to return propositions
        mock_response = (
            "Docker containers run in isolation from each other.\n"
            "Kubernetes orchestrates Docker containers at scale.\n"
            "For local development, docker-compose is recommended."
        )
        with patch("engines.llm.call_llm", return_value=mock_response):
            text = "Docker containers run in isolation. Kubernetes orchestrates them at scale. For local dev, use docker-compose instead."
            results = dag._chunk_proposition(text, 1000, 0)
            assert len(results) == 3
            assert "Docker containers" in results[0][0]
            assert "Kubernetes" in results[1][0]
            assert "docker-compose" in results[2][0]

    def test_proposition_fallback_no_llm(self, tmp_path):
        """With provider=none, proposition falls back to sentence splitting."""
        dag = ragdag.init(str(tmp_path))
        text = "First sentence here. Second sentence follows. Third sentence ends."
        results = dag._chunk_proposition(text, 1000, 0)
        assert len(results) >= 2
        # Each result is (text, heading) tuple
        for chunk_text, heading in results:
            assert isinstance(chunk_text, str)
            assert len(chunk_text.strip()) > 0

    def test_proposition_strategy_in_chunk_text(self, tmp_path):
        """_chunk_text_with_meta routes 'proposition' strategy correctly."""
        dag = ragdag.init(str(tmp_path))
        text = "One fact. Another fact. Third fact."
        results = dag._chunk_text_with_meta(text, 1000, 0, "proposition")
        assert len(results) >= 2

    def test_proposition_ingest_creates_files(self, tmp_path):
        """add() with proposition strategy creates proposition chunk files."""
        dag = ragdag.init(str(tmp_path))

        # Set strategy to proposition in config
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("chunk_strategy = heading", "chunk_strategy = proposition")
        config_path.write_text(config_text)

        md_file = tmp_path / "doc.md"
        md_file.write_text("The sky is blue. Water is wet. Fire is hot.")
        dag.add(str(md_file))

        store = tmp_path / ".ragdag"
        chunk_files = sorted(f for f in store.rglob("*.txt") if f.name[0].isdigit())
        assert len(chunk_files) >= 2

    def test_proposition_preserves_heading_from_parent(self, tmp_path):
        """Propositions from a heading chunk inherit the heading."""
        dag = ragdag.init(str(tmp_path))
        text = "# Setup\nInstall Docker. Configure networking. Start services."

        mock_response = (
            "Docker should be installed.\n"
            "Networking needs to be configured.\n"
            "Services should be started."
        )
        with patch("engines.llm.call_llm", return_value=mock_response):
            # First get heading chunks, then decompose
            heading_chunks = dag._chunk_heading_with_meta(text, 5000, 0)
            assert len(heading_chunks) >= 1
            parent_text, parent_heading = heading_chunks[0]
            assert "Setup" in parent_heading

            props = dag._chunk_proposition(parent_text, 1000, 0)
            assert len(props) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_propositions.py -v`
Expected: FAIL — `_chunk_proposition` not defined

- [ ] **Step 3: Implement _chunk_proposition method**

In `sdk/ragdag/core.py`, after `_chunk_function` (find the end of that method), add:

```python
    def _chunk_proposition(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[tuple]:
        """Decompose text into atomic propositions via LLM, with sentence-split fallback.

        Returns list of (proposition_text, heading) tuples.
        Heading is empty string — caller should set from parent chunk if available.
        """
        llm_provider = self._read_config("llm.provider", "none")

        if llm_provider != "none":
            try:
                sys.path.insert(0, str(self._ragdag_dir))
                from engines.llm import call_llm

                llm_model = self._read_config("llm.model", "gpt-4o-mini")
                prompt = (
                    "Decompose this text into self-contained factual statements. "
                    "Each statement should be understandable without context. "
                    "Resolve pronouns and references. Return one statement per line. "
                    "Do not number them or add bullets.\n\n"
                    f"Text:\n{text}"
                )
                system_msg = "You are a text decomposition engine. Return only the propositions, one per line."
                raw = call_llm(system_msg, prompt, llm_provider, llm_model)
                propositions = [
                    line.strip() for line in raw.strip().splitlines()
                    if line.strip() and len(line.strip()) > 10
                ]
                if propositions:
                    return [(p, "") for p in propositions]
            except Exception:
                pass  # Fall through to sentence splitting

        # Fallback: regex sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return [(text, "")]
        return [(s, "") for s in sentences]
```

- [ ] **Step 4: Wire proposition into _chunk_text_with_meta**

In `sdk/ragdag/core.py`, update `_chunk_text_with_meta` to handle proposition:

```python
    def _chunk_text_with_meta(
        self, text: str, chunk_size: int, overlap: int, strategy: str = "heading"
    ) -> List[tuple]:
        """Split text into chunks, returning (text, heading) tuples."""
        if strategy == "heading":
            return self._chunk_heading_with_meta(text, chunk_size, overlap)
        elif strategy == "proposition":
            # Two-pass: heading chunks first, then decompose each
            heading_chunks = self._chunk_heading_with_meta(text, chunk_size, overlap)
            all_props = []
            for chunk_text, heading in heading_chunks:
                props = self._chunk_proposition(chunk_text, chunk_size, overlap)
                # Inherit heading from parent chunk
                all_props.extend([(p_text, heading) for p_text, _ in props])
            return all_props if all_props else heading_chunks
        else:
            plain_chunks = self._chunk_text(text, chunk_size, overlap, strategy)
            return [(c, "") for c in plain_chunks]
```

- [ ] **Step 5: Run proposition tests**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_propositions.py -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest --tb=short -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
cd /Users/vivek/jet/ragdag
git add sdk/ragdag/core.py tests/test_propositions.py
git commit -m "feat: add proposition chunking strategy with LLM decomposition"
```

---

### Task 4: HyDE query transformation

**Files:**
- Modify: `sdk/ragdag/core.py:955-1060` (`_python_search` — add HyDE before vector embedding)
- Modify: `engines/search_cli.py:54-140` (`cmd_search` — add HyDE for CLI path)
- Test: `tests/test_hyde.py` (create)

- [ ] **Step 1: Write failing tests for HyDE**

Create `tests/test_hyde.py`:

```python
"""Tests for HyDE (Hypothetical Document Embeddings) query transformation."""

import hashlib
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from ragdag.core import RagDag


class TestHyDE:
    def test_hyde_expand_generates_hypothetical(self, tmp_path):
        """_hyde_expand calls LLM to generate hypothetical answer."""
        dag = ragdag.init(str(tmp_path))

        # Enable HyDE + LLM
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)  # LLM only
        config_path.write_text(config_text)

        mock_answer = "To deploy, build the Docker image and push to the registry."
        with patch("engines.llm.call_llm", return_value=mock_answer):
            result = dag._hyde_expand("How do I deploy?")
            assert result == mock_answer

    def test_hyde_expand_returns_original_when_disabled(self, tmp_path):
        """With hyde=false (default), _hyde_expand returns the original query."""
        dag = ragdag.init(str(tmp_path))
        result = dag._hyde_expand("How do I deploy?")
        assert result == "How do I deploy?"

    def test_hyde_expand_returns_original_when_no_llm(self, tmp_path):
        """With provider=none, _hyde_expand returns original query."""
        dag = ragdag.init(str(tmp_path))

        # Enable HyDE but no LLM
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text += "\nhyde = true\n"
        config_path.write_text(config_text)

        result = dag._hyde_expand("How do I deploy?")
        assert result == "How do I deploy?"

    def test_hyde_caching(self, tmp_path):
        """Second call with same query uses cache, not LLM."""
        dag = ragdag.init(str(tmp_path))

        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)
        config_text += "\nhyde = true\n"
        config_path.write_text(config_text)

        mock_answer = "Deploy using Docker."
        with patch("engines.llm.call_llm", return_value=mock_answer) as mock_llm:
            result1 = dag._hyde_expand("How do I deploy?")
            result2 = dag._hyde_expand("How do I deploy?")
            assert result1 == result2 == mock_answer
            assert mock_llm.call_count == 1  # Only called once — second was cached

    def test_hyde_graceful_degradation(self, tmp_path):
        """If LLM call fails, _hyde_expand returns original query."""
        dag = ragdag.init(str(tmp_path))

        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)
        config_text += "\nhyde = true\n"
        config_path.write_text(config_text)

        with patch("engines.llm.call_llm", side_effect=Exception("API error")):
            result = dag._hyde_expand("How do I deploy?")
            assert result == "How do I deploy?"

    def test_hyde_config_default_false(self, tmp_path):
        """HyDE is disabled by default in config."""
        dag = ragdag.init(str(tmp_path))
        assert dag._read_config("search.hyde", "false") == "false"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_hyde.py -v`
Expected: FAIL — `_hyde_expand` not defined

- [ ] **Step 3: Add hyde config default**

In `sdk/ragdag/core.py`, in `_init_store()` (line 100-130), add `hyde = false` to the `[search]` section of the default config string:

```python
                "[search]\n"
                "default_mode = hybrid\n"
                "top_k = 10\n"
                "keyword_weight = 0.3\n"
                "vector_weight = 0.7\n"
                "rerank = false\n"
                "rerank_model = cross-encoder/ms-marco-MiniLM-L-6-v2\n"
                "hyde = false\n\n"
```

- [ ] **Step 4: Implement _hyde_expand method**

In `sdk/ragdag/core.py`, before the `_python_search` method (line 955), add:

```python
    def _hyde_expand(self, query: str) -> str:
        """Expand query into a hypothetical answer for better vector matching.

        Returns the original query if HyDE is disabled, LLM is unavailable,
        or the LLM call fails.
        """
        hyde_enabled = self._read_config("search.hyde", "false") == "true"
        if not hyde_enabled:
            return query

        llm_provider = self._read_config("llm.provider", "none")
        if llm_provider == "none":
            return query

        # Check cache
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        cache_dir = self._store / ".hyde_cache"
        cache_file = cache_dir / f"{query_hash}.txt"

        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        # Generate hypothetical answer
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.llm import call_llm

            llm_model = self._read_config("llm.model", "gpt-4o-mini")
            system_msg = "You are a document retrieval assistant. Write a short factual passage that would answer the given question. Do not add disclaimers."
            prompt = f"Write a short passage (3-4 sentences) that would answer this question:\n{query}"
            hypothetical = call_llm(system_msg, prompt, llm_provider, llm_model)

            if hypothetical and hypothetical.strip():
                # Cache result
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(hypothetical.strip(), encoding="utf-8")
                return hypothetical.strip()
        except Exception:
            pass

        return query
```

- [ ] **Step 5: Wire HyDE into _python_search**

In `sdk/ragdag/core.py`, in `_python_search` (line 979), change:

```python
        query_vec = engine.embed([query])[0]
```

to:

```python
        # HyDE: embed hypothetical answer for vector search, keep original for BM25
        hyde_query = self._hyde_expand(query)
        query_vec = engine.embed([hyde_query])[0]
```

No other changes needed — `query` (original) is still used for BM25 on line 987, and `query_vec` (HyDE-expanded) is used for vector search on line 993.

- [ ] **Step 6: Run HyDE tests**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_hyde.py -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest --tb=short -q`
Expected: All tests pass — HyDE is disabled by default so existing behavior unchanged

- [ ] **Step 8: Commit**

```bash
cd /Users/vivek/jet/ragdag
git add sdk/ragdag/core.py tests/test_hyde.py
git commit -m "feat: add HyDE query transformation for improved vector search"
```

---

### Task 5: CRAG (Corrective RAG) loop in ask()

**Files:**
- Modify: `sdk/ragdag/core.py:20-26` (`AskResult` — add confidence and retrieval_attempts fields)
- Modify: `sdk/ragdag/core.py:1066-1187` (`ask()` — add relevance gate + retry)
- Test: `tests/test_crag.py` (create)

- [ ] **Step 1: Write failing tests for CRAG**

Create `tests/test_crag.py`:

```python
"""Tests for CRAG (Corrective RAG) loop."""

import sys
from pathlib import Path
from unittest.mock import patch, call

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from ragdag.core import RagDag, AskResult


class TestAskResultFields:
    def test_ask_result_has_confidence(self):
        """AskResult has confidence field defaulting to 'unknown'."""
        result = AskResult(answer="test", context="ctx", sources=["a.txt"])
        assert result.confidence == "unknown"

    def test_ask_result_has_retrieval_attempts(self):
        """AskResult has retrieval_attempts field defaulting to 1."""
        result = AskResult(answer="test", context="ctx")
        assert result.retrieval_attempts == 1

    def test_ask_result_explicit_confidence(self):
        result = AskResult(answer="ok", context="c", confidence="sufficient", retrieval_attempts=2)
        assert result.confidence == "sufficient"
        assert result.retrieval_attempts == 2


class TestCRAGRelevanceCheck:
    def test_crag_sufficient_proceeds_normally(self, tmp_path):
        """When relevance check returns SUFFICIENT, ask() proceeds to answer."""
        dag = ragdag.init(str(tmp_path))

        # Enable CRAG + LLM
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)
        config_path.write_text(config_text + "\ncrag = true\n")

        # Ingest content
        doc = tmp_path / "deploy.md"
        doc.write_text("# Deploy\nUse docker-compose up to deploy the app.")
        dag.add(str(doc))

        call_count = 0
        def mock_llm(system_msg, user_msg, provider="openai", model="gpt-4o-mini"):
            nonlocal call_count
            call_count += 1
            if "rate the context" in system_msg.lower() or "rate the context" in user_msg.lower():
                return "SUFFICIENT"
            return "Use docker-compose up."

        with patch("engines.llm.call_llm", side_effect=mock_llm):
            result = dag.ask("How do I deploy?")
            assert result.confidence == "sufficient"
            assert result.retrieval_attempts == 1

    def test_crag_insufficient_reformulates(self, tmp_path):
        """When relevance returns INSUFFICIENT, ask() reformulates and retries."""
        dag = ragdag.init(str(tmp_path))

        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)
        config_path.write_text(config_text + "\ncrag = true\n")

        doc = tmp_path / "deploy.md"
        doc.write_text("# Deploy\nUse docker-compose up to deploy.")
        dag.add(str(doc))

        responses = iter([
            "INSUFFICIENT",          # First relevance check
            "deploy docker setup",   # Reformulated query
            "SUFFICIENT",            # Second relevance check
            "Run docker-compose up"  # Final answer
        ])

        def mock_llm(system_msg, user_msg, provider="openai", model="gpt-4o-mini"):
            return next(responses)

        with patch("engines.llm.call_llm", side_effect=mock_llm):
            result = dag.ask("How to set up?")
            assert result.retrieval_attempts == 2

    def test_crag_gives_up_after_max_retries(self, tmp_path):
        """After max retries, ask() returns with confidence=insufficient."""
        dag = ragdag.init(str(tmp_path))

        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)
        config_path.write_text(config_text + "\ncrag = true\n")

        doc = tmp_path / "other.md"
        doc.write_text("# Cooking\nRecipe for pasta.")
        dag.add(str(doc))

        def mock_llm(system_msg, user_msg, provider="openai", model="gpt-4o-mini"):
            if "rate" in user_msg.lower() or "rate" in system_msg.lower():
                return "INSUFFICIENT"
            if "reformulate" in system_msg.lower() or "search query" in user_msg.lower():
                return "deployment kubernetes"
            return "I don't have enough information."

        with patch("engines.llm.call_llm", side_effect=mock_llm):
            result = dag.ask("How do I deploy to k8s?")
            assert result.confidence == "insufficient"
            assert result.retrieval_attempts == 2

    def test_crag_disabled_by_default(self, tmp_path):
        """With crag=false (default), ask() works as before without relevance check."""
        dag = ragdag.init(str(tmp_path))
        assert dag._read_config("search.crag", "false") == "false"

    def test_crag_partial_triggers_sub_query(self, tmp_path):
        """PARTIAL response extracts gap and searches for missing info."""
        dag = ragdag.init(str(tmp_path))

        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)
        config_path.write_text(config_text + "\ncrag = true\n")

        doc = tmp_path / "doc.md"
        doc.write_text("# Architecture\nThe system uses microservices with REST APIs.")
        dag.add(str(doc))

        responses = iter([
            "PARTIAL: missing information about database layer",  # Relevance check
            "database schema microservices",                       # Sub-query
            "SUFFICIENT",                                          # Second check
            "The system uses microservices with REST APIs."        # Answer
        ])

        def mock_llm(system_msg, user_msg, provider="openai", model="gpt-4o-mini"):
            return next(responses)

        with patch("engines.llm.call_llm", side_effect=mock_llm):
            result = dag.ask("Describe the full architecture")
            assert result.retrieval_attempts == 2
            assert result.confidence == "sufficient"

    def test_crag_no_llm_skips_check(self, tmp_path):
        """With use_llm=False, CRAG is skipped even if enabled."""
        dag = ragdag.init(str(tmp_path))

        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_path.write_text(config_text + "\ncrag = true\n")

        doc = tmp_path / "doc.md"
        doc.write_text("# Test\nSome content here.")
        dag.add(str(doc))

        result = dag.ask("What is this?", use_llm=False)
        assert result.answer is None
        assert result.confidence == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_crag.py -v`
Expected: FAIL — AskResult missing `confidence` and `retrieval_attempts` fields

- [ ] **Step 3: Update AskResult dataclass**

In `sdk/ragdag/core.py`, update the `AskResult` dataclass (lines 21-26):

```python
@dataclass
class AskResult:
    answer: Optional[str]
    context: str
    sources: List[str] = field(default_factory=list)
    confidence: str = "unknown"
    retrieval_attempts: int = 1
```

- [ ] **Step 4: Add crag config defaults**

In `sdk/ragdag/core.py`, in `_init_store()`, add to the `[search]` section:

```python
                "crag = false\n"
                "crag_max_retries = 1\n\n"
```

- [ ] **Step 5: Implement _crag_relevance_check method**

In `sdk/ragdag/core.py`, before `ask()`, add:

```python
    def _crag_relevance_check(self, question: str, context: str) -> tuple:
        """Check if retrieved context is relevant to the question.

        Returns (rating, gap_description) where rating is 'sufficient',
        'partial', or 'insufficient', and gap_description is what's missing
        (empty string for 'sufficient').
        """
        sys.path.insert(0, str(self._ragdag_dir))
        from engines.llm import call_llm

        llm_provider = self._read_config("llm.provider", "none")
        llm_model = self._read_config("llm.model", "gpt-4o-mini")

        # Use first 2000 chars of context to keep costs low
        context_preview = context[:2000]

        system_msg = (
            "You are a relevance evaluator. Rate whether the provided context "
            "can answer the question. Respond with exactly one of:\n"
            "SUFFICIENT\n"
            "PARTIAL: <what information is missing>\n"
            "INSUFFICIENT\n"
            "Nothing else."
        )
        user_msg = f"Question: {question}\n\nContext:\n{context_preview}"

        try:
            raw = call_llm(system_msg, user_msg, llm_provider, llm_model)
            raw = raw.strip().upper() if raw else "INSUFFICIENT"

            if raw.startswith("SUFFICIENT"):
                return ("sufficient", "")
            elif raw.startswith("PARTIAL"):
                gap = raw.split(":", 1)[1].strip() if ":" in raw else ""
                return ("partial", gap)
            else:
                return ("insufficient", "")
        except Exception:
            # On failure, assume sufficient to avoid blocking
            return ("sufficient", "")

    def _crag_reformulate(self, question: str, gap: str) -> str:
        """Generate a reformulated search query based on identified gaps."""
        sys.path.insert(0, str(self._ragdag_dir))
        from engines.llm import call_llm

        llm_provider = self._read_config("llm.provider", "none")
        llm_model = self._read_config("llm.model", "gpt-4o-mini")

        system_msg = "Generate a concise search query to find the missing information. Return only the query, nothing else."
        if gap:
            user_msg = f"Original question: {question}\nMissing information: {gap}\nSearch query:"
        else:
            user_msg = f"Rephrase this question as a search query with key terms:\n{question}"

        try:
            result = call_llm(system_msg, user_msg, llm_provider, llm_model)
            return result.strip() if result else question
        except Exception:
            return question
```

- [ ] **Step 6: Modify ask() to include CRAG loop**

Replace `ask()` in `sdk/ragdag/core.py` (lines 1066-1187) with:

```python
    def ask(
        self,
        question: str,
        domain: Optional[str] = None,
        use_llm: bool = True,
    ) -> AskResult:
        """Ask a question using RAG, with optional CRAG validation loop."""
        crag_enabled = (
            self._read_config("search.crag", "false") == "true"
            and use_llm
            and self._read_config("llm.provider", "none") != "none"
        )
        max_retries = int(self._read_config("search.crag_max_retries", "1"))
        current_query = question
        all_results = []
        seen_paths = set()
        attempts = 0
        confidence = "unknown"

        for attempt in range(1 + max_retries):
            attempts = attempt + 1

            # Search
            results = self.search(current_query, mode="hybrid", domain=domain, top=10)

            # Merge with previous results (dedup by path)
            for r in results:
                if r.path not in seen_paths:
                    all_results.append(r)
                    seen_paths.add(r.path)

            if not all_results:
                if attempt < max_retries:
                    current_query = self._crag_reformulate(question, "no results found") if crag_enabled else question
                    continue
                return AskResult(answer=None, context="", sources=[], confidence="insufficient", retrieval_attempts=attempts)

            # Graph expansion: pull in related/referenced chunks
            expanded = self._expand_via_graph(all_results[:5], seen_paths)
            working_results = all_results + expanded

            # Build context
            context, sources = self._build_context(working_results)

            # CRAG relevance check
            if crag_enabled and attempt < max_retries:
                rating, gap = self._crag_relevance_check(question, context)
                confidence = rating

                if rating == "sufficient":
                    break
                elif rating == "partial":
                    current_query = self._crag_reformulate(question, gap)
                    continue
                else:  # insufficient
                    current_query = self._crag_reformulate(question, "")
                    continue
            else:
                if crag_enabled:
                    # Final attempt — do one last check
                    rating, _ = self._crag_relevance_check(question, context)
                    confidence = rating
                break

        if not use_llm:
            return AskResult(answer=None, context=context, sources=sources, confidence=confidence, retrieval_attempts=attempts)

        llm_provider = self._read_config("llm.provider", "none")
        if llm_provider == "none":
            return AskResult(answer=None, context=context, sources=sources, confidence=confidence, retrieval_attempts=attempts)

        sys.path.insert(0, str(self._ragdag_dir))
        from engines.llm import get_answer

        llm_model = self._read_config("llm.model", "gpt-4o-mini")

        answer = get_answer(
            question=question,
            context=context,
            provider=llm_provider,
            model=llm_model,
        )

        # File answer back into the store (if enabled)
        on_query = self._read_config("synthesis.on_query", "off")
        if on_query != "off" and answer:
            self._file_answer(question, answer, sources)

        return AskResult(answer=answer, context=context, sources=sources, confidence=confidence, retrieval_attempts=attempts)
```

- [ ] **Step 7: Extract _expand_via_graph and _build_context helpers**

These are extracted from the old `ask()` method. Add them before the new `ask()`:

```python
    def _expand_via_graph(self, results: List, seen_paths: set) -> List:
        """Pull in related chunks via graph edges."""
        edges_file = self._edges_path()
        expanded = []
        expansion_types = ("related_to", "references", "derived_from")

        # Try fast path via edge index
        use_index = False
        try:
            sys.path.insert(0, str(self._ragdag_dir))
            from engines.edge_index import load_edge_index, lookup_edges
            if self._edge_index_is_fresh() and load_edge_index(str(self._store)) is not None:
                use_index = True
        except ImportError:
            pass

        if use_index:
            for r in results:
                node_edges = lookup_edges(str(self._store), r.path)
                if not node_edges:
                    continue
                for e in node_edges:
                    if e["direction"] == "outgoing" and e["edge_type"] in expansion_types:
                        target = e["node"]
                        if target not in seen_paths:
                            target_file = self._store / target
                            if target_file.exists():
                                content = target_file.read_text(encoding="utf-8")
                                content = self._strip_frontmatter(content)
                                expanded.append(SearchResult(
                                    path=target, score=r.score * 0.8,
                                    content=content,
                                    domain=target.split("/")[0] if len(target.split("/")) >= 3 else "",
                                ))
                                seen_paths.add(target)
        elif edges_file.exists():
            edges_text = edges_file.read_text()
            for r in results:
                for line in edges_text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    source, target, etype = parts[0], parts[1], parts[2]
                    if source == r.path and etype in expansion_types:
                        if target not in seen_paths:
                            target_file = self._store / target
                            if target_file.exists():
                                content = target_file.read_text(encoding="utf-8")
                                content = self._strip_frontmatter(content)
                                expanded.append(SearchResult(
                                    path=target, score=r.score * 0.8,
                                    content=content,
                                    domain=target.split("/")[0] if len(target.split("/")) >= 3 else "",
                                ))
                                seen_paths.add(target)
        return expanded

    def _build_context(self, results: List) -> tuple:
        """Assemble context string from search results, respecting token limits."""
        max_context = int(self._read_config("llm.max_context", "8000"))
        context_parts = []
        sources = []
        tokens_used = 0

        for r in results:
            chunk_tokens = int(len(r.content.split()) * 1.3)
            if tokens_used + chunk_tokens > max_context:
                break
            context_parts.append(
                f"--- Source: {r.path} (score: {r.score:.4f}) ---\n{r.content}"
            )
            sources.append(r.path)
            tokens_used += chunk_tokens

        return "\n\n".join(context_parts), sources
```

- [ ] **Step 8: Run CRAG tests**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_crag.py -v`
Expected: PASS

- [ ] **Step 9: Run full test suite**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest --tb=short -q`
Expected: All tests pass — CRAG disabled by default, AskResult new fields have defaults

- [ ] **Step 10: Commit**

```bash
cd /Users/vivek/jet/ragdag
git add sdk/ragdag/core.py tests/test_crag.py
git commit -m "feat: add CRAG loop with relevance gate and query reformulation"
```

---

### Task 6: Integration test — full quality stack end-to-end

**Files:**
- Test: `tests/test_quality_stack.py` (create)

- [ ] **Step 1: Write integration test**

Create `tests/test_quality_stack.py`:

```python
"""Integration tests for the full RAG quality stack (provenance + propositions + HyDE + CRAG)."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from engines.synthesis import read_frontmatter, read_body


class TestFullStack:
    def _setup_dag_with_all_features(self, tmp_path):
        """Create a dag with all quality features enabled."""
        dag = ragdag.init(str(tmp_path))
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        # Keep LLM as none for most tests — mock when needed
        config_text = config_text.replace("chunk_strategy = heading", "chunk_strategy = proposition")
        config_path.write_text(config_text)
        return dag

    def test_provenance_survives_proposition_chunking(self, tmp_path):
        """Proposition chunks still have provenance frontmatter."""
        dag = self._setup_dag_with_all_features(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("# Overview\nThe sky is blue. Water is wet. Fire is hot.")
        dag.add(str(doc))

        store = tmp_path / ".ragdag"
        chunks = sorted(f for f in store.rglob("*.txt") if f.name[0].isdigit())
        assert len(chunks) >= 2

        for cf in chunks:
            fm = read_frontmatter(cf)
            assert fm is not None
            assert fm["type"] == "chunk"
            assert fm["source"] == str(doc.resolve())
            assert fm["strategy"] == "proposition"

    def test_search_returns_clean_content(self, tmp_path):
        """Search results have content without frontmatter artifacts."""
        dag = self._setup_dag_with_all_features(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("# Testing\nPytest is a testing framework for Python.")
        dag.add(str(doc))

        results = dag.search("pytest", mode="keyword")
        for r in results:
            assert "---\ntype:" not in r.content
            assert "source:" not in r.content.split("\n")[0] if r.content else True

    def test_backward_compat_bare_chunks(self, tmp_path):
        """Old-style bare chunks (no frontmatter) still work in search."""
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"

        # Manually create bare chunk (no frontmatter)
        doc_dir = store / "legacy"
        doc_dir.mkdir()
        (doc_dir / "01.txt").write_text("Kubernetes orchestrates containers at scale.")

        results = dag.search("kubernetes", mode="keyword")
        assert len(results) >= 1
        assert "Kubernetes" in results[0].content

    def test_ask_result_always_has_new_fields(self, tmp_path):
        """AskResult from ask() always includes confidence and retrieval_attempts."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "doc.md"
        doc.write_text("# Hello\nWorld content.")
        dag.add(str(doc))

        result = dag.ask("hello", use_llm=False)
        assert hasattr(result, "confidence")
        assert hasattr(result, "retrieval_attempts")
        assert result.retrieval_attempts >= 1
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest tests/test_quality_stack.py -v`
Expected: PASS

- [ ] **Step 3: Run full suite one final time**

Run: `cd /Users/vivek/jet/ragdag && python -m pytest --tb=short -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
cd /Users/vivek/jet/ragdag
git add tests/test_quality_stack.py
git commit -m "test: add integration tests for full RAG quality stack"
```
