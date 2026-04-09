# RAG Quality Stack Design

Four bottom-up enhancements to ragdag's retrieval and answer quality.

## Layer 1: Chunk Provenance Metadata

### Problem
Chunks are bare `.txt` files with no metadata about origin. Only synthesis nodes have frontmatter. Without provenance, the LLM can't assess source authority, and users can't navigate back to context.

### Design
Add YAML frontmatter to every chunk file:

```yaml
---
source: /absolute/path/to/original/file.md
heading: "## Deployment > ### Docker Setup"
position: 3
total: 7
strategy: heading
hash: a1b2c3d4
---
Chunk content here...
```

### Fields
- `source` — original file absolute path
- `heading` — heading breadcrumb (markdown) or function name (code)
- `position` / `total` — chunk N of M for sequential context
- `strategy` — which chunker produced this
- `hash` — content hash of source file at ingest time

### Changes Required
- `_chunk_text()` returns `List[ChunkResult]` (dataclass with text + metadata) instead of `List[str]`
- Chunk writing in `add()` writes frontmatter before content
- All chunk consumers (BM25, embeddings, search) strip frontmatter — pattern already exists for synthesis nodes via `read_body()` / `read_frontmatter()` in `synthesis.py`
- Generalize `synthesis.py` frontmatter functions to handle chunk provenance too

### Backward Compatibility
- Existing bare chunks (no frontmatter) continue to work — `read_body()` returns full text when no frontmatter found
- Re-ingest adds provenance; old chunks degrade gracefully

---

## Layer 2: Proposition Chunking

### Problem
Current chunkers (heading, paragraph, function, fixed) split by formatting, not meaning. A single chunk may contain multiple unrelated claims, diluting retrieval precision.

### Design
Two-pass approach:
1. Structural chunker produces coarse chunks (existing behavior)
2. LLM decomposes each coarse chunk into atomic, self-contained propositions

### Proposition Properties
- Self-contained: understandable without surrounding context
- Atomic: one fact/claim per proposition
- Decontextualized: pronouns resolved, references made explicit

### File Layout
```
.ragdag/domain/doc/
  01.txt          # coarse parent chunk (kept for context expansion)
  01_p01.txt      # proposition 1 from chunk 01
  01_p02.txt      # proposition 2 from chunk 01
  02.txt          # coarse parent chunk
  02_p01.txt      # proposition from chunk 02
```

### Configuration
```ini
[general]
chunk_strategy = proposition   # opt-in
```

### Fallback
If LLM provider is `none`, proposition strategy degrades to sentence splitting via regex.

### LLM Prompt
```
Decompose this text into self-contained factual statements.
Each statement should be understandable without context.
Resolve pronouns and references. Return one statement per line.

Text:
{chunk_text}
```

---

## Layer 3: HyDE Query Transformation

### Problem
Short queries ("How do I deploy?") produce poor vector embeddings compared to the longer document chunks they should match. Query-document length mismatch hurts vector recall.

### Design
Before vector embedding, generate a hypothetical answer and embed that instead.

```
search(query)
  ├─ BM25: uses original query (exact term matching)
  ├─ Vector: uses HyDE-expanded query (semantic matching)
  └─ RRF fuses both
```

### Implementation
- New function: `_hyde_expand(query, provider, model) -> str`
- Prompt: "Write a short passage (3-4 sentences) that would answer: {query}"
- Applied in both `_python_search()` (SDK) and `cmd_search()` (CLI)
- BM25 always gets the raw query — HyDE only affects vector path

### Caching
- Query hash → `.ragdag/.hyde_cache/{hash}.txt`
- Eliminates redundant LLM calls for repeated queries

### Configuration
```ini
[search]
hyde = false   # opt-in, requires LLM
```

### Graceful Degradation
If LLM call fails, fall back to raw query embedding silently.

---

## Layer 4: CRAG Loop (Corrective RAG)

### Problem
`ask()` is one-shot: search → context → LLM → answer. No validation that retrieved context actually answers the question. Weak context produces hallucinated answers.

### Design
Add relevance gate + retry between retrieval and answer generation:

```
ask(question)
  1. search(question) → results
  2. RELEVANCE CHECK → SUFFICIENT / PARTIAL / INSUFFICIENT
  3a. SUFFICIENT → answer generation (normal path)
  3b. PARTIAL → extract gap, generate sub-query, search again, merge results
  3c. INSUFFICIENT → reformulate query, retry once
      Still INSUFFICIENT → return "insufficient context" honestly
```

### Relevance Check Prompt
```
Given this question and search results, rate the context:
- SUFFICIENT: results clearly answer the question
- PARTIAL: some relevant info but gaps exist (state what's missing)
- INSUFFICIENT: results don't address the question

Question: {question}
Context: {first 2000 chars of assembled context}

Rating:
```

### Constraints
- Max 2 retrieval attempts (original + one reformulation)
- Config: `search.crag = false` (opt-in)
- Config: `search.crag_max_retries = 1`

### AskResult Changes
```python
@dataclass
class AskResult:
    answer: Optional[str]
    context: str
    sources: List[str] = field(default_factory=list)
    confidence: str = "unknown"       # NEW: sufficient/partial/insufficient
    retrieval_attempts: int = 1       # NEW: how many search rounds
```

### Interaction with Other Layers
- HyDE improves initial retrieval → fewer CRAG retries needed
- Provenance metadata enables honest "found X in section Y but it doesn't address Z"
- Proposition chunks give more precise matches → higher SUFFICIENT rate

---

## Implementation Order

| Phase | Layer | Files Modified | New Files | Tests |
|-------|-------|---------------|-----------|-------|
| 1 | Provenance | `core.py`, `synthesis.py` | — | `test_provenance.py` |
| 2 | Propositions | `core.py` | — | `test_propositions.py` |
| 3 | HyDE | `core.py`, `search_cli.py` | — | `test_hyde.py` |
| 4 | CRAG | `core.py` | — | `test_crag.py` |

Each phase is independently testable and backward-compatible via config toggles.
