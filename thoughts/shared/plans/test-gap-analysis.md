# ragdag Test Coverage Gap Analysis

Generated: 2026-02-15
Current state: 195 tests across 11 files (up from 53 when TEST_PLAN.md was written)

---

## Methodology

Every test function in the 11 existing test files was read and cross-referenced against:
1. The TEST_PLAN.md gap requirements (~260 tests identified)
2. The production source code symbols (core.py, mcp.py, api.py, llm.py, embeddings.py, similarity.py, base.py, openai_engine.py, local_engine.py, embed_cli.py, search_cli.py, relate_cli.py, ask_cli.py)
3. The 12 bash scripts in lib/

Tests are marked as COVERED only if a corresponding test function was verified to exist in the test files. Each gap below was confirmed absent by reading all 11 test files.

---

## What Was Filled Since TEST_PLAN.md

142 new tests were added covering:
- Binary embedding format (28 tests) -- was 0, now fully covered
- Cosine similarity + vector search (23 tests) -- was 0, now fully covered
- Chunking strategies: heading/paragraph/function (24 tests) -- was 0, now fully covered
- File parsing + sanitization (24 tests) -- was 0, now fully covered
- FastAPI HTTP API (22 tests) -- was 0, now well covered
- Ask context assembly, domain rules, config, graph advanced (21 tests) -- was 0, now partially covered

---

## REMAINING GAPS

### 1. CRITICAL (Must Test) -- 18 tests

These represent core functionality that is untested and could cause data loss or silent corruption.

| # | Test Function Name | File (new or existing) | What It Tests | New Infrastructure Needed |
|---|-------------------|----------------------|---------------|--------------------------|
| 1 | `test_mcp_search` | `tests/test_mcp.py` (NEW) | MCP ragdag_search tool returns results | MCP test harness fixture |
| 2 | `test_mcp_ask` | `tests/test_mcp.py` (NEW) | MCP ragdag_ask tool returns answer/context | MCP test harness fixture |
| 3 | `test_hybrid_search_score_fusion` | `tests/test_search_hybrid.py` (NEW) | Hybrid search fuses keyword + vector scores correctly | Mock embedding engine returning deterministic vectors |
| 4 | `test_hybrid_fallback_to_keyword` | `tests/test_search_hybrid.py` (NEW) | Hybrid search degrades to keyword when no embeddings exist | None |
| 5 | `test_add_embed_failure_stores_chunks` | `tests/test_sdk_integration.py` | Embedding failure does not block chunk storage | Mock embedding engine that raises |
| 6 | `test_reingestion_preserves_manual_edges` | `tests/test_sdk_integration.py` | Re-adding file preserves user-created edges (e.g., related_to) | None |
| 7 | `test_relate_creates_edges` | `tests/test_advanced.py` | `relate()` creates related_to edges above threshold | Pre-written embeddings.bin with known similar vectors |
| 8 | `test_trace_max_depth_limit` | `tests/test_advanced.py` | Trace terminates at max depth (20) even without cycle | Build 25-hop chain in .edges |
| 9 | `test_open_nonexistent_store_raises` | `tests/test_sdk_integration.py` | `ragdag.open()` on missing path raises appropriate error | None |
| 10 | `test_verify_healthy_store` | `tests/test_maintenance.py` (NEW) | Verify on valid store reports no issues | Import maintain functions or call via subprocess |
| 11 | `test_gc_removes_orphaned_edges` | `tests/test_maintenance.py` (NEW) | GC removes edges pointing to nonexistent chunks | Same |
| 12 | `test_gc_removes_stale_processed` | `tests/test_maintenance.py` (NEW) | GC removes .processed entries for deleted files | Same |
| 13 | `test_repair_removes_orphaned_edges` | `tests/test_maintenance.py` (NEW) | Repair removes broken edge references | Same |
| 14 | `test_repair_preserves_valid_edges` | `tests/test_maintenance.py` (NEW) | Repair does not touch valid edges | Same |
| 15 | `test_config_get` | `tests/test_maintenance.py` (NEW) | Config get returns correct value (bash config.sh via SDK) | None (SDK _read_config covered, but bash config.sh is not) |
| 16 | `test_config_set` | `tests/test_maintenance.py` (NEW) | Config set updates value in .config file | Config write function or subprocess call |
| 17 | `test_embed_failure_chunks_preserved` | `tests/test_sdk_integration.py` | NFR-05: Embedding crash still stores chunks | Mock engine that raises |
| 18 | `test_no_eval_in_bash` | `tests/test_security_static.py` (NEW) | Static analysis: no `eval` of user content in lib/*.sh | `grep` or AST scan of bash files |

---

### 2. HIGH (Should Test) -- 38 tests

Important edge cases, integration points, and secondary features.

| # | Test Function Name | File | What It Tests | New Infrastructure |
|---|-------------------|------|---------------|-------------------|
| **MCP Server** | | | | |
| 19 | `test_mcp_add` | `tests/test_mcp.py` (NEW) | MCP ragdag_add tool ingests file | MCP test harness |
| 20 | `test_mcp_graph` | `tests/test_mcp.py` (NEW) | MCP ragdag_graph tool returns stats | MCP test harness |
| 21 | `test_mcp_neighbors` | `tests/test_mcp.py` (NEW) | MCP ragdag_neighbors tool returns edges | MCP test harness |
| 22 | `test_mcp_trace` | `tests/test_mcp.py` (NEW) | MCP ragdag_trace tool returns chain | MCP test harness |
| 23 | `test_mcp_error_handling` | `tests/test_mcp.py` (NEW) | MCP tools return error text on failure | MCP test harness |
| 24 | `test_mcp_store_context` | `tests/test_mcp.py` (NEW) | RAGDAG_STORE env var controls MCP store | MCP test harness |
| **Hybrid/SDK Search** | | | | |
| 25 | `test_hybrid_search_weight_config` | `tests/test_search_hybrid.py` (NEW) | keyword_weight/vector_weight config respected | Mock embedding engine |
| 26 | `test_hybrid_search_domain_filter` | `tests/test_search_hybrid.py` (NEW) | Hybrid search with domain= restricts results | Mock embedding engine |
| 27 | `test_keyword_search_case_insensitive` | `tests/test_sdk_integration.py` | "OAuth2" matches "oauth2" in keyword search | None |
| 28 | `test_keyword_results_ordered_by_score` | `tests/test_sdk_integration.py` | Results come back sorted descending | None |
| 29 | `test_keyword_scoring_more_matches_higher` | `tests/test_sdk_integration.py` | Doc with 3 keyword hits scores above doc with 1 | None |
| 30 | `test_keyword_search_empty_store` | `tests/test_sdk_integration.py` | Search on empty store returns [] | None |
| 31 | `test_search_result_fields` | `tests/test_sdk_integration.py` | SearchResult has .path, .score, .content, .domain | None |
| **Ask Pipeline** | | | | |
| 32 | `test_ask_llm_includes_citations` | `tests/test_advanced.py` | LLM answer includes source references | Mock LLM |
| 33 | `test_ask_records_query_edges` | `tests/test_advanced.py` | record_queries=true creates query edges in .edges | None |
| 34 | `test_ask_records_retrieved_edges` | `tests/test_advanced.py` | record_queries=true creates retrieved edges | None |
| 35 | `test_ollama_provider_mocked` | `tests/test_llm.py` | Ollama provider sends correct HTTP request | Mock urllib |
| 36 | `test_llm_missing_api_key_raises` | `tests/test_llm.py` | OpenAI/Anthropic without API key raises clearly | None |
| **Init** | | | | |
| 37 | `test_init_default_config_values` | `tests/test_sdk_integration.py` | Init creates .config with expected defaults | None |
| 38 | `test_init_idempotent_no_overwrite` | `tests/test_sdk_integration.py` | Re-init does not overwrite existing config | None |
| **Add Edge Cases** | | | | |
| 39 | `test_reingestion_removes_old_chunks` | `tests/test_sdk_integration.py` | Changed file replaces old chunk files | None |
| 40 | `test_add_empty_file` | `tests/test_sdk_integration.py` | Empty file produces no chunks (files=0 or chunks=0) | None |
| 41 | `test_add_flat_flag` | `tests/test_sdk_integration.py` | --flat stores without domain subdirectory | None |
| 42 | `test_add_domain_auto_batch_unsorted` | `tests/test_advanced.py` | domain='auto' sends unmatched files to unsorted/ | None |
| **Graph/Relate** | | | | |
| 43 | `test_relate_skips_below_threshold` | `tests/test_advanced.py` | relate() does not create edges for low similarity | Pre-written embeddings |
| 44 | `test_relate_skips_existing` | `tests/test_advanced.py` | relate() does not duplicate existing edges | Pre-written embeddings |
| 45 | `test_relate_domain_scope` | `tests/test_advanced.py` | relate(domain=X) only processes domain X | Pre-written embeddings |
| **Embedding Engines** | | | | |
| 46 | `test_embedding_engine_interface` | `tests/test_embedding_engines.py` (NEW) | EmbeddingEngine ABC has embed/dimensions/model_name | None |
| 47 | `test_openai_engine_embed_mocked` | `tests/test_embedding_engines.py` (NEW) | OpenAIEngine.embed() returns correct dims (mocked) | Mock openai module |
| 48 | `test_engine_factory_selection` | `tests/test_embedding_engines.py` (NEW) | get_engine("openai") returns OpenAIEngine, etc. | Mock imports |
| **API Validation** | | | | |
| 49 | `test_api_validation_errors` | `tests/test_api.py` | POST /search without query returns 422 | None |
| **Maintenance** | | | | |
| 50 | `test_verify_detects_orphaned_edges` | `tests/test_maintenance.py` (NEW) | Verify flags edges pointing to missing chunks | Subprocess or import |
| 51 | `test_verify_detects_stale_processed` | `tests/test_maintenance.py` (NEW) | Verify flags .processed entries for deleted files | Same |
| 52 | `test_gc_preserves_valid_entries` | `tests/test_maintenance.py` (NEW) | GC does not remove valid edges | Same |
| 53 | `test_repair_healthy_noop` | `tests/test_maintenance.py` (NEW) | Repair on healthy store changes nothing | Same |
| 54 | `test_reindex_rebuilds_embeddings` | `tests/test_maintenance.py` (NEW) | Reindex recreates embeddings.bin | Mock embedding engine |
| 55 | `test_reindex_requires_provider` | `tests/test_maintenance.py` (NEW) | Reindex without embedding provider errors | None |
| **Security** | | | | |
| 56 | `test_api_keys_env_only` | `tests/test_llm.py` | API keys read from env, not config file | None |

---

### 3. MEDIUM (Nice to Have) -- 32 tests

Completeness, polish, and additional robustness.

| # | Test Function Name | File | What It Tests | New Infrastructure |
|---|-------------------|------|---------------|-------------------|
| **Init** | | | | |
| 57 | `test_init_explicit_path` | `tests/test_sdk_integration.py` | Init with non-cwd explicit path | None |
| 58 | `test_init_gitignore_in_git_repo` | `tests/test_sdk_integration.py` | Init adds .ragdag/ to .gitignore in git repos | git init in tmp_path |
| 59 | `test_init_no_gitignore_without_git` | `tests/test_sdk_integration.py` | No .gitignore created outside git repos | None |
| 60 | `test_init_gitignore_append_no_duplicate` | `tests/test_sdk_integration.py` | Re-init does not duplicate .gitignore entry | None |
| **Parsing** | | | | |
| 61 | `test_detect_type_pdf` | `tests/test_parse_security.py` | .pdf returns 'pdf' | None |
| 62 | `test_detect_type_html` | `tests/test_parse_security.py` | .html returns 'html' | None |
| 63 | `test_parse_code_passthrough` | `tests/test_parse_security.py` | .py content passes through unchanged | None |
| 64 | `test_parse_unknown_as_text` | `tests/test_parse_security.py` | Unknown extension parsed as text | None |
| 65 | `test_parse_failure_single_chunk` | `tests/test_parse_security.py` | Failed parse stores whole file as one chunk | Force parse error |
| 66 | `test_parse_pdf_mocked` | `tests/test_parse_security.py` | _parse_pdf calls pdftotext (mocked) | Mock subprocess |
| 67 | `test_parse_docx_mocked` | `tests/test_parse_security.py` | _parse_docx calls pandoc (mocked) | Mock subprocess |
| **Chunking** | | | | |
| 68 | `test_chunk_function_bash` | `tests/test_chunk_strategies.py` | Bash function splitting (no specific pattern for bash functions exists) | None |
| 69 | `test_chunk_numbering_format` | `tests/test_chunk_strategies.py` | Stored chunks use 01.txt, 02.txt naming | None |
| **Search** | | | | |
| 70 | `test_keyword_search_multiword` | `tests/test_sdk_integration.py` | Multi-word query matches docs with all terms | None |
| 71 | `test_keyword_scoring_shorter_docs_higher` | `tests/test_sdk_integration.py` | TF-IDF: shorter docs score higher per match | None |
| 72 | `test_keyword_short_words_ignored` | `tests/test_sdk_integration.py` | Words < 2 chars filtered from query | None |
| **Ask** | | | | |
| 73 | `test_ask_custom_prompt_template` | `tests/test_advanced.py` | Custom prompt.txt file used in ask | Create prompt.txt in store |
| **Graph** | | | | |
| 74 | `test_neighbors_includes_metadata` | `tests/test_sdk_integration.py` | Neighbor dict includes metadata field | None |
| **Add** | | | | |
| 75 | `test_add_binary_file_skipped` | `tests/test_sdk_integration.py` | Binary file (e.g., .png) skipped or treated as text | Create binary file |
| 76 | `test_processed_records_source_hash` | `tests/test_sdk_integration.py` | .processed file has source path and content hash | None |
| **Maintenance** | | | | |
| 77 | `test_verify_detects_unreadable_chunks` | `tests/test_maintenance.py` (NEW) | Verify flags unreadable chunk files | Create unreadable file |
| 78 | `test_verify_detects_manifest_mismatch` | `tests/test_maintenance.py` (NEW) | Verify flags manifest/binary count mismatch | Corrupt manifest |
| 79 | `test_verify_detects_bad_magic` | `tests/test_maintenance.py` (NEW) | Verify flags invalid magic number in embeddings.bin | Corrupt binary |
| 80 | `test_gc_reports_counts` | `tests/test_maintenance.py` (NEW) | GC output reports removal counts | Same |
| 81 | `test_reindex_domain_filter` | `tests/test_maintenance.py` (NEW) | Reindex with domain= only reindexes that domain | Mock engine |
| 82 | `test_reindex_all_domains` | `tests/test_maintenance.py` (NEW) | Reindex --all processes all domains | Mock engine |
| 83 | `test_config_set_new_key` | `tests/test_maintenance.py` (NEW) | Config set creates new key in existing section | Subprocess |
| 84 | `test_config_set_new_section` | `tests/test_maintenance.py` (NEW) | Config set creates new section if missing | Subprocess |
| 85 | `test_config_show` | `tests/test_maintenance.py` (NEW) | Config show displays all settings | Subprocess |
| **Reliability** | | | | |
| 86 | `test_store_atomic_move` | `tests/test_sdk_integration.py` | Chunk storage uses atomic staging dir | Inspect intermediate state |
| 87 | `test_store_reingestion_atomic` | `tests/test_sdk_integration.py` | Re-ingestion swaps atomically (no partial state) | Same |
| **Embedding Engines** | | | | |
| 88 | `test_local_engine_embed_mocked` | `tests/test_embedding_engines.py` (NEW) | LocalEngine.embed() with mocked sentence-transformers | Mock module |

---

### Bash Tests (All Missing -- 0 of ~70 tests)

Zero `.bats` test files exist. Zero `bats-core` infrastructure is set up. This is a complete gap.

The following bash scripts have zero test coverage:

| Script | Functions to Test | Estimated Tests |
|--------|------------------|----------------|
| `lib/compat.sh` | ragdag_sanitize, ragdag_sha256, ragdag_find_store, ragdag_estimate_tokens, ragdag_realpath, ragdag_sed_i, ragdag_file_size, ragdag_has | 14 |
| `lib/config.sh` | ragdag_config get/set/show, INI parsing | 8 |
| `lib/init.sh` | Store structure creation, dependency check | 2 |
| `lib/parse.sh` | File type detection, markdown/CSV/JSON/code extraction | 9 |
| `lib/chunk.sh` | heading/paragraph/function/fixed strategies | 8 |
| `lib/store.sh` | Atomic staging, dedup, chunk numbering | 4 |
| `lib/search.sh` | Keyword search, JSON/human output formatting | 7 |
| `lib/graph.sh` | Summary, neighbors, trace | 5 |
| `lib/ask.sh` | Context assembly, LLM call, source extraction | 5 |
| `lib/add.sh` | Orchestration, domain routing, processed tracking | 4 |
| `lib/maintain.sh` | verify, repair, gc, reindex | 8 |
| `lib/serve.sh` | HTTP + MCP server launch | 2 |
| **Total** | | **~76** |

**Infrastructure needed for bash tests:**
- Install `bats-core` (brew install bats-core)
- Create `tests/bash/test_helper.bash` with setup/teardown functions
- Create temp ragdag stores in each test
- Source `lib/*.sh` scripts for unit testing

---

## Summary Matrix

| Category | Tests in Plan | Now Covered | Still Missing | Priority Breakdown |
|----------|--------------|-------------|---------------|-------------------|
| SDK init | 9 | 2 | 7 | 1 CRITICAL, 2 HIGH, 4 MEDIUM |
| SDK add | 27 | 7 | 7 | 2 CRITICAL, 4 HIGH, 1 MEDIUM |
| SDK search (keyword) | 11 | 4 | 7 | 0 CRITICAL, 4 HIGH, 3 MEDIUM |
| SDK search (vector) | 10 | 10 | 0 | -- |
| SDK search (hybrid) | 4 | 0 | 4 | 2 CRITICAL, 2 HIGH |
| SDK ask | 15 | 8 | 5 | 0 CRITICAL, 4 HIGH, 1 MEDIUM |
| SDK graph/trace | 15 | 11 | 4 | 1 CRITICAL, 3 HIGH |
| SDK relate | 4 | 0 | 4 | 1 CRITICAL, 3 HIGH |
| engines/llm.py | 11 | 10 | 3 | 0 CRITICAL, 2 HIGH, 0 MEDIUM |
| engines/embeddings.py | 15 | 15 | 0 | -- |
| engines/similarity.py | 6 | 6 | 0 | -- |
| engines/base+providers | 4 | 0 | 4 | 0 CRITICAL, 3 HIGH, 1 MEDIUM |
| server/api.py | 17 | 22 | 1 | 0 CRITICAL, 1 HIGH |
| server/mcp.py | 8 | 0 | 8 | 2 CRITICAL, 6 HIGH |
| Maintenance (Python) | 17 | 0 | 17 | 5 CRITICAL, 8 HIGH, 4 MEDIUM |
| Security | 8 | 6 | 2 | 1 CRITICAL, 1 HIGH |
| Reliability | 3 | 0 | 3 | 1 CRITICAL, 0 HIGH, 2 MEDIUM |
| Bash scripts | ~76 | 0 | ~76 | All unprioritized (needs bats-core) |
| **TOTAL** | **~260** | **195** | **~88 Python + ~76 bash** | 18 CRIT + 38 HIGH + 32 MED |

---

## New Test Files Needed

| File | Purpose | Estimated Tests | Key Dependencies |
|------|---------|----------------|-----------------|
| `tests/test_mcp.py` | MCP server tool tests | 8 | MCP SDK test utilities, `mcp` package |
| `tests/test_search_hybrid.py` | Hybrid search score fusion and fallback | 4 | Mock embedding engine fixture |
| `tests/test_maintenance.py` | verify, repair, gc, reindex, config set | 17 | subprocess for bash commands or direct import |
| `tests/test_embedding_engines.py` | Engine ABC, OpenAI/Local engine mocks, factory | 4 | Mock openai, mock sentence_transformers |
| `tests/test_security_static.py` | Static analysis: no eval in bash | 1 | grep/regex scan |
| `tests/bash/test_helper.bash` | Shared bats setup/teardown | 0 (infrastructure) | bats-core installed |
| `tests/bash/*.bats` | All bash script tests | ~76 | bats-core, test_helper.bash |

---

## New conftest.py Fixtures Needed

```python
# In tests/conftest.py -- add these:

@pytest.fixture
def mock_embedding_engine():
    """Mock embedding engine returning deterministic vectors."""
    # Returns fixed-dimension vectors based on text hash
    # Needed by: test_search_hybrid.py, test_maintenance.py (reindex)

@pytest.fixture
def populated_store_with_embeddings(tmp_store):
    """Store with docs AND pre-computed embeddings.bin."""
    # Needed by: test_search_hybrid.py, test_relate tests

@pytest.fixture
def mcp_client(tmp_store):
    """MCP test client wired to a temp store."""
    # Needed by: test_mcp.py
```

---

## Recommended Implementation Order

### Phase 1: CRITICAL gaps (18 tests, ~2-3 hours)
Priority: MCP server tests, hybrid search, data integrity

1. Create `tests/test_mcp.py` -- 2 CRITICAL + 6 HIGH tests
2. Create `tests/test_search_hybrid.py` -- 2 CRITICAL + 2 HIGH tests
3. Add to `tests/test_sdk_integration.py` -- open_nonexistent_store_raises, add_embed_failure, reingestion_preserves_edges
4. Add to `tests/test_advanced.py` -- relate_creates_edges, trace_max_depth
5. Create `tests/test_maintenance.py` -- verify/repair/gc (5 CRITICAL)
6. Create `tests/test_security_static.py` -- no_eval_in_bash

### Phase 2: HIGH gaps (38 tests, ~3-4 hours)
Fill out MCP, search, ask, engines, API validation, maintenance.

### Phase 3: MEDIUM gaps (32 tests, ~2-3 hours)
Init edge cases, parsing edge cases, additional maintenance, reliability.

### Phase 4: Bash tests (~76 tests, ~8+ hours)
Install bats-core, create infrastructure, systematically test all 12 bash scripts.
