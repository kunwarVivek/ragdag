# ragdag Comprehensive Test Plan

Generated: 2026-02-15

---

## 1. Executive Summary

**Current state**: 53 pytest tests across 4 test files. Zero bash (bats) tests.

**Coverage breakdown**:
- Python SDK core operations: PARTIAL (add, search keyword-only, ask no-LLM, graph, neighbors, trace, link)
- Python SDK chunking: `_chunk_fixed` only (10 tests)
- LLM integration: prompt structure and provider dispatch (8 tests)
- TSV exact matching: correctness guards (10 tests)
- SDK integration (init/add/search/ask/graph/neighbors/trace/link): 25 tests

**Major gaps**:
1. Zero bash script tests (0 of 12 scripts tested)
2. Zero server tests (HTTP API and MCP server completely untested)
3. Zero binary embedding format tests
4. Zero maintenance operation tests (verify, repair, gc, reindex)
5. Zero vector/hybrid search tests
6. Zero file parsing tests (markdown frontmatter, PDF, HTML, CSV, JSON)
7. Zero chunking strategy tests for heading/paragraph/function (only fixed tested)
8. Zero config management tests
9. Zero domain rules tests
10. Zero performance/benchmark tests

---

## 2. FR-to-Test Mapping

### 2.1 FR-INIT: Initialization

| FR ID | Requirement | Existing Tests | Status |
|-------|-------------|---------------|--------|
| FR-INIT-01 | Initialize a ragdag store | `test_init_creates_store`, `test_open_existing_store` | PARTIAL |

**EXISTING COVERAGE** (2 tests):
- `test_init_creates_store` -- verifies .ragdag dir and .config/.edges/.processed/.domain-rules exist
- `test_open_existing_store` -- verifies ragdag.open() returns correct store_dir

**GAP TESTS NEEDED**:

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Init creates correct default config values | `test_init_default_config_values` | HIGH | unit | SDK |
| Init on existing store is idempotent (no error, no overwrite) | `test_init_idempotent_no_overwrite` | HIGH | unit | SDK |
| Init with explicit non-cwd path | `test_init_explicit_path` | MEDIUM | unit | SDK |
| Init adds .ragdag/ to .gitignore when git repo detected | `test_init_gitignore_in_git_repo` | MEDIUM | integration | SDK |
| Init does not create .gitignore when not git repo | `test_init_no_gitignore_without_git` | MEDIUM | unit | SDK |
| Init appends to existing .gitignore without duplication | `test_init_gitignore_append_no_duplicate` | MEDIUM | unit | SDK |
| ragdag.open() on nonexistent store raises | `test_open_nonexistent_store_raises` | HIGH | unit | SDK |
| Bash: ragdag init creates store structure | `test_init_creates_structure` (bats) | CRITICAL | unit | bash/init.sh |
| Bash: ragdag init dependency check output | `test_init_dependency_check` (bats) | LOW | unit | bash/init.sh |

---

### 2.2 FR-ADD: Document Ingestion

| FR ID | Requirement | Existing Tests | Status |
|-------|-------------|---------------|--------|
| FR-ADD-01 | Ingest a file or directory | 7 tests in TestAdd + some in tsv_matching | PARTIAL |
| FR-ADD-02 | Idempotent re-ingestion | `test_add_idempotent`, `test_add_changed_file_reprocesses` | PARTIAL |
| FR-ADD-03 | Domain rules | None | NOT TESTED |

**EXISTING COVERAGE** (7 tests):
- `test_add_single_file` -- adds a .md file, checks file/chunk counts
- `test_add_idempotent` -- re-add same file, checks skipped=1
- `test_add_with_domain` -- checks chunks land under .ragdag/<domain>/
- `test_add_directory` -- adds dir with 2 files, checks files=2
- `test_add_creates_chunked_from_edges` -- checks "chunked_from" in .edges
- `test_add_nonexistent_file_raises` -- FileNotFoundError
- `test_add_changed_file_reprocesses` -- modified file reprocessed

**GAP TESTS NEEDED**:

#### File Type Detection (parse.sh / SDK _detect_file_type)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Detect .md as markdown | `test_detect_type_markdown` | HIGH | unit | SDK + bash |
| Detect .txt as text | `test_detect_type_text` | HIGH | unit | SDK + bash |
| Detect .pdf as pdf | `test_detect_type_pdf` | MEDIUM | unit | SDK + bash |
| Detect .html as html | `test_detect_type_html` | MEDIUM | unit | SDK + bash |
| Detect .csv as csv | `test_detect_type_csv` | MEDIUM | unit | SDK + bash |
| Detect .json as json | `test_detect_type_json` | MEDIUM | unit | SDK + bash |
| Detect .py as code | `test_detect_type_code` | MEDIUM | unit | SDK + bash |
| Detect unknown extension uses mime fallback | `test_detect_type_mime_fallback` | LOW | unit | SDK + bash |

#### File Parsing (parse.sh / SDK _parse_file)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Parse markdown strips YAML frontmatter | `test_parse_markdown_strips_frontmatter` | CRITICAL | unit | SDK + bash |
| Parse markdown preserves headers and content | `test_parse_markdown_preserves_content` | CRITICAL | unit | SDK + bash |
| Parse markdown without frontmatter passes through | `test_parse_markdown_no_frontmatter` | HIGH | unit | SDK + bash |
| Parse plain text passthrough | `test_parse_text_passthrough` | HIGH | unit | SDK + bash |
| Parse CSV to key-value format | `test_parse_csv_to_keyvalue` | HIGH | unit | SDK + bash |
| Parse JSON flattens to key-value | `test_parse_json_flatten` | HIGH | unit | SDK + bash |
| Parse code passthrough | `test_parse_code_passthrough` | MEDIUM | unit | SDK + bash |
| Parse HTML strips tags (fallback) | `test_parse_html_strip_tags` | MEDIUM | unit | SDK + bash |
| Parse unknown type treated as text | `test_parse_unknown_as_text` | LOW | unit | SDK + bash |
| Parse failure stores as single chunk | `test_parse_failure_single_chunk` | HIGH | integration | SDK |

#### Chunking Strategies (chunk.sh / SDK _chunk_text)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Heading chunking splits on # headers | `test_chunk_heading_splits_on_headers` | CRITICAL | unit | SDK + bash |
| Heading chunking respects chunk_size | `test_chunk_heading_respects_size` | HIGH | unit | SDK + bash |
| Heading chunking overlap | `test_chunk_heading_overlap` | HIGH | unit | SDK + bash |
| Heading chunking: no headers = single chunk | `test_chunk_heading_no_headers` | MEDIUM | unit | SDK + bash |
| Paragraph chunking splits on blank lines | `test_chunk_paragraph_splits_on_blanks` | CRITICAL | unit | SDK + bash |
| Paragraph chunking respects chunk_size | `test_chunk_paragraph_respects_size` | HIGH | unit | SDK + bash |
| Paragraph chunking overlap | `test_chunk_paragraph_overlap` | HIGH | unit | SDK + bash |
| Function chunking splits on def/class/function | `test_chunk_function_splits_on_defs` | CRITICAL | unit | SDK + bash |
| Function chunking: Python def | `test_chunk_function_python` | HIGH | unit | SDK + bash |
| Function chunking: JS/TS function | `test_chunk_function_javascript` | MEDIUM | unit | SDK + bash |
| Function chunking: Go func | `test_chunk_function_go` | MEDIUM | unit | SDK + bash |
| Function chunking: Rust fn | `test_chunk_function_rust` | MEDIUM | unit | SDK + bash |
| Function chunking: bash function | `test_chunk_function_bash` | MEDIUM | unit | SDK + bash |
| Function chunking safety flush at 2x chunk_size | `test_chunk_function_safety_flush` | HIGH | unit | SDK + bash |
| Auto-strategy selection: markdown uses heading | `test_strategy_auto_markdown_heading` | HIGH | integration | SDK |
| Auto-strategy selection: code uses function | `test_strategy_auto_code_function` | HIGH | integration | SDK |
| Chunk numbering format (01.txt, 02.txt, ...) | `test_chunk_numbering_format` | MEDIUM | unit | SDK + bash |

#### Domain Operations

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| --flat flag stores without domain subdirectory | `test_add_flat_flag` | HIGH | integration | SDK |
| --domain auto applies domain rules | `test_add_domain_auto_applies_rules` | HIGH | integration | SDK |
| --domain auto --batch sends unknowns to unsorted | `test_add_domain_auto_batch_unsorted` | HIGH | integration | SDK |
| Domain rules: pattern matching | `test_domain_rules_pattern_match` | CRITICAL | unit | SDK + bash |
| Domain rules: multiple patterns per rule | `test_domain_rules_multi_pattern` | HIGH | unit | SDK + bash |
| Domain rules: comments ignored | `test_domain_rules_comments_ignored` | MEDIUM | unit | SDK + bash |
| Domain rules: no match returns empty | `test_domain_rules_no_match` | HIGH | unit | SDK + bash |

#### Ingestion Edge Cases

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| --no-embed flag skips embedding | `test_add_no_embed_flag` | HIGH | integration | SDK |
| Embedding failure does not block chunk storage | `test_add_embed_failure_stores_chunks` | CRITICAL | integration | SDK |
| Filename sanitization on ingest | `test_add_filename_sanitization` | CRITICAL | unit | SDK + bash |
| Filename with spaces sanitized | `test_sanitize_spaces` | HIGH | unit | SDK + bash |
| Filename with special chars sanitized | `test_sanitize_special_chars` | HIGH | unit | SDK + bash |
| Re-ingestion preserves user-added edges | `test_reingestion_preserves_manual_edges` | CRITICAL | integration | SDK |
| Re-ingestion removes old chunks | `test_reingestion_removes_old_chunks` | HIGH | integration | SDK |
| .processed records source path and hash | `test_processed_records_source_hash` | HIGH | unit | SDK |
| Add empty file produces no chunks | `test_add_empty_file` | MEDIUM | unit | SDK |
| Add binary file skipped as unknown | `test_add_binary_file_skipped` | MEDIUM | unit | SDK |

---

### 2.3 FR-SEARCH: Search & Retrieval

| FR ID | Requirement | Existing Tests | Status |
|-------|-------------|---------------|--------|
| FR-SEARCH-01 | Keyword search | 4 tests in TestSearch | PARTIAL |
| FR-SEARCH-02 | Vector search | None | NOT TESTED |
| FR-SEARCH-03 | Hybrid search | None | NOT TESTED |
| FR-SEARCH-04 | Output formatting | None | NOT TESTED |

**EXISTING COVERAGE** (4 tests):
- `test_search_finds_matching_chunks` -- keyword search returns results with content
- `test_search_no_results` -- nonexistent term returns empty
- `test_search_with_domain_filter` -- results only from specified domain
- `test_search_respects_top_k` -- at most top_k results

**GAP TESTS NEEDED**:

#### Keyword Search Depth

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Keyword search is case-insensitive | `test_keyword_search_case_insensitive` | CRITICAL | unit | SDK + bash |
| Multi-word query matches documents with all terms | `test_keyword_search_multiword` | HIGH | unit | SDK + bash |
| TF-IDF scoring: more matches = higher score | `test_keyword_scoring_more_matches_higher` | HIGH | unit | SDK + bash |
| TF-IDF scoring: shorter docs score higher per match | `test_keyword_scoring_shorter_docs_higher` | MEDIUM | unit | SDK + bash |
| Results ordered by score descending | `test_keyword_results_ordered_by_score` | HIGH | unit | SDK |
| Short query words (<2 chars) ignored | `test_keyword_short_words_ignored` | MEDIUM | unit | SDK + bash |
| Empty store returns no results | `test_keyword_search_empty_store` | MEDIUM | unit | SDK + bash |

#### Vector Search

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Vector search returns results sorted by similarity | `test_vector_search_basic` | CRITICAL | unit | SDK |
| Vector search with domain filter | `test_vector_search_domain_filter` | HIGH | unit | SDK |
| Vector search respects top_k | `test_vector_search_top_k` | HIGH | unit | SDK |
| Vector search with mock embeddings | `test_vector_search_mock_embeddings` | CRITICAL | unit | SDK |
| Cosine similarity computation correctness | `test_cosine_similarity_correctness` | CRITICAL | unit | engines/similarity.py |
| Cosine similarity with zero vector | `test_cosine_similarity_zero_vector` | HIGH | unit | engines/similarity.py |
| Cosine similarity with identical vectors | `test_cosine_similarity_identical` | HIGH | unit | engines/similarity.py |
| Cosine similarity with orthogonal vectors | `test_cosine_similarity_orthogonal` | HIGH | unit | engines/similarity.py |
| search_vectors loads correct domain embeddings | `test_search_vectors_loads_domain` | HIGH | unit | engines/similarity.py |
| search_vectors candidate filtering | `test_search_vectors_candidate_filter` | HIGH | unit | engines/similarity.py |

#### Hybrid Search

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Hybrid search fuses keyword + vector scores | `test_hybrid_search_score_fusion` | CRITICAL | integration | SDK |
| Hybrid search respects keyword_weight/vector_weight | `test_hybrid_search_weight_config` | HIGH | integration | SDK |
| Hybrid search falls back to keyword when no embeddings | `test_hybrid_fallback_to_keyword` | CRITICAL | unit | SDK + bash |
| Hybrid search with domain filter | `test_hybrid_search_domain_filter` | HIGH | integration | SDK |

#### Output Formatting

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| SearchResult object has path, score, content, domain | `test_search_result_fields` | HIGH | unit | SDK |
| SDK search returns list of SearchResult objects | `test_search_returns_search_results` | HIGH | unit | SDK |
| Bash JSON output is valid JSON array | `test_search_json_output_valid` (bats) | HIGH | unit | bash |
| Bash human output shows rank, score, path | `test_search_human_output_format` (bats) | MEDIUM | unit | bash |

---

### 2.4 FR-ASK: Question Answering

| FR ID | Requirement | Existing Tests | Status |
|-------|-------------|---------------|--------|
| FR-ASK-01 | Ask a question | 2 tests + 1 in tsv_matching | PARTIAL |
| FR-ASK-02 | Context assembly | None | NOT TESTED |
| FR-ASK-03 | LLM integration | 8 tests in test_llm.py | PARTIAL |

**EXISTING COVERAGE** (3 + 8 tests):
- `test_ask_returns_context_no_llm` -- context has content, sources populated, answer is None
- `test_ask_no_results` -- empty context and sources on no match
- `test_ask_graph_expansion_exact_match` -- graph expansion uses exact matching
- test_llm.py: provider dispatch, prompt structure, system/user separation

**GAP TESTS NEEDED**:

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Ask with mocked LLM returns generated answer | `test_ask_with_mocked_llm` | CRITICAL | integration | SDK |
| Ask with LLM includes source citations | `test_ask_llm_includes_citations` | HIGH | integration | SDK |
| Context assembly respects max_context token budget | `test_ask_context_token_budget` | CRITICAL | unit | SDK |
| Context assembly stops adding chunks at budget | `test_ask_context_stops_at_budget` | HIGH | unit | SDK |
| Context format includes Source: path (score: n) | `test_ask_context_format` | MEDIUM | unit | SDK |
| Graph expansion adds related_to chunks | `test_ask_graph_expansion_related_to` | HIGH | integration | SDK |
| Graph expansion adds references chunks | `test_ask_graph_expansion_references` | HIGH | integration | SDK |
| Query recording when record_queries=true | `test_ask_records_query_edges` | HIGH | integration | SDK + bash |
| Query recording creates retrieved edges | `test_ask_records_retrieved_edges` | HIGH | integration | SDK + bash |
| Custom prompt template from prompt.txt | `test_ask_custom_prompt_template` | MEDIUM | integration | SDK |
| Ollama provider integration (mocked) | `test_ollama_provider_mocked` | HIGH | unit | engines/llm.py |
| LLM provider missing API key raises | `test_llm_missing_api_key_raises` | HIGH | unit | engines/llm.py |

---

### 2.5 FR-GRAPH: Knowledge Graph Operations

| FR ID | Requirement | Existing Tests | Status |
|-------|-------------|---------------|--------|
| FR-GRAPH-01 | Graph summary | 2 tests | PARTIAL |
| FR-GRAPH-02 | Neighbors | 4 tests | GOOD |
| FR-GRAPH-03 | Trace | 3 tests | PARTIAL |
| FR-GRAPH-04 | Relate | None | NOT TESTED |
| FR-GRAPH-05 | Manual edge creation | 2 tests | GOOD |

**EXISTING COVERAGE** (11 tests):
- `test_graph_stats` -- domains >= 1, chunks >= 1, edges >= 1, edge_types has chunked_from
- `test_graph_with_domain_filter` -- domain filter limits count
- `test_neighbors_outgoing` / `test_neighbors_incoming` -- directional neighbor lookup
- `test_neighbors_exact_source_match` / `test_neighbors_exact_target_match` -- exact matching
- `test_trace_follows_chunked_from` / `test_trace_no_edges` / `test_trace_exact_source_match`
- `test_link_creates_edge` / `test_link_default_edge_type`

**GAP TESTS NEEDED**:

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Graph summary: edge type breakdown counts | `test_graph_edge_type_counts` | MEDIUM | unit | SDK |
| Graph summary: document count accuracy | `test_graph_document_count` | MEDIUM | unit | SDK |
| Graph summary: empty store returns zeros | `test_graph_empty_store` | HIGH | unit | SDK |
| Neighbors: no edges for node returns empty | `test_neighbors_no_edges` | MEDIUM | unit | SDK |
| Neighbors: includes metadata field | `test_neighbors_includes_metadata` | LOW | unit | SDK |
| Trace: multi-hop (3+ levels deep) | `test_trace_multi_hop` | HIGH | unit | SDK |
| Trace: follows derived_via edges | `test_trace_derived_via` | HIGH | unit | SDK |
| Trace: cycle detection halts | `test_trace_cycle_detection` | CRITICAL | unit | SDK + bash |
| Trace: max_depth limit (20) | `test_trace_max_depth_limit` | MEDIUM | unit | bash |
| Relate: creates related_to edges above threshold | `test_relate_creates_edges` | CRITICAL | integration | SDK |
| Relate: skips pairs below threshold | `test_relate_skips_below_threshold` | HIGH | integration | SDK |
| Relate: skips existing edges | `test_relate_skips_existing` | HIGH | integration | SDK |
| Relate: domain scoping | `test_relate_domain_scope` | HIGH | integration | SDK |
| Relate: requires Python (bash error) | `test_relate_requires_python` (bats) | MEDIUM | unit | bash |

---

### 2.6 FR-SERVE: Server Interfaces

| FR ID | Requirement | Existing Tests | Status |
|-------|-------------|---------------|--------|
| FR-SERVE-01 | MCP server | None | NOT TESTED |
| FR-SERVE-02 | HTTP API server | None | NOT TESTED |
| FR-SERVE-03 | Python SDK | 25 tests | PARTIAL |

**GAP TESTS NEEDED**:

#### HTTP API (server/api.py) -- all endpoints

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| GET /health returns 200 | `test_api_health` | CRITICAL | unit | server/api.py |
| POST /add with file path | `test_api_add` | CRITICAL | integration | server/api.py |
| POST /add with domain | `test_api_add_with_domain` | HIGH | integration | server/api.py |
| POST /add nonexistent path returns error | `test_api_add_nonexistent` | HIGH | unit | server/api.py |
| POST /search keyword mode | `test_api_search_keyword` | CRITICAL | integration | server/api.py |
| POST /search with domain filter | `test_api_search_domain` | HIGH | integration | server/api.py |
| POST /search no results | `test_api_search_no_results` | HIGH | unit | server/api.py |
| POST /ask returns answer | `test_api_ask` | CRITICAL | integration | server/api.py |
| POST /ask with no LLM | `test_api_ask_no_llm` | HIGH | integration | server/api.py |
| GET /graph returns stats | `test_api_graph` | HIGH | integration | server/api.py |
| GET /graph?domain= filters | `test_api_graph_domain` | MEDIUM | integration | server/api.py |
| GET /neighbors/:node_path | `test_api_neighbors` | HIGH | integration | server/api.py |
| POST /link creates edge | `test_api_link` | HIGH | integration | server/api.py |
| GET /trace/:node_path | `test_api_trace` | HIGH | integration | server/api.py |
| POST /relate | `test_api_relate` | MEDIUM | integration | server/api.py |
| API request validation (missing fields) | `test_api_validation_errors` | HIGH | unit | server/api.py |
| API RAGDAG_STORE env var initialization | `test_api_store_init` | HIGH | unit | server/api.py |

#### MCP Server (server/mcp.py)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| MCP ragdag_search tool | `test_mcp_search` | CRITICAL | integration | server/mcp.py |
| MCP ragdag_ask tool | `test_mcp_ask` | CRITICAL | integration | server/mcp.py |
| MCP ragdag_add tool | `test_mcp_add` | HIGH | integration | server/mcp.py |
| MCP ragdag_graph tool | `test_mcp_graph` | HIGH | integration | server/mcp.py |
| MCP ragdag_neighbors tool | `test_mcp_neighbors` | HIGH | integration | server/mcp.py |
| MCP ragdag_trace tool | `test_mcp_trace` | HIGH | integration | server/mcp.py |
| MCP tool error handling | `test_mcp_error_handling` | HIGH | unit | server/mcp.py |
| MCP RAGDAG_STORE context var | `test_mcp_store_context` | HIGH | unit | server/mcp.py |

---

### 2.7 FR-MAINTAIN: Maintenance Operations

| FR ID | Requirement | Existing Tests | Status |
|-------|-------------|---------------|--------|
| FR-MAINTAIN-01 | Verify integrity | None | NOT TESTED |
| FR-MAINTAIN-02 | Repair | None | NOT TESTED |
| FR-MAINTAIN-03 | Garbage collection | None | NOT TESTED |
| FR-MAINTAIN-04 | Reindex | None | NOT TESTED |
| FR-MAINTAIN-05 | Config management | None | NOT TESTED |

**GAP TESTS NEEDED**:

#### Verify (maintain.sh ragdag_verify)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Verify healthy store reports no issues | `test_verify_healthy_store` | CRITICAL | integration | bash |
| Verify detects unreadable chunks | `test_verify_detects_unreadable_chunks` | HIGH | integration | bash |
| Verify detects manifest/binary mismatch | `test_verify_detects_manifest_mismatch` | HIGH | integration | bash |
| Verify detects orphaned edges | `test_verify_detects_orphaned_edges` | HIGH | integration | bash |
| Verify detects stale .processed entries | `test_verify_detects_stale_processed` | HIGH | integration | bash |
| Verify detects invalid magic number | `test_verify_detects_bad_magic` | MEDIUM | integration | bash |

#### Repair (maintain.sh ragdag_repair)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Repair removes orphaned edges | `test_repair_removes_orphaned_edges` | CRITICAL | integration | bash |
| Repair preserves valid edges | `test_repair_preserves_valid_edges` | CRITICAL | integration | bash |
| Repair on healthy store is no-op | `test_repair_healthy_noop` | HIGH | integration | bash |

#### Garbage Collection (maintain.sh ragdag_gc)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| GC removes orphaned edges | `test_gc_removes_orphaned_edges` | CRITICAL | integration | bash |
| GC removes stale .processed entries | `test_gc_removes_stale_processed` | CRITICAL | integration | bash |
| GC preserves valid entries | `test_gc_preserves_valid_entries` | HIGH | integration | bash |
| GC reports counts | `test_gc_reports_counts` | MEDIUM | integration | bash |

#### Reindex (maintain.sh ragdag_reindex)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Reindex rebuilds embeddings.bin | `test_reindex_rebuilds_embeddings` | HIGH | integration | bash |
| Reindex with domain filter | `test_reindex_domain_filter` | HIGH | integration | bash |
| Reindex --all processes all domains | `test_reindex_all_domains` | MEDIUM | integration | bash |
| Reindex requires embedding provider | `test_reindex_requires_provider` | HIGH | unit | bash |

#### Config Management (config.sh ragdag_config)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| config get returns correct value | `test_config_get` | CRITICAL | unit | bash + SDK |
| config set updates value | `test_config_set` | CRITICAL | unit | bash + SDK |
| config set creates new key in existing section | `test_config_set_new_key` | HIGH | unit | bash |
| config set creates new section | `test_config_set_new_section` | HIGH | unit | bash |
| config show displays all settings | `test_config_show` | MEDIUM | unit | bash |
| config get default when key missing | `test_config_get_default` | HIGH | unit | bash |
| INI parsing: comments skipped | `test_config_parse_comments` | MEDIUM | unit | bash |
| INI parsing: whitespace handling | `test_config_parse_whitespace` | MEDIUM | unit | bash |

---

### 2.8 Binary Embedding Format (engines/embeddings.py)

All tests in this section are NEW -- zero current coverage.

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| write_embeddings creates valid binary file | `test_write_embeddings_creates_file` | CRITICAL | unit | engines/embeddings.py |
| Binary file has correct magic number 0x52414744 | `test_embeddings_magic_number` | CRITICAL | unit | engines/embeddings.py |
| Binary file has correct format version | `test_embeddings_format_version` | HIGH | unit | engines/embeddings.py |
| Binary file has correct dimensions | `test_embeddings_dimensions` | HIGH | unit | engines/embeddings.py |
| Binary file has correct vector count | `test_embeddings_vector_count` | HIGH | unit | engines/embeddings.py |
| Model hash computed correctly | `test_model_hash_computation` | HIGH | unit | engines/embeddings.py |
| load_embeddings reads back written vectors | `test_load_embeddings_roundtrip` | CRITICAL | unit | engines/embeddings.py |
| load_embeddings_mmap reads back written vectors | `test_load_embeddings_mmap_roundtrip` | HIGH | unit | engines/embeddings.py |
| load_embeddings validates magic number | `test_load_embeddings_invalid_magic` | HIGH | unit | engines/embeddings.py |
| Manifest TSV created with correct entries | `test_manifest_created_correctly` | CRITICAL | unit | engines/embeddings.py |
| load_manifest reads manifest entries | `test_load_manifest_roundtrip` | HIGH | unit | engines/embeddings.py |
| Append mode adds vectors to existing file | `test_write_embeddings_append` | CRITICAL | unit | engines/embeddings.py |
| Append mode deduplicates by chunk path | `test_write_embeddings_append_dedup` | HIGH | unit | engines/embeddings.py |
| Write with empty vectors list | `test_write_embeddings_empty` | MEDIUM | unit | engines/embeddings.py |
| Header is 32 bytes (HEADER_SIZE constant) | `test_header_size_constant` | MEDIUM | unit | engines/embeddings.py |

---

### 2.9 Embedding Engines (engines/base.py, openai_engine.py, local_engine.py)

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| EmbeddingEngine interface has embed, dimensions, model_name | `test_embedding_engine_interface` | HIGH | unit | engines/base.py |
| OpenAI engine embed returns correct dimensions (mocked) | `test_openai_engine_embed_mocked` | HIGH | unit | engines/openai_engine.py |
| Local engine embed returns correct dimensions (mocked) | `test_local_engine_embed_mocked` | MEDIUM | unit | engines/local_engine.py |
| Engine selection by provider name | `test_engine_factory_selection` | HIGH | unit | engines/ |

---

### 2.10 Compat Layer (lib/compat.sh) -- ALL BASH TESTS

Zero bash tests exist. This entire section is new.

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| ragdag_sanitize strips special characters | `test_sanitize_strips_special` (bats) | CRITICAL | unit | bash/compat.sh |
| ragdag_sanitize lowercases | `test_sanitize_lowercases` (bats) | HIGH | unit | bash/compat.sh |
| ragdag_sanitize preserves alphanumeric, dot, dash, underscore | `test_sanitize_preserves_valid` (bats) | HIGH | unit | bash/compat.sh |
| ragdag_sha256 returns 64-char hex | `test_sha256_returns_hex` (bats) | CRITICAL | unit | bash/compat.sh |
| ragdag_sha256 same content same hash | `test_sha256_deterministic` (bats) | HIGH | unit | bash/compat.sh |
| ragdag_find_store walks up to find .ragdag | `test_find_store_walks_up` (bats) | CRITICAL | unit | bash/compat.sh |
| ragdag_find_store returns 1 when not found | `test_find_store_not_found` (bats) | HIGH | unit | bash/compat.sh |
| ragdag_estimate_tokens approximation | `test_estimate_tokens` (bats) | MEDIUM | unit | bash/compat.sh |
| ragdag_realpath resolves symlinks | `test_realpath_resolves` (bats) | MEDIUM | unit | bash/compat.sh |
| ragdag_sed_i works on current OS | `test_sed_i_portable` (bats) | HIGH | unit | bash/compat.sh |
| ragdag_file_size returns correct bytes | `test_file_size_correct` (bats) | MEDIUM | unit | bash/compat.sh |
| ragdag_has detects installed commands | `test_has_installed` (bats) | LOW | unit | bash/compat.sh |
| ragdag_has returns 1 for missing commands | `test_has_missing` (bats) | LOW | unit | bash/compat.sh |
| NO_COLOR disables color output | `test_no_color_env` (bats) | LOW | unit | bash/compat.sh |

---

### 2.11 Non-Functional Requirements

#### NFR-04: Security

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Filename sanitization blocks shell injection chars | `test_sanitize_blocks_injection` | CRITICAL | unit | SDK + bash |
| Sanitize: backticks stripped | `test_sanitize_backticks` | CRITICAL | unit | SDK + bash |
| Sanitize: dollar signs stripped | `test_sanitize_dollar_signs` | CRITICAL | unit | SDK + bash |
| Sanitize: semicolons stripped | `test_sanitize_semicolons` | CRITICAL | unit | SDK + bash |
| Sanitize: pipe characters stripped | `test_sanitize_pipes` | CRITICAL | unit | SDK + bash |
| No eval of user content in bash scripts | `test_no_eval_in_bash` | CRITICAL | static | bash |
| Prompt injection: context markers prevent escape | `test_prompt_injection_markers` | COVERED | unit | engines/llm.py |
| API keys from env vars only, not config file | `test_api_keys_env_only` | HIGH | unit | engines/ |

#### NFR-05: Reliability

| Test | Function Name | Priority | Type | Target |
|------|--------------|----------|------|--------|
| Atomic staging directory move | `test_store_atomic_move` | HIGH | integration | bash/store.sh |
| Re-ingestion atomic swap (no partial state) | `test_store_reingestion_atomic` | HIGH | integration | bash/store.sh |
| Embedding failure still stores chunks | `test_embed_failure_chunks_preserved` | CRITICAL | integration | SDK |

---

## 3. Gap Summary Matrix

### By Component

| Component | Existing Tests | Tests Needed | Gap |
|-----------|---------------|--------------|-----|
| SDK init | 2 | 9 | 7 |
| SDK add | 7 | 27+ | 20+ |
| SDK search (keyword) | 4 | 11 | 7 |
| SDK search (vector) | 0 | 10 | 10 |
| SDK search (hybrid) | 0 | 4 | 4 |
| SDK ask | 3 | 15 | 12 |
| SDK graph | 2 | 5 | 3 |
| SDK neighbors | 4 | 6 | 2 |
| SDK trace | 3 | 7 | 4 |
| SDK relate | 0 | 4 | 4 |
| SDK link | 2 | 2 | 0 |
| engines/llm.py | 8 | 11 | 3 |
| engines/embeddings.py | 0 | 15 | 15 |
| engines/similarity.py | 0 | 6 | 6 |
| engines/base+providers | 0 | 4 | 4 |
| server/api.py | 0 | 17 | 17 |
| server/mcp.py | 0 | 8 | 8 |
| bash/compat.sh | 0 | 14 | 14 |
| bash/config.sh | 0 | 8 | 8 |
| bash/init.sh | 0 | 2 | 2 |
| bash/parse.sh | 0 | 9 | 9 |
| bash/chunk.sh | 0 | 17 | 17 |
| bash/store.sh | 0 | 4 | 4 |
| bash/search.sh | 0 | 7 | 7 |
| bash/graph.sh | 0 | 5+ | 5+ |
| bash/ask.sh | 0 | 5+ | 5+ |
| bash/maintain.sh | 0 | 17 | 17 |
| Security (NFR-04) | 5 | 13 | 8 |
| Reliability (NFR-05) | 2 | 5 | 3 |
| **TOTAL** | **53** | **~260+** | **~210+** |

### By Priority

| Priority | Count | Description |
|----------|-------|-------------|
| CRITICAL | ~45 | Must test before any release. Functional correctness of core operations. |
| HIGH | ~95 | Should test. Important edge cases and integration points. |
| MEDIUM | ~50 | Nice to have. Completeness and robustness. |
| LOW | ~20 | Informational. Polish and UX. |

---

## 4. Test Infrastructure Gaps

### 4.1 No bats-core Test Framework

The architecture document specifies bats-core for bash testing, but:
- Zero .bats files exist
- No bats-core installation or configuration
- No test runner for bash tests

**Action needed**: Install bats-core, create test scaffolding, write tests for all 12 bash scripts.

### 4.2 No Test Fixtures

The architecture document specifies `tests/fixtures/` with sample documents, but:
- No fixtures directory exists
- No sample markdown, text, PDF, HTML, CSV, JSON, or code files for testing
- No expected output files for comparison

**Action needed**: Create `tests/fixtures/` with:
- `markdown/simple.md`, `markdown/with-headers.md`, `markdown/with-frontmatter.md`
- `text/plain.txt`
- `csv/sample.csv`
- `json/sample.json`
- `code/sample.py`, `code/sample.js`
- `expected/` directory with golden outputs

### 4.3 No Mock Embedding Infrastructure

Vector search and hybrid search testing requires mock embeddings:
- No mock embedding engine exists
- No pre-computed embeddings.bin fixtures
- No way to test vector search without real API calls

**Action needed**: Create a mock embedding engine that returns deterministic vectors for testing. Create pre-computed embeddings.bin fixtures.

### 4.4 No Server Test Client Setup

Neither the FastAPI TestClient nor MCP test infrastructure is set up:
- No httpx/TestClient fixture for API testing
- No MCP client test harness

**Action needed**: Add pytest fixtures for TestClient (FastAPI) and MCP tool invocation.

### 4.5 No conftest for Bash Tests

Need a test helper that:
- Sets up temp ragdag stores
- Sources all lib/*.sh files
- Provides cleanup

---

## 5. Recommended Test File Organization

```
tests/
  conftest.py                     # existing - add mock embedding fixture
  fixtures/
    markdown/
      simple.md
      with-headers.md
      with-frontmatter.md
    text/plain.txt
    csv/sample.csv
    json/sample.json
    code/sample.py
    expected/
  # Python unit tests
  test_chunk_fixed.py             # existing (10)
  test_chunk_strategies.py        # NEW: heading, paragraph, function
  test_llm.py                     # existing (8)
  test_tsv_matching.py            # existing (10)
  test_sdk_integration.py         # existing (25)
  test_parse.py                   # NEW: file parsing
  test_embeddings.py              # NEW: binary format read/write
  test_similarity.py              # NEW: cosine similarity
  test_embedding_engines.py       # NEW: engine factory, mock engines
  test_search_vector.py           # NEW: vector and hybrid search
  test_ask_advanced.py            # NEW: LLM integration, context assembly
  test_domain_rules.py            # NEW: domain rule matching
  test_config.py                  # NEW: INI config parsing (Python SDK)
  test_security.py                # NEW: sanitization, injection prevention
  test_maintenance.py             # NEW: verify, repair, gc, reindex (SDK)
  # Server tests
  test_api.py                     # NEW: FastAPI endpoint tests
  test_mcp.py                     # NEW: MCP server tool tests
  # Bash tests (bats-core)
  bash/
    test_helper.bash              # NEW: shared setup/teardown
    test_compat.bats              # NEW: compat.sh functions
    test_config.bats              # NEW: config.sh get/set
    test_init.bats                # NEW: init.sh
    test_parse.bats               # NEW: parse.sh type detection + extraction
    test_chunk.bats               # NEW: chunk.sh all strategies
    test_store.bats               # NEW: store.sh atomic moves, dedup
    test_search.bats              # NEW: search.sh keyword, JSON output
    test_graph.bats               # NEW: graph.sh summary, neighbors, trace
    test_ask.bats                 # NEW: ask.sh full flow
    test_add.bats                 # NEW: add.sh orchestration
    test_maintain.bats            # NEW: verify, repair, gc, reindex
```

---

## 6. Implementation Priority Order

### Phase 1: CRITICAL gaps (estimated ~45 tests)

1. **engines/embeddings.py tests** -- binary format is the foundation of vector search
2. **engines/similarity.py tests** -- cosine similarity correctness
3. **Chunking strategy tests** (heading, paragraph, function) -- core data pipeline
4. **File parsing tests** -- markdown frontmatter, CSV, JSON
5. **Security tests** -- filename sanitization, shell injection prevention
6. **Hybrid search fallback** -- tests that hybrid falls back to keyword
7. **API health endpoint** -- basic server smoke test
8. **MCP search/ask tools** -- primary integration surface for AI agents

### Phase 2: HIGH gaps (estimated ~95 tests)

1. **Server endpoint tests** (all FastAPI endpoints)
2. **MCP tool tests** (all tools)
3. **Vector search with mock embeddings**
4. **Ask with mocked LLM**
5. **Context assembly and token budget**
6. **Domain rules matching**
7. **Config INI parsing**
8. **Bash compat.sh unit tests**
9. **Bash store.sh atomic operations**
10. **Graph trace advanced (multi-hop, cycle detection)**

### Phase 3: MEDIUM gaps (estimated ~50 tests)

1. **Bash search.sh output formatting**
2. **Bash chunk.sh strategies**
3. **Maintenance operations** (verify, repair, gc)
4. **Reindex tests**
5. **Init edge cases** (gitignore, dependency check)
6. **Remaining embedding engine tests**

### Phase 4: LOW gaps (estimated ~20 tests)

1. **NO_COLOR env var handling**
2. **Command detection (ragdag_has)**
3. **Dependency check output format**
4. **Performance benchmarks** (informational, not blocking)

---

## 7. Test Dependencies and Prerequisites

| Test Category | Requires |
|---------------|----------|
| Bash tests | bats-core installed (`brew install bats-core`) |
| Server API tests | httpx, pytest-asyncio |
| MCP tests | mcp SDK test utilities |
| Vector search tests | numpy, mock embedding engine |
| Embedding format tests | numpy, struct |
| LLM provider tests | unittest.mock |
| PDF parsing tests | pdftotext (optional, skip if missing) |
| HTML parsing tests | pandoc or lynx (optional, skip if missing) |

---

## 8. Coverage Metric Targets

| Metric | Current | Target |
|--------|---------|--------|
| Total test count | 53 | ~260+ |
| Python SDK coverage | ~40% | >85% |
| Bash script coverage | 0% | >70% |
| Server endpoint coverage | 0% | >90% |
| Binary format coverage | 0% | >95% |
| Security test coverage | ~30% | >90% |
| FR requirement coverage | ~35% | >95% |
