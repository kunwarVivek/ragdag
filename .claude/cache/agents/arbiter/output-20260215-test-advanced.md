# Validation Report: ragdag SDK Advanced Features (test_advanced.py)
Generated: 2026-02-15

## Overall Status: PASSED

## Test Summary
| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| ASK - Context Assembly | 6 | 6 | 0 | 0 |
| Domain Rules | 6 | 6 | 0 | 0 |
| Config | 5 | 5 | 0 | 0 |
| Graph Advanced | 4 | 4 | 0 | 0 |
| **Total** | **21** | **21** | **0** | **0** |

## Test Execution

### Command
```bash
cd /Users/vivek/jet/ragdag && PYTHONPATH=".:sdk" python3 -m pytest tests/test_advanced.py -v --tb=short
```

### Output
```
21 passed in 0.39s
```

## Acceptance Criteria

| # | Criterion | Status | Test Name |
|---|-----------|--------|-----------|
| 1 | ask context respects token budget | PASS | test_ask_context_respects_token_budget |
| 2 | ask context format has source path | PASS | test_ask_context_format_has_source_path |
| 3 | ask graph expansion related_to | PASS | test_ask_graph_expansion_related_to |
| 4 | ask graph expansion references | PASS | test_ask_graph_expansion_references |
| 5 | ask graph expansion does not duplicate | PASS | test_ask_graph_expansion_does_not_duplicate |
| 6 | ask with mocked LLM | PASS | test_ask_with_mocked_llm |
| 7 | domain rules basic match | PASS | test_domain_rules_basic_match |
| 8 | domain rules multiple patterns | PASS | test_domain_rules_multiple_patterns |
| 9 | domain rules comments ignored | PASS | test_domain_rules_comments_ignored |
| 10 | domain rules no match returns empty | PASS | test_domain_rules_no_match_returns_empty |
| 11 | domain rules case insensitive | PASS | test_domain_rules_case_insensitive |
| 12 | add domain auto | PASS | test_add_domain_auto |
| 13 | read config existing key | PASS | test_read_config_existing_key |
| 14 | read config missing key returns default | PASS | test_read_config_missing_key_returns_default |
| 15 | read config wrong section | PASS | test_read_config_wrong_section |
| 16 | read config comments ignored | PASS | test_read_config_comments_ignored |
| 17 | read config whitespace handling | PASS | test_read_config_whitespace_handling |
| 18 | trace multi hop | PASS | test_trace_multi_hop |
| 19 | trace cycle detection | PASS | test_trace_cycle_detection |
| 20 | graph empty store | PASS | test_graph_empty_store |
| 21 | graph edge type counts | PASS | test_graph_edge_type_counts |

## Notes

### Fixes Applied During Development

Two tests required adjustment after the initial run:

1. **test_ask_context_respects_token_budget**: Initial max_context=50 was too small for even a single chunk (each chunk had ~160 words = ~208 estimated tokens). Fixed by using max_context=200 with 15 smaller chunks (~20 words each = ~26 tokens), ensuring some but not all chunks fit.

2. **test_ask_with_mocked_llm**: Initial approach used `patch.dict("sys.modules", {"engines": MagicMock()})` which broke the keyword search path (search internally uses `self._store.rglob` not engines, but the MagicMock replacement of the engines package caused import side effects). Fixed by manually inserting a fake `engines.llm` module into `sys.modules` with proper cleanup, without replacing the `engines` package itself.

### Test Design Approach

- Each test uses `tmp_path` for full isolation
- Helper functions (`_make_dag`, `_write_config`, `_write_edges`, `_add_chunk`, `_write_domain_rules`) keep tests concise
- The LLM mock test uses `types.ModuleType` for a clean fake module with proper `sys.modules` restoration in a `finally` block
- Graph tests create store structures manually (via `_add_chunk` and `_write_edges`) rather than going through `dag.add()`, testing the graph/trace logic in isolation
