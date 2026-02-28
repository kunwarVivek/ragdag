# Validation Report: Bats Tests for graph.sh, search.sh, maintain.sh
Generated: 2026-02-15

## Overall Status: PASSED

## Test Summary
| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| test_graph.bats | 14 | 14 | 0 | 0 |
| test_search.bats | 11 | 11 | 0 | 0 |
| test_maintain.bats | 12 | 12 | 0 | 0 |
| **Total new tests** | **37** | **37** | **0** | **0** |

## Test Execution

### Command
```bash
bats tests/bash/test_graph.bats tests/bash/test_search.bats tests/bash/test_maintain.bats
```

### Output Summary
All 37 tests passed on first run. Full suite run (138 tests) shows 1 pre-existing failure in test_chunk.bats (bash 3.2 array subscript issue), unrelated to this work.

## Files Created

| File | Tests | Functions Covered |
|------|-------|-------------------|
| `tests/bash/test_graph.bats` | 14 | ragdag_graph, ragdag_neighbors, ragdag_trace, ragdag_link |
| `tests/bash/test_search.bats` | 11 | _search_keyword, ragdag_search, _format_results_human |
| `tests/bash/test_maintain.bats` | 12 | ragdag_verify, ragdag_repair, ragdag_gc, ragdag_reindex |

## Test Coverage by Function

### lib/graph.sh
| Function | Tests | Coverage |
|----------|-------|----------|
| `ragdag_graph` | 3 | Full summary counts, domain filter behavior, empty store zeros |
| `ragdag_neighbors` | 4 | Outgoing edges, incoming edges, unconnected node, missing argument error |
| `ragdag_trace` | 4 | chunked_from traversal, origin termination, cycle detection, max depth 20 |
| `ragdag_link` | 3 | Edge creation, default type "references", missing args validation |

### lib/search.sh
| Function | Tests | Coverage |
|----------|-------|----------|
| `_search_keyword` | 7 | Matching, TF-IDF ranking, top-K, domain filter, case insensitivity, no matches, short word filtering |
| `ragdag_search` | 3 | --keyword flag, --json output, missing query error |
| `_format_results_human` | 1 | Path, score, content preview display |

### lib/maintain.sh
| Function | Tests | Coverage |
|----------|-------|----------|
| `ragdag_verify` | 4 | Healthy store, orphaned edges, stale processed, chunk counting |
| `ragdag_repair` | 2 | Orphan removal, valid edge preservation |
| `ragdag_gc` | 4 | Orphan edge removal, stale processed removal, valid entry preservation, count reporting |
| `ragdag_reindex` | 2 | Python requirement, embedding provider requirement |

## Implementation Notes

1. **Store discovery**: All functions use `ragdag_find_store()` which walks from cwd upward looking for `.ragdag/`. Tests `cd` into `$TEST_TMPDIR` in setup.

2. **Domain filter in ragdag_graph**: The `$1` argument sets `search_path` but the counting loop always iterates from `$store_dir/*/`, so the filter is currently a no-op. Test documents this actual behavior.

3. **grep -cv on empty files**: Returns count 0 but exit code 1. The implementation handles this with `|| echo 0`.

4. **ragdag_reindex Python test**: Uses function override of `ragdag_has` to simulate missing python3.

## Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 14 graph tests per spec | PASS | All 14 tests in test_graph.bats |
| 11 search tests per spec | PASS | All 11 tests in test_search.bats |
| 12 maintain tests per spec | PASS | All 12 tests in test_maintain.bats |
| Tests exercise existing functionality | PASS | All assertions verified against current implementation |
| Tests pass against current codebase | PASS | 37/37 green on first run |
| Uses shared test_helper fixtures | PASS | setup_store, teardown_store, create_test_chunks, add_test_edge all used |
