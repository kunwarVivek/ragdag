# Validation Report: MCP Server Tests (server/mcp.py)
Generated: 2026-02-15

## Overall Status: PASSED

## Test Summary
| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| Unit     | 21    | 21     | 0      | 0       |

## Test Execution

### Command
```bash
cd /Users/vivek/jet/ragdag && PYTHONPATH=".:sdk:engines" python3 -m pytest tests/test_mcp.py -v --tb=short
```

### Output Summary
```
21 passed in 1.53s
```

## Test Breakdown

### 1. TestMcpSearch (CRITICAL) - 4 tests
- `test_mcp_search_keyword` - keyword search returns numbered, scored results
- `test_mcp_search_default_mode` - hybrid mode falls back to keyword (provider=none)
- `test_mcp_search_with_domain_filter` - domain filter restricts results
- `test_mcp_search_top_param` - top=1 limits to single result

### 2. TestMcpAsk (CRITICAL) - 3 tests
- `test_mcp_ask_returns_context_and_sources` - returns context with **Sources:** section
- `test_mcp_ask_with_domain` - domain-scoped results
- `test_mcp_ask_no_results` - nonsense query returns no Sources section

### 3. TestMcpAdd - 2 tests
- `test_mcp_add_ingests_file` - returns JSON with files/chunks/skipped counts
- `test_mcp_add_with_domain` - creates chunks under domain directory

### 4. TestMcpGraph - 3 tests
- `test_mcp_graph_returns_stats` - returns Domains/Documents/Chunks/Edges/Edge types
- `test_mcp_graph_with_domain_filter` - single domain count
- `test_mcp_graph_nonexistent_domain` - zeroed stats

### 5. TestMcpNeighbors - 3 tests
- `test_mcp_neighbors_returns_edges` - outgoing edges with arrow formatting
- `test_mcp_neighbors_incoming` - incoming edges
- `test_mcp_neighbors_no_edges` - "No neighbors found" message

### 6. TestMcpTrace - 2 tests
- `test_mcp_trace_returns_provenance` - chunked_from provenance chain
- `test_mcp_trace_no_provenance` - origin-only for unknown nodes

### 7. TestMcpErrorHandling - 2 tests
- `test_mcp_search_no_results` - returns "No results found."
- `test_mcp_search_empty_store` - empty store returns "No results found."

### 8. TestMcpStoreContext - 2 tests
- `test_ragdag_store_env_controls_store` - RAGDAG_STORE env var controls _get_dag()
- `test_ragdag_store_default_uses_cwd` - defaults to cwd without env var

## Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| test_mcp_search - keyword mode returns formatted results | PASS | TestMcpSearch::test_mcp_search_keyword |
| test_mcp_ask - returns answer/context with sources | PASS | TestMcpAsk::test_mcp_ask_returns_context_and_sources |
| test_mcp_add - ingests file, returns JSON summary | PASS | TestMcpAdd::test_mcp_add_ingests_file |
| test_mcp_graph - returns stats with Domains/Chunks/Edges | PASS | TestMcpGraph::test_mcp_graph_returns_stats |
| test_mcp_neighbors - returns formatted edge list | PASS | TestMcpNeighbors::test_mcp_neighbors_returns_edges |
| test_mcp_trace - returns provenance chain | PASS | TestMcpTrace::test_mcp_trace_returns_provenance |
| test_mcp_error_handling - no results returns message | PASS | TestMcpErrorHandling::test_mcp_search_no_results |
| test_mcp_store_context - env var controls store | PASS | TestMcpStoreContext::test_ragdag_store_env_controls_store |

## Technical Notes

### FunctionTool Wrapper
The `@mcp.tool` decorator (FastMCP) wraps async functions into `FunctionTool` objects,
which are not directly callable. The tests extract the original async function via
the `.fn` attribute:
```python
_search_fn = mcp_module.ragdag_search.fn  # original async function
```

### Async Execution
Tests use a synchronous `_run()` helper that creates a new event loop per call:
```python
def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

### Mock Strategy
All tool tests mock `server.mcp._get_dag` to return a real RagDag instance
backed by a tmp_path store with test data. No external services needed.

## Failure Analysis

No failures.

## Recommendations

### Should Fix (Non-blocking)
None - all tests pass cleanly.

### Missing Coverage
1. Error path when `_get_dag()` fails (e.g., missing store directory)
2. `ragdag_add` with a nonexistent path (FileNotFoundError propagation)
3. MCP server `run()` and `run_http()` startup (would require transport mocking)
4. Context parameter (`ctx`) logging paths (would require mocking FastMCP Context)
