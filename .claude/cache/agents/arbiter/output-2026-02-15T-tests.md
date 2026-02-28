# Validation Report: ragdag test additions (relate, init edge cases, add edge cases)
Generated: 2026-02-15

## Overall Status: PASSED

## Test Summary
| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| test_advanced.py | 26 | 26 | 0 | 0 |
| test_sdk_integration.py | 42 | 42 | 0 | 0 |
| **Total** | **68** | **68** | **0** | **0** |

## Test Execution

### Command
```bash
cd /Users/vivek/jet/ragdag && PYTHONPATH=".:sdk:engines" python3 -m pytest tests/test_advanced.py tests/test_sdk_integration.py -v --tb=short
```

### Output Summary
```
68 passed in 0.65s
```

## New Tests Added

### test_advanced.py - TestRelate (2 new tests)

1. **`test_relate_needs_embeddings_bin_to_create_edges`** - Creates chunks in a domain directory without embeddings.bin. Verifies relate() adds no related_to edges. Also confirms no embeddings.bin was created.

2. **`test_relate_checks_existing_edges`** - Pre-populates .edges with existing related_to and chunked_from edges. Calls relate() with no embeddings. Verifies the code safely reads existing edges (the dedup set) without crashing and preserves all existing edges.

### test_sdk_integration.py - TestInit (1 new test)

3. **`test_init_gitignore_in_git_repo`** - Creates a .git/ directory to simulate a git repo, then calls init(). Confirms init succeeds without error. Documents that _init_store() does not currently implement .gitignore handling.

### test_sdk_integration.py - TestAdd (4 new tests)

4. **`test_add_flat_no_domain_subdir`** - Adds a file without domain parameter. Verifies chunks are stored at .ragdag/<doc_name>/ (not under a domain subdirectory). Confirms doc dir parent is the store root.

5. **`test_add_domain_auto_unsorted`** - Adds a file with domain="auto" when no domain rules match. Verifies chunks land under .ragdag/unsorted/ per the fallback logic at core.py:196.

6. **`test_add_binary_file_handled`** - Creates a fake .png binary file and adds it. Verifies add() does not crash (falls back to read_text with errors='replace').

7. **`test_processed_records_source_hash`** - Adds a file and reads .processed. Verifies the absolute source path appears in the file and the content hash is a valid 64-character hex SHA256.

### test_sdk_integration.py - TestSearch (2 new tests)

8. **`test_keyword_search_multiword`** - Searches "JWT tokens" against a doc containing both words. Confirms multiword queries match via word-level summing.

9. **`test_keyword_short_words_ignored`** - Searches "a b authentication" and confirms single-char words are filtered out (len >= 2 filter). Also verifies a query of only short words ("a b c") returns zero results.

### test_sdk_integration.py - TestNeighbors (1 new test)

10. **`test_neighbors_includes_metadata`** - Uses link() to create an edge, then verifies neighbors() returns a dict with a 'metadata' key. Also manually writes an edge with metadata "weight=0.9" and confirms the value is parsed correctly.

## Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| test_relate_needs_embeddings_bin_to_create_edges added to TestRelate | PASS | Lines 606-638 of test_advanced.py |
| test_relate_checks_existing_edges added to TestRelate | PASS | Lines 640-667 of test_advanced.py |
| test_init_gitignore_in_git_repo added to TestInit | PASS | Lines 69-86 of test_sdk_integration.py |
| test_add_flat_no_domain_subdir added to TestAdd | PASS | Lines 284-310 of test_sdk_integration.py |
| test_add_domain_auto_unsorted added to TestAdd | PASS | Lines 312-332 of test_sdk_integration.py |
| test_add_binary_file_handled added to TestAdd | PASS | Lines 334-351 of test_sdk_integration.py |
| test_processed_records_source_hash added to TestAdd | PASS | Lines 353-386 of test_sdk_integration.py |
| test_keyword_search_multiword added to TestSearch | PASS | Lines 509-525 of test_sdk_integration.py |
| test_keyword_short_words_ignored added to TestSearch | PASS | Lines 527-549 of test_sdk_integration.py |
| test_neighbors_includes_metadata added to TestNeighbors | PASS | Lines 624-663 of test_sdk_integration.py |
| All existing tests still pass | PASS | 68/68 passed, 0 failures |
| No existing tests modified or removed | PASS | All original tests preserved verbatim |
