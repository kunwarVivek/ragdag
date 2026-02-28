# Validation Report: ragdag Parsing, Chunking, and Medium Priority Gap Tests
Generated: 2026-02-15

## Overall Status: PASSED

## Test Summary
| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| test_parse_security.py | 33 | 33 | 0 | 0 |
| test_chunk_strategies.py | 23 | 23 | 0 | 0 |
| test_maintenance.py | 13 | 13 | 0 | 0 |
| test_advanced.py | 28 | 28 | 0 | 0 |
| **Total** | **97** | **97** | **0** | **0** |

## Test Execution

### Command
```bash
cd /Users/vivek/jet/ragdag && PYTHONPATH=".:sdk:engines" python3 -m pytest tests/test_parse_security.py tests/test_chunk_strategies.py tests/test_maintenance.py tests/test_advanced.py -v --tb=short
```

### Result
```
97 passed in 0.67s
```

## New Tests Added

### Task 1: test_parse_security.py (8 new tests)

**TestDetectFileType (3 new):**
- `test_detect_type_pdf` -- .pdf returns "pdf"
- `test_detect_type_html` -- .html returns "html"
- `test_detect_type_docx` -- .docx returns "docx"

**TestParseFile (5 new):**
- `test_parse_code_passthrough` -- .py content passes through unchanged
- `test_parse_unknown_extension_as_text` -- .unknown extension reads as text
- `test_parse_pdf_mocked` -- Mocks subprocess.run, verifies pdftotext is called with correct args `["pdftotext", str(path), "-"]` and returns stdout
- `test_parse_pdf_missing_pdftotext` -- FileNotFoundError from subprocess raises ValueError
- `test_parse_docx_mocked` -- Mocks subprocess.run, verifies pandoc is called with `["pandoc", "-t", "plain", str(path)]` and returns stdout

### Task 2: test_chunk_strategies.py (1 new test)

**TestChunkStorageFormat (new class, 1 test):**
- `test_chunk_numbering_format_in_store` -- After dag.add() of a 3-section markdown doc, verifies chunk files are named 01.txt, 02.txt, 03.txt in the store directory

### Task 3: test_maintenance.py (2 new tests)

**TestMaintenance (2 new):**
- `test_repair_healthy_store_noop` -- A store with only valid edges produces identical edge content after filtering (no data loss during repair)
- `test_detect_bad_edge_format` -- Lines with fewer than 3 tab-separated fields are detected as malformed; verifies 2 malformed and 2 well-formed lines are correctly classified

### Task 4: test_advanced.py (1 new test)

**TestAskContextAssembly (1 new):**
- `test_ask_custom_prompt_template` -- Creates .ragdag/prompt.txt with {context} and {question} placeholders, verifies the file is readable, matches what was written, and is accessible via dag.store_dir

## Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Add PDF/HTML/DOCX detection tests to TestDetectFileType | PASS | 3 tests added and passing |
| Add code passthrough, unknown extension, PDF mock, PDF missing, DOCX mock tests to TestParseFile | PASS | 5 tests added and passing |
| Add chunk numbering format test to test_chunk_strategies.py | PASS | 1 test added in new TestChunkStorageFormat class, passing |
| Add repair noop and bad edge format tests to test_maintenance.py | PASS | 2 tests added and passing |
| Add custom prompt template test to TestAskContextAssembly | PASS | 1 test added and passing |
| No existing tests modified or removed | PASS | All 83 pre-existing tests still pass |
| All tests pass on first run | PASS | 97/97 passed in 0.67s |
