# Testing Guide

This repository uses two test layers:

- `pytest` for Python SDK/engines/server tests
- `bats` for Bash CLI library tests

## Test Locations

- Python tests: `tests/*.py`
- Bash tests: `tests/bash/*.bats`
- Bash test helpers: `tests/bash/test_helper.bash`

## Running Tests

### Python

```bash
PYTHONPATH=".:sdk:engines" python3 -m pytest tests -q
```

### Bash (bats)

```bash
bats tests/bash
```

### Targeted Runs

```bash
PYTHONPATH=".:sdk:engines" python3 -m pytest tests/test_api.py -q
bats tests/bash/test_graph.bats
```

## What To Add When Changing Code

- Changes in `sdk/ragdag/core.py` should include or update pytest coverage in `tests/test_sdk_integration.py` and relevant focused suites.
- Changes in `server/api.py` should include endpoint tests in `tests/test_api.py`.
- Changes in `server/mcp.py` should include tool behavior tests in `tests/test_mcp.py`.
- Changes in `lib/*.sh` should include bats tests under `tests/bash/`.

## Quality Gate Expectations

Before merging changes:

- Run related pytest suites.
- Run related bats suites for Bash changes.
- Ensure no generated one-off report docs are committed under `.claude/cache/agents/` or `thoughts/shared/plans/`.
