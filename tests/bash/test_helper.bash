#!/usr/bin/env bash
# test_helper.bash — Shared setup/teardown for ragdag bats tests

# Project root
RAGDAG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RAGDAG_DIR

# Source the modules we need
source "${RAGDAG_DIR}/lib/compat.sh"
source "${RAGDAG_DIR}/lib/config.sh"

# Create a temporary ragdag store for testing
setup_store() {
  TEST_TMPDIR="$(mktemp -d)"
  TEST_STORE="${TEST_TMPDIR}/.ragdag"
  mkdir -p "$TEST_STORE"

  # Create default config
  cat > "${TEST_STORE}/.config" <<'CONF'
[general]
chunk_strategy = heading
chunk_size = 1000
chunk_overlap = 100

[embedding]
provider = none
model = text-embedding-3-small
dimensions = 1536

[llm]
provider = none
model = gpt-4o-mini
max_context = 8000

[search]
default_mode = hybrid
top_k = 10
keyword_weight = 0.3
vector_weight = 0.7

[edges]
auto_relate = false
relate_threshold = 0.8
record_queries = false
CONF

  # Create empty metadata files
  touch "${TEST_STORE}/.edges"
  touch "${TEST_STORE}/.processed"
  touch "${TEST_STORE}/.domain-rules"

  export RAGDAG_STORE="$TEST_TMPDIR"
}

# Clean up after tests
teardown_store() {
  if [[ -n "${TEST_TMPDIR:-}" ]] && [[ -d "${TEST_TMPDIR:-}" ]]; then
    rm -rf "$TEST_TMPDIR"
  fi
}

# Create a test markdown file
create_test_markdown() {
  local path="${1:-${TEST_TMPDIR}/test.md}"
  mkdir -p "$(dirname "$path")"
  cat > "$path" <<'MD'
# Introduction

This is the introduction section with some content about ragdag.

# Features

ragdag supports keyword search, vector search, and hybrid search.
It uses flat files and bash as its primary interface.

# Installation

Clone the repo and add to PATH. No build step required.
MD
  echo "$path"
}

# Create a test text file
create_test_text() {
  local path="${1:-${TEST_TMPDIR}/test.txt}"
  local content="${2:-This is test content for ragdag testing.}"
  mkdir -p "$(dirname "$path")"
  echo "$content" > "$path"
  echo "$path"
}

# Create test chunks in a domain
create_test_chunks() {
  local domain="${1:-testdomain}"
  local doc="${2:-testdoc}"
  local num_chunks="${3:-3}"

  local chunk_dir="${TEST_STORE}/${domain}/${doc}"
  mkdir -p "$chunk_dir"

  for i in $(seq 1 "$num_chunks"); do
    local fname
    fname="$(printf '%02d.txt' "$i")"
    echo "This is chunk $i content for testing. It has some words about ragdag and searching." > "${chunk_dir}/${fname}"
  done

  echo "${domain}/${doc}"
}

# Add edges to the test store
add_test_edge() {
  local source="$1"
  local target="$2"
  local etype="${3:-chunked_from}"
  local metadata="${4:-}"
  printf '%s\t%s\t%s\t%s\n' "$source" "$target" "$etype" "$metadata" >> "${TEST_STORE}/.edges"
}
