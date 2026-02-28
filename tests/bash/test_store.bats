#!/usr/bin/env bats

# Tests for lib/store.sh — storage, edges, processed tracking, domain rules

load test_helper

setup() {
  setup_store
  source "${RAGDAG_DIR}/lib/store.sh"
}

teardown() {
  teardown_store
}

# --- ragdag_store ---

@test "ragdag_store moves staging dir to target domain/docname" {
  # Create staging dir with chunk files
  local staging="${TEST_TMPDIR}/staging_$$"
  mkdir -p "$staging"
  echo "chunk 1 content" > "${staging}/01.txt"
  echo "chunk 2 content" > "${staging}/02.txt"

  run ragdag_store "$TEST_STORE" "$staging" "mydoc" "notes" "/tmp/source.md" "abc123"
  [ "$status" -eq 0 ]
  [ "$output" = "notes/mydoc" ]
  [ -f "${TEST_STORE}/notes/mydoc/01.txt" ]
  [ -f "${TEST_STORE}/notes/mydoc/02.txt" ]
}

@test "ragdag_store atomic move creates correct directory structure" {
  local staging="${TEST_TMPDIR}/staging_$$"
  mkdir -p "$staging"
  echo "content" > "${staging}/01.txt"

  ragdag_store "$TEST_STORE" "$staging" "testdoc" "domain1" "/src/file.txt" "hash1" >/dev/null

  # Verify the directory structure
  [ -d "${TEST_STORE}/domain1" ]
  [ -d "${TEST_STORE}/domain1/testdoc" ]
  [ -f "${TEST_STORE}/domain1/testdoc/01.txt" ]
  # Staging dir should be gone (it was moved)
  [ ! -d "$staging" ]
}

@test "ragdag_store records source in .processed file" {
  local staging="${TEST_TMPDIR}/staging_$$"
  mkdir -p "$staging"
  echo "content" > "${staging}/01.txt"

  ragdag_store "$TEST_STORE" "$staging" "doc1" "dom1" "/path/to/source.md" "sha256hash" >/dev/null

  # .processed should have an entry
  [ -f "${TEST_STORE}/.processed" ]
  run cat "${TEST_STORE}/.processed"
  [[ "$output" == *"/path/to/source.md"* ]]
  [[ "$output" == *"sha256hash"* ]]
  [[ "$output" == *"dom1"* ]]
}

@test "ragdag_store re-ingestion removes old chunks before storing new" {
  # First ingestion
  local staging1="${TEST_TMPDIR}/staging1_$$"
  mkdir -p "$staging1"
  echo "old chunk 1" > "${staging1}/01.txt"
  echo "old chunk 2" > "${staging1}/02.txt"
  echo "old chunk 3" > "${staging1}/03.txt"
  ragdag_store "$TEST_STORE" "$staging1" "doc" "dom" "/src.md" "hash1" >/dev/null

  [ -f "${TEST_STORE}/dom/doc/01.txt" ]
  [ -f "${TEST_STORE}/dom/doc/02.txt" ]
  [ -f "${TEST_STORE}/dom/doc/03.txt" ]

  # Re-ingestion with fewer chunks
  local staging2="${TEST_TMPDIR}/staging2_$$"
  mkdir -p "$staging2"
  echo "new chunk 1" > "${staging2}/01.txt"
  ragdag_store "$TEST_STORE" "$staging2" "doc" "dom" "/src.md" "hash2" >/dev/null

  # New chunk exists
  [ -f "${TEST_STORE}/dom/doc/01.txt" ]
  run cat "${TEST_STORE}/dom/doc/01.txt"
  [[ "$output" == *"new chunk 1"* ]]
  # Old extra chunks should be removed
  [ ! -f "${TEST_STORE}/dom/doc/02.txt" ]
  [ ! -f "${TEST_STORE}/dom/doc/03.txt" ]
}

# --- ragdag_store_edges ---

@test "ragdag_store_edges creates chunked_from edges for each .txt chunk" {
  # Setup chunks using helper
  create_test_chunks "edgedomain" "edgedoc" 3

  run ragdag_store_edges "$TEST_STORE" "edgedomain/edgedoc" "/original/file.md"
  [ "$status" -eq 0 ]

  # Verify edges file
  run cat "${TEST_STORE}/.edges"
  [[ "$output" == *"edgedomain/edgedoc/01.txt"* ]]
  [[ "$output" == *"edgedomain/edgedoc/02.txt"* ]]
  [[ "$output" == *"edgedomain/edgedoc/03.txt"* ]]
  [[ "$output" == *"chunked_from"* ]]
  [[ "$output" == *"/original/file.md"* ]]
}

@test "ragdag_store_edges uses tab-separated fields" {
  create_test_chunks "tabdom" "tabdoc" 1

  ragdag_store_edges "$TEST_STORE" "tabdom/tabdoc" "/src/file.md"

  # Each edge line should have exactly 4 tab-separated fields
  # Format: chunk_rel\tsource_path\tchunked_from\t<empty>
  local line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    local field_count
    field_count=$(echo "$line" | awk -F'\t' '{print NF}')
    [ "$field_count" -eq 4 ]
  done < "${TEST_STORE}/.edges"
}

# --- ragdag_is_processed ---

@test "ragdag_is_processed returns success for processed files" {
  # Write a processed entry manually
  printf '/path/to/file.md\tabc123\tmydomain\t2024-01-01T00:00:00Z\n' >> "${TEST_STORE}/.processed"

  run ragdag_is_processed "$TEST_STORE" "/path/to/file.md" "abc123"
  [ "$status" -eq 0 ]
}

@test "ragdag_is_processed returns failure for unprocessed files" {
  run ragdag_is_processed "$TEST_STORE" "/nonexistent/file.md" "somehash"
  [ "$status" -eq 1 ]
}

@test "ragdag_is_processed returns failure when hash differs" {
  printf '/path/to/file.md\tabc123\tmydomain\t2024-01-01T00:00:00Z\n' >> "${TEST_STORE}/.processed"

  run ragdag_is_processed "$TEST_STORE" "/path/to/file.md" "differenthash"
  [ "$status" -eq 1 ]
}

@test "ragdag_is_processed uses exact matching not substring" {
  printf '/path/to/file.md\thash1\tdom\t2024-01-01T00:00:00Z\n' >> "${TEST_STORE}/.processed"

  # Substring of the path should NOT match
  run ragdag_is_processed "$TEST_STORE" "/path/to/file" "hash1"
  [ "$status" -eq 1 ]

  # Substring of the hash should NOT match
  run ragdag_is_processed "$TEST_STORE" "/path/to/file.md" "hash"
  [ "$status" -eq 1 ]
}

# --- ragdag_apply_domain_rules ---

@test "ragdag_apply_domain_rules matches patterns from .domain-rules" {
  cat > "${TEST_STORE}/.domain-rules" <<'RULES'
# Domain rules
*.md notes → documentation
*.py src → codebase
RULES

  run ragdag_apply_domain_rules "$TEST_STORE" "/project/src/main.py"
  [ "$status" -eq 0 ]
  [ "$output" = "codebase" ]
}

@test "ragdag_apply_domain_rules returns empty for no match" {
  cat > "${TEST_STORE}/.domain-rules" <<'RULES'
*.md notes → documentation
RULES

  run ragdag_apply_domain_rules "$TEST_STORE" "/some/random/file.xyz"
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}

@test "ragdag_apply_domain_rules skips comment lines" {
  cat > "${TEST_STORE}/.domain-rules" <<'RULES'
# This is a comment
*.py → code
RULES

  run ragdag_apply_domain_rules "$TEST_STORE" "/app/main.py"
  [ "$status" -eq 0 ]
  [ "$output" = "code" ]
}

@test "ragdag_apply_domain_rules returns first matching rule" {
  cat > "${TEST_STORE}/.domain-rules" <<'RULES'
*.py → python-code
*.py → scripts
RULES

  run ragdag_apply_domain_rules "$TEST_STORE" "/app/main.py"
  [ "$status" -eq 0 ]
  [ "$output" = "python-code" ]
}

@test "ragdag_apply_domain_rules returns empty for empty rules file" {
  # .domain-rules is empty from setup_store
  run ragdag_apply_domain_rules "$TEST_STORE" "/any/file.md"
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}
