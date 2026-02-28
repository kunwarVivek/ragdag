#!/usr/bin/env bats
# test_maintain.bats -- Tests for lib/maintain.sh

load test_helper

setup() {
  setup_store
  source "${RAGDAG_DIR}/lib/maintain.sh"
  # ragdag_find_store walks from cwd looking for .ragdag
  cd "$TEST_TMPDIR"
}

teardown() {
  teardown_store
}

# --- ragdag_verify ---

@test "ragdag_verify reports healthy for clean store" {
  create_test_chunks "docs" "readme" 3
  # Edges pointing to existing chunks
  add_test_edge "docs/readme/01.txt" "/src/readme.md" "chunked_from"

  run ragdag_verify
  [ "$status" -eq 0 ]
  [[ "$output" == *"healthy"* ]]
}

@test "ragdag_verify detects orphaned edges" {
  create_test_chunks "docs" "readme" 1
  # Edge whose source file does not exist
  add_test_edge "docs/readme/99.txt" "/src/readme.md" "chunked_from"

  run ragdag_verify
  [ "$status" -eq 0 ]
  [[ "$output" == *"orphaned"* ]]
  [[ "$output" == *"1 orphaned"* ]]
  [[ "$output" == *"issue"* ]]
}

@test "ragdag_verify detects stale .processed entries" {
  create_test_chunks "docs" "readme" 1
  # Add a .processed entry pointing to a non-existent source file
  printf '/nonexistent/source.md\tabc123\tdocs\t2025-01-01T00:00:00Z\n' >> "${TEST_STORE}/.processed"

  run ragdag_verify
  [ "$status" -eq 0 ]
  [[ "$output" == *"stale"* ]]
  [[ "$output" == *"1 stale"* ]]
}

@test "ragdag_verify counts chunks correctly" {
  create_test_chunks "docs" "readme" 5

  run ragdag_verify
  [ "$status" -eq 0 ]
  [[ "$output" == *"Chunks: 5 total"* ]]
}

# --- ragdag_repair ---

@test "ragdag_repair removes orphaned edges" {
  create_test_chunks "docs" "readme" 1
  # Valid edge (source exists)
  add_test_edge "docs/readme/01.txt" "/src/readme.md" "chunked_from"
  # Orphaned edge (source does not exist)
  add_test_edge "docs/readme/99.txt" "/src/gone.md" "chunked_from"

  run ragdag_repair
  [ "$status" -eq 0 ]
  [[ "$output" == *"Removed 1 orphaned edges"* ]]

  # Verify the orphaned edge is gone
  local edges_content
  edges_content="$(cat "${TEST_STORE}/.edges")"
  [[ "$edges_content" != *"docs/readme/99.txt"* ]]
}

@test "ragdag_repair preserves valid edges" {
  create_test_chunks "docs" "readme" 2
  add_test_edge "docs/readme/01.txt" "/src/readme.md" "chunked_from"
  add_test_edge "docs/readme/02.txt" "/src/readme.md" "chunked_from"

  run ragdag_repair
  [ "$status" -eq 0 ]

  # Both valid edges should still exist
  local edges_content
  edges_content="$(cat "${TEST_STORE}/.edges")"
  [[ "$edges_content" == *"docs/readme/01.txt"* ]]
  [[ "$edges_content" == *"docs/readme/02.txt"* ]]
}

# --- ragdag_gc ---

@test "ragdag_gc removes orphaned edges" {
  create_test_chunks "docs" "readme" 1
  add_test_edge "docs/readme/01.txt" "/src/readme.md" "chunked_from"
  add_test_edge "docs/readme/99.txt" "/src/gone.md" "chunked_from"

  run ragdag_gc
  [ "$status" -eq 0 ]

  # Orphaned edge should be removed
  local edges_content
  edges_content="$(cat "${TEST_STORE}/.edges")"
  [[ "$edges_content" != *"docs/readme/99.txt"* ]]
}

@test "ragdag_gc removes stale processed entries" {
  create_test_chunks "docs" "readme" 1
  # Valid processed entry (source file exists)
  local real_file="${TEST_TMPDIR}/real_source.md"
  echo "# Real doc" > "$real_file"
  printf '%s\tabc123\tdocs\t2025-01-01T00:00:00Z\n' "$real_file" >> "${TEST_STORE}/.processed"
  # Stale processed entry (source file does not exist)
  printf '/nonexistent/file.md\tdef456\tdocs\t2025-01-01T00:00:00Z\n' >> "${TEST_STORE}/.processed"

  run ragdag_gc
  [ "$status" -eq 0 ]

  # Stale entry should be removed
  local proc_content
  proc_content="$(cat "${TEST_STORE}/.processed")"
  [[ "$proc_content" != *"/nonexistent/file.md"* ]]
}

@test "ragdag_gc preserves valid entries" {
  create_test_chunks "docs" "readme" 2
  # Valid edge
  add_test_edge "docs/readme/01.txt" "/src/readme.md" "chunked_from"
  # Valid processed entry
  local real_file="${TEST_TMPDIR}/real_source.md"
  echo "# Real doc" > "$real_file"
  printf '%s\tabc123\tdocs\t2025-01-01T00:00:00Z\n' "$real_file" >> "${TEST_STORE}/.processed"

  run ragdag_gc
  [ "$status" -eq 0 ]

  # Valid edge should remain
  local edges_content
  edges_content="$(cat "${TEST_STORE}/.edges")"
  [[ "$edges_content" == *"docs/readme/01.txt"* ]]

  # Valid processed entry should remain
  local proc_content
  proc_content="$(cat "${TEST_STORE}/.processed")"
  [[ "$proc_content" == *"$real_file"* ]]
}

@test "ragdag_gc reports counts of cleaned items" {
  create_test_chunks "docs" "readme" 1
  # 2 orphaned edges
  add_test_edge "docs/readme/88.txt" "/src/a.md" "chunked_from"
  add_test_edge "docs/readme/99.txt" "/src/b.md" "chunked_from"
  # 1 stale processed entry
  printf '/gone/file.md\tabc\tdocs\t2025-01-01T00:00:00Z\n' >> "${TEST_STORE}/.processed"

  run ragdag_gc
  [ "$status" -eq 0 ]
  [[ "$output" == *"2 edges"* ]]
  [[ "$output" == *"1 processed"* ]]
}

# --- ragdag_reindex ---

@test "ragdag_reindex requires Python (fails gracefully without)" {
  # Override ragdag_has to pretend python3 doesn't exist
  ragdag_has() {
    [[ "$1" != "python3" ]] && command -v "$1" &>/dev/null
  }
  export -f ragdag_has

  run ragdag_reindex "docs"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Python"* ]]
}

@test "ragdag_reindex requires embedding provider" {
  # Config has provider = none by default
  create_test_chunks "docs" "readme" 1

  run ragdag_reindex "docs"
  [ "$status" -eq 1 ]
  [[ "$output" == *"provider"* ]] || [[ "$output" == *"Python"* ]]
}
