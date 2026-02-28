#!/usr/bin/env bats
# test_search.bats -- Tests for lib/search.sh

load test_helper

setup() {
  setup_store
  source "${RAGDAG_DIR}/lib/search.sh"
  # ragdag_find_store walks from cwd looking for .ragdag
  cd "$TEST_TMPDIR"
}

teardown() {
  teardown_store
}

# Helper to create a chunk with specific content
create_chunk_with_content() {
  local domain="$1"
  local doc="$2"
  local filename="$3"
  local content="$4"
  local chunk_dir="${TEST_STORE}/${domain}/${doc}"
  mkdir -p "$chunk_dir"
  echo "$content" > "${chunk_dir}/${filename}"
}

# --- _search_keyword ---

@test "_search_keyword finds matching chunks by grep" {
  create_chunk_with_content "docs" "readme" "01.txt" \
    "ragdag is a knowledge graph engine using flat files and bash"
  create_chunk_with_content "docs" "readme" "02.txt" \
    "this chunk has nothing relevant to the query"

  run _search_keyword "$TEST_STORE" "knowledge graph" "" "10" "0"
  [ "$status" -eq 0 ]
  [[ "$output" == *"docs/readme/01.txt"* ]]
}

@test "_search_keyword ranks by TF-IDF-like score (match_count * 10000 / len)" {
  # Chunk with more keyword density should rank higher
  create_chunk_with_content "docs" "a" "01.txt" \
    "search search search search"
  create_chunk_with_content "docs" "b" "01.txt" \
    "search is a feature. There are many features in this application that do various things besides search."

  run _search_keyword "$TEST_STORE" "search" "" "10" "0"
  [ "$status" -eq 0 ]
  # The denser chunk (docs/a/01.txt) should appear first
  local first_result
  first_result="$(echo "$output" | head -1)"
  [[ "$first_result" == *"docs/a/01.txt"* ]]
}

@test "_search_keyword respects top-K limit" {
  # Create 5 chunks that all match
  for i in 1 2 3 4 5; do
    create_chunk_with_content "docs" "doc${i}" "01.txt" \
      "ragdag search test content number ${i}"
  done

  run _search_keyword "$TEST_STORE" "ragdag" "" "2" "0"
  [ "$status" -eq 0 ]
  # Count result lines that contain a path (numbered results like "1. [score] path")
  local result_count
  result_count="$(echo "$output" | grep -c '^\d\+\.' || true)"
  [ "$result_count" -le 2 ]
}

@test "_search_keyword domain filter restricts search scope" {
  create_chunk_with_content "alpha" "doc1" "01.txt" "unique keyword here"
  create_chunk_with_content "beta" "doc2" "01.txt" "unique keyword here"

  run _search_keyword "$TEST_STORE" "unique keyword" "alpha" "10" "0"
  [ "$status" -eq 0 ]
  [[ "$output" == *"alpha/doc1/01.txt"* ]]
  [[ "$output" != *"beta/doc2/01.txt"* ]]
}

@test "_search_keyword case insensitive matching" {
  create_chunk_with_content "docs" "doc" "01.txt" \
    "RagDag Is A Knowledge Graph Engine"

  run _search_keyword "$TEST_STORE" "ragdag" "" "10" "0"
  [ "$status" -eq 0 ]
  [[ "$output" == *"docs/doc/01.txt"* ]]
}

@test "_search_keyword returns empty for no matches" {
  create_chunk_with_content "docs" "doc" "01.txt" \
    "some content about dogs and cats"

  run _search_keyword "$TEST_STORE" "xyzzyzombieword" "" "10" "0"
  [ "$status" -eq 0 ]
  [[ "$output" == *"No results found"* ]]
}

@test "short words (< 2 chars) are ignored in search" {
  create_chunk_with_content "docs" "doc" "01.txt" \
    "a b c d e f g h i j k l m n o p q r s t u v w x y z"

  # Query of only single-character words should find nothing
  run _search_keyword "$TEST_STORE" "a b c" "" "10" "0"
  [ "$status" -eq 0 ]
  [[ "$output" == *"No results found"* ]]
}

# --- ragdag_search ---

@test "ragdag_search --keyword parses --keyword flag" {
  create_chunk_with_content "docs" "doc" "01.txt" \
    "ragdag supports keyword search and vector search"

  run ragdag_search "keyword search" --keyword
  [ "$status" -eq 0 ]
  [[ "$output" == *"docs/doc/01.txt"* ]]
}

@test "ragdag_search --json outputs JSON format" {
  create_chunk_with_content "docs" "doc" "01.txt" \
    "ragdag supports keyword search"

  run ragdag_search "ragdag" --keyword --json
  [ "$status" -eq 0 ]
  # JSON output starts with [ and ends with ]
  [[ "$output" == "["* ]]
  [[ "$output" == *"]" ]]
  [[ "$output" == *"\"path\""* ]]
}

@test "ragdag_search requires query argument" {
  run ragdag_search
  [ "$status" -eq 1 ]
  [[ "$output" == *"Usage"* ]]
}

# --- _format_results_human ---

@test "_format_results_human shows path, score, and content preview" {
  create_chunk_with_content "docs" "readme" "01.txt" \
    "ragdag is a knowledge graph engine for testing"

  # Build a result line in the format the function expects: score\tpath
  local results
  results="$(printf '150\tdocs/readme/01.txt')"

  run _format_results_human "$TEST_STORE" "$results"
  [ "$status" -eq 0 ]
  # Should show rank number, score, and path
  [[ "$output" == *"1."* ]]
  [[ "$output" == *"docs/readme/01.txt"* ]]
  # Should show content preview
  [[ "$output" == *"ragdag"* ]]
}
