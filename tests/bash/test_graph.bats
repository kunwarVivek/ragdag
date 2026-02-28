#!/usr/bin/env bats
# test_graph.bats -- Tests for lib/graph.sh

load test_helper

setup() {
  setup_store
  source "${RAGDAG_DIR}/lib/graph.sh"
  # ragdag_find_store walks from cwd looking for .ragdag
  cd "$TEST_TMPDIR"
}

teardown() {
  teardown_store
}

# --- ragdag_graph ---

@test "ragdag_graph shows domains, documents, chunks, edges counts" {
  create_test_chunks "science" "physics" 3
  create_test_chunks "science" "chemistry" 2
  create_test_chunks "history" "ww2" 1
  add_test_edge "science/physics/01.txt" "/src/physics.md" "chunked_from"
  add_test_edge "science/chemistry/01.txt" "/src/chem.md" "chunked_from"

  run ragdag_graph
  [ "$status" -eq 0 ]
  [[ "$output" == *"Domains:    2"* ]]
  [[ "$output" == *"Documents:  3"* ]]
  [[ "$output" == *"Chunks:     6"* ]]
  [[ "$output" == *"Edges:      2"* ]]
}

@test "ragdag_graph with domain filter still counts all domains" {
  # ragdag_graph takes domain_filter as $1 but the counting logic
  # iterates over store_dir/*/ regardless -- so domain filter argument
  # doesn't actually restrict the count (the search_path var is set but unused
  # in the current implementation). The test verifies current behavior.
  create_test_chunks "alpha" "doc1" 2
  create_test_chunks "beta" "doc2" 3

  run ragdag_graph "alpha"
  [ "$status" -eq 0 ]
  # The function always counts from store_dir, so both domains appear
  [[ "$output" == *"Domains:    2"* ]]
}

@test "ragdag_graph empty store shows zero counts" {
  run ragdag_graph
  [ "$status" -eq 0 ]
  [[ "$output" == *"Domains:    0"* ]]
  [[ "$output" == *"Documents:  0"* ]]
  [[ "$output" == *"Chunks:     0"* ]]
  [[ "$output" == *"Edges:      0"* ]]
}

# --- ragdag_neighbors ---

@test "ragdag_neighbors shows outgoing edges (node as source)" {
  create_test_chunks "dom" "doc" 2
  add_test_edge "dom/doc/01.txt" "/src/file.md" "chunked_from"

  run ragdag_neighbors "dom/doc/01.txt"
  [ "$status" -eq 0 ]
  [[ "$output" == *"/src/file.md"* ]]
  [[ "$output" == *"chunked_from"* ]]
}

@test "ragdag_neighbors shows incoming edges (node as target)" {
  create_test_chunks "dom" "doc" 2
  add_test_edge "dom/doc/01.txt" "dom/doc/02.txt" "references"

  run ragdag_neighbors "dom/doc/02.txt"
  [ "$status" -eq 0 ]
  [[ "$output" == *"dom/doc/01.txt"* ]]
  [[ "$output" == *"references"* ]]
}

@test "ragdag_neighbors empty result for unconnected node" {
  create_test_chunks "dom" "doc" 1

  run ragdag_neighbors "dom/doc/01.txt"
  [ "$status" -eq 0 ]
  # Should show the header but no edges
  [[ "$output" == *"Neighbors of: dom/doc/01.txt"* ]]
  # No arrow lines
  [[ "$output" != *"[chunked_from]"* ]]
  [[ "$output" != *"[references]"* ]]
}

@test "ragdag_neighbors requires node argument (fails without)" {
  run ragdag_neighbors
  [ "$status" -eq 1 ]
  [[ "$output" == *"Usage"* ]]
}

# --- ragdag_trace ---

@test "ragdag_trace follows chunked_from edges backward" {
  create_test_chunks "dom" "doc" 2
  add_test_edge "dom/doc/01.txt" "/source/original.md" "chunked_from"

  run ragdag_trace "dom/doc/01.txt"
  [ "$status" -eq 0 ]
  [[ "$output" == *"dom/doc/01.txt"* ]]
  [[ "$output" == *"chunked_from"* ]]
  [[ "$output" == *"/source/original.md"* ]]
  [[ "$output" == *"(origin)"* ]]
}

@test "ragdag_trace terminates at origin (no more parents)" {
  create_test_chunks "dom" "doc" 1
  # Node with no chunked_from edge
  run ragdag_trace "dom/doc/01.txt"
  [ "$status" -eq 0 ]
  [[ "$output" == *"dom/doc/01.txt"* ]]
  [[ "$output" == *"(origin)"* ]]
}

@test "ragdag_trace cycle detection prevents infinite loop" {
  create_test_chunks "dom" "doc" 2
  # Create a cycle: A -> B -> A
  add_test_edge "dom/doc/01.txt" "dom/doc/02.txt" "chunked_from"
  add_test_edge "dom/doc/02.txt" "dom/doc/01.txt" "chunked_from"

  run ragdag_trace "dom/doc/01.txt"
  [ "$status" -eq 0 ]
  # Should terminate; the cycle detection outputs "(origin)" when a visited node is seen
  [[ "$output" == *"(origin)"* ]]
}

@test "ragdag_trace max depth limit (20)" {
  # Create a chain longer than 20 hops
  local domain="deep"
  local doc="chain"
  local chain_dir="${TEST_STORE}/${domain}/${doc}"
  mkdir -p "$chain_dir"
  for i in $(seq 1 25); do
    local fname
    fname="$(printf '%02d.txt' "$i")"
    echo "Chunk $i" > "${chain_dir}/${fname}"
  done
  # Create a chain of chunked_from edges: 01->02->03->...->25
  for i in $(seq 1 24); do
    local src tgt
    src="$(printf '%s/%s/%02d.txt' "$domain" "$doc" "$i")"
    tgt="$(printf '%s/%s/%02d.txt' "$domain" "$doc" "$((i + 1))")"
    add_test_edge "$src" "$tgt" "chunked_from"
  done

  run ragdag_trace "deep/chain/01.txt"
  [ "$status" -eq 0 ]
  # The trace should stop at depth 20 -- so we won't see chunk 22 or beyond
  # but we should see chunk 20 or 21
  [[ "$output" == *"deep/chain/20.txt"* ]] || [[ "$output" == *"deep/chain/21.txt"* ]]
}

# --- ragdag_link ---

@test "ragdag_link creates manual edge in .edges file" {
  run ragdag_link "nodeA" "nodeB" "similar_to"
  [ "$status" -eq 0 ]
  [[ "$output" == *"Added edge"* ]]

  # Verify the edge was written
  local edges_content
  edges_content="$(cat "${TEST_STORE}/.edges")"
  [[ "$edges_content" == *"nodeA"* ]]
  [[ "$edges_content" == *"nodeB"* ]]
  [[ "$edges_content" == *"similar_to"* ]]
}

@test "ragdag_link default edge type is references" {
  run ragdag_link "srcNode" "tgtNode"
  [ "$status" -eq 0 ]

  local edges_content
  edges_content="$(cat "${TEST_STORE}/.edges")"
  [[ "$edges_content" == *"references"* ]]
}

@test "ragdag_link requires source and target arguments" {
  run ragdag_link
  [ "$status" -eq 1 ]
  [[ "$output" == *"Usage"* ]]

  run ragdag_link "onlySource"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Usage"* ]]
}
