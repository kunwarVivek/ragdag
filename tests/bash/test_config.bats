#!/usr/bin/env bats
# test_config.bats — Tests for lib/config.sh

load test_helper

setup() {
  setup_store
}

teardown() {
  teardown_store
}

# --- ragdag_config_get_from ---

@test "ragdag_config_get_from reads existing key from section" {
  run ragdag_config_get_from "${TEST_STORE}/.config" "general.chunk_strategy"
  [ "$status" -eq 0 ]
  [ "$output" = "heading" ]
}

@test "ragdag_config_get_from reads numeric value" {
  run ragdag_config_get_from "${TEST_STORE}/.config" "general.chunk_size"
  [ "$status" -eq 0 ]
  [ "$output" = "1000" ]
}

@test "ragdag_config_get_from returns default for missing key" {
  run ragdag_config_get_from "${TEST_STORE}/.config" "general.nonexistent" "fallback"
  [ "$status" -eq 0 ]
  [ "$output" = "fallback" ]
}

@test "ragdag_config_get_from returns empty string for missing key without default" {
  run ragdag_config_get_from "${TEST_STORE}/.config" "general.nonexistent"
  [ "$status" -eq 0 ]
  [ "$output" = "" ]
}

@test "ragdag_config_get_from reads from correct section not wrong one" {
  # 'model' exists in both [embedding] and [llm] sections
  run ragdag_config_get_from "${TEST_STORE}/.config" "embedding.model"
  [ "$status" -eq 0 ]
  [ "$output" = "text-embedding-3-small" ]

  run ragdag_config_get_from "${TEST_STORE}/.config" "llm.model"
  [ "$status" -eq 0 ]
  [ "$output" = "gpt-4o-mini" ]
}

@test "ragdag_config_get_from returns default for missing config file" {
  run ragdag_config_get_from "${TEST_TMPDIR}/nonexistent.ini" "general.key" "mydefault"
  [ "$status" -eq 0 ]
  [ "$output" = "mydefault" ]
}

@test "ragdag_config_get_from reads key from last section" {
  # edges is the last section in the config
  run ragdag_config_get_from "${TEST_STORE}/.config" "edges.auto_relate"
  [ "$status" -eq 0 ]
  [ "$output" = "false" ]
}

# --- ragdag_config_set_in ---

@test "ragdag_config_set_in sets new key in existing section" {
  ragdag_config_set_in "${TEST_STORE}/.config" "general.new_key" "new_value"
  run ragdag_config_get_from "${TEST_STORE}/.config" "general.new_key"
  [ "$status" -eq 0 ]
  [ "$output" = "new_value" ]
}

@test "ragdag_config_set_in replaces existing value" {
  ragdag_config_set_in "${TEST_STORE}/.config" "general.chunk_size" "2000"
  run ragdag_config_get_from "${TEST_STORE}/.config" "general.chunk_size"
  [ "$status" -eq 0 ]
  [ "$output" = "2000" ]
}

@test "ragdag_config_set_in creates new section if missing" {
  ragdag_config_set_in "${TEST_STORE}/.config" "newsection.mykey" "myvalue"
  run ragdag_config_get_from "${TEST_STORE}/.config" "newsection.mykey"
  [ "$status" -eq 0 ]
  [ "$output" = "myvalue" ]
}

@test "ragdag_config_set_in does not corrupt other sections" {
  ragdag_config_set_in "${TEST_STORE}/.config" "general.chunk_size" "5000"
  # Verify embedding section is still intact
  run ragdag_config_get_from "${TEST_STORE}/.config" "embedding.provider"
  [ "$status" -eq 0 ]
  [ "$output" = "none" ]
}

@test "ragdag_config_set_in fails without section.key format" {
  run ragdag_config_set_in "${TEST_STORE}/.config" "nosectionkey" "val"
  [ "$status" -ne 0 ]
}

# --- ragdag_config_show ---

@test "ragdag_config_show displays all config content" {
  run ragdag_config_show "${TEST_STORE}/.config"
  [ "$status" -eq 0 ]
  [[ "$output" == *"[general]"* ]]
  [[ "$output" == *"[embedding]"* ]]
  [[ "$output" == *"[llm]"* ]]
  [[ "$output" == *"[search]"* ]]
  [[ "$output" == *"[edges]"* ]]
}

@test "ragdag_config_show fails for nonexistent file" {
  run ragdag_config_show "${TEST_TMPDIR}/no_such_config"
  [ "$status" -ne 0 ]
}

# --- Round-trip and isolation ---

@test "config round-trip: set then get returns same value" {
  ragdag_config_set_in "${TEST_STORE}/.config" "search.top_k" "42"
  run ragdag_config_get_from "${TEST_STORE}/.config" "search.top_k"
  [ "$status" -eq 0 ]
  [ "$output" = "42" ]
}

@test "section isolation: same key in different sections returns correct value" {
  ragdag_config_set_in "${TEST_STORE}/.config" "general.custom" "general_val"
  ragdag_config_set_in "${TEST_STORE}/.config" "search.custom" "search_val"

  local general_val search_val
  general_val="$(ragdag_config_get_from "${TEST_STORE}/.config" "general.custom")"
  search_val="$(ragdag_config_get_from "${TEST_STORE}/.config" "search.custom")"

  [ "$general_val" = "general_val" ]
  [ "$search_val" = "search_val" ]
}
