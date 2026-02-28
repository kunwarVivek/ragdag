#!/usr/bin/env bats
# test_init.bats — Tests for lib/init.sh

load test_helper

# Source init.sh in addition to helper
source "${RAGDAG_DIR:-$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)}/lib/init.sh"

setup() {
  # Use a clean tmpdir (NOT setup_store, since we test init creating the store)
  TEST_TMPDIR="$(mktemp -d)"
  INIT_TARGET="${TEST_TMPDIR}/project"
  mkdir -p "$INIT_TARGET"
}

teardown() {
  if [[ -n "${TEST_TMPDIR:-}" ]] && [[ -d "${TEST_TMPDIR:-}" ]]; then
    rm -rf "$TEST_TMPDIR"
  fi
}

# --- ragdag_init ---

@test "ragdag init creates .ragdag directory" {
  run ragdag_init "$INIT_TARGET"
  [ "$status" -eq 0 ]
  [ -d "${INIT_TARGET}/.ragdag" ]
}

@test "ragdag init creates .config with default values" {
  ragdag_init "$INIT_TARGET"
  local config="${INIT_TARGET}/.ragdag/.config"
  [ -f "$config" ]

  # Verify default config has expected sections and values
  run ragdag_config_get_from "$config" "general.chunk_strategy"
  [ "$output" = "heading" ]

  run ragdag_config_get_from "$config" "embedding.provider"
  [ "$output" = "none" ]
}

@test "ragdag init creates .edges with header" {
  ragdag_init "$INIT_TARGET"
  local edges="${INIT_TARGET}/.ragdag/.edges"
  [ -f "$edges" ]
  # File should contain a header comment
  run head -1 "$edges"
  [[ "$output" == *"source"* ]]
}

@test "ragdag init creates .processed with header" {
  ragdag_init "$INIT_TARGET"
  local processed="${INIT_TARGET}/.ragdag/.processed"
  [ -f "$processed" ]
  run head -1 "$processed"
  [[ "$output" == *"source_path"* ]]
}

@test "ragdag init creates .domain-rules with header" {
  ragdag_init "$INIT_TARGET"
  local rules="${INIT_TARGET}/.ragdag/.domain-rules"
  [ -f "$rules" ]
  run head -1 "$rules"
  [[ "$output" == *"domain"* ]]
}

@test "ragdag init is idempotent — running twice does not break anything" {
  ragdag_init "$INIT_TARGET"
  # First init creates the store
  [ -d "${INIT_TARGET}/.ragdag" ]

  # Set a custom config value
  ragdag_config_set_in "${INIT_TARGET}/.ragdag/.config" "general.chunk_size" "9999"

  # Second init should not overwrite
  run ragdag_init "$INIT_TARGET"
  [ "$status" -eq 0 ]
  [[ "$output" == *"Already initialized"* ]]

  # Custom value should still be there
  run ragdag_config_get_from "${INIT_TARGET}/.ragdag/.config" "general.chunk_size"
  [ "$output" = "9999" ]
}

@test "ragdag init with custom path works" {
  local custom_dir="${TEST_TMPDIR}/custom/nested/path"
  mkdir -p "$custom_dir"
  run ragdag_init "$custom_dir"
  [ "$status" -eq 0 ]
  [ -d "${custom_dir}/.ragdag" ]
  [ -f "${custom_dir}/.ragdag/.config" ]
}

# --- _ragdag_check_deps ---

@test "_ragdag_check_deps reports available tools without error" {
  run _ragdag_check_deps
  [ "$status" -eq 0 ]
  # Should report bash and grep at minimum since they exist
  [[ "$output" == *"bash"* ]]
  [[ "$output" == *"grep"* ]]
}

@test "_ragdag_check_deps reports awk and sort" {
  run _ragdag_check_deps
  [ "$status" -eq 0 ]
  [[ "$output" == *"awk"* ]]
  [[ "$output" == *"sort"* ]]
}
