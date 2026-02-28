#!/usr/bin/env bats
# test_compat.bats — Tests for lib/compat.sh

load test_helper

setup() {
  setup_store
}

teardown() {
  teardown_store
}

# --- ragdag_sanitize ---

@test "ragdag_sanitize lowercases and keeps alphanumeric, dash, underscore, dot" {
  run ragdag_sanitize "Hello-World_123.txt"
  [ "$status" -eq 0 ]
  [ "$output" = "hello-world_123.txt" ]
}

@test "ragdag_sanitize strips spaces" {
  run ragdag_sanitize "my file name"
  [ "$status" -eq 0 ]
  [ "$output" = "myfilename" ]
}

@test "ragdag_sanitize strips special characters" {
  run ragdag_sanitize "foo@bar#baz!qux"
  [ "$status" -eq 0 ]
  [ "$output" = "foobarbazqux" ]
}

@test "ragdag_sanitize strips unicode characters" {
  run ragdag_sanitize "cafe"
  [ "$status" -eq 0 ]
  [ "$output" = "cafe" ]
}

@test "ragdag_sanitize returns empty string for empty input" {
  run ragdag_sanitize ""
  [ "$status" -eq 0 ]
  [ "$output" = "" ]
}

@test "ragdag_sanitize returns empty string for all-special input" {
  run ragdag_sanitize "@#\$%^&*()"
  [ "$status" -eq 0 ]
  [ "$output" = "" ]
}

# --- ragdag_has ---

@test "ragdag_has detects available command (bash)" {
  run ragdag_has bash
  [ "$status" -eq 0 ]
}

@test "ragdag_has detects available command (grep)" {
  run ragdag_has grep
  [ "$status" -eq 0 ]
}

@test "ragdag_has returns failure for nonexistent command" {
  run ragdag_has totally_nonexistent_command_xyz_99
  [ "$status" -ne 0 ]
}

# --- ragdag_sha256 ---

@test "ragdag_sha256 produces a 64-character hex hash" {
  local tmpfile="${TEST_TMPDIR}/hashtest.txt"
  echo "test content" > "$tmpfile"
  run ragdag_sha256 "$tmpfile"
  [ "$status" -eq 0 ]
  # Must be exactly 64 hex characters
  [[ "$output" =~ ^[0-9a-f]{64}$ ]]
}

@test "ragdag_sha256 produces consistent hash for same content" {
  local tmpfile="${TEST_TMPDIR}/hashtest.txt"
  echo "deterministic content" > "$tmpfile"
  local hash1 hash2
  hash1="$(ragdag_sha256 "$tmpfile")"
  hash2="$(ragdag_sha256 "$tmpfile")"
  [ "$hash1" = "$hash2" ]
}

@test "ragdag_sha256 produces different hashes for different content" {
  local file1="${TEST_TMPDIR}/hash1.txt"
  local file2="${TEST_TMPDIR}/hash2.txt"
  echo "content one" > "$file1"
  echo "content two" > "$file2"
  local hash1 hash2
  hash1="$(ragdag_sha256 "$file1")"
  hash2="$(ragdag_sha256 "$file2")"
  [ "$hash1" != "$hash2" ]
}

# --- ragdag_estimate_tokens ---

@test "ragdag_estimate_tokens estimates word_count * 1.3 (integer)" {
  # 5 words * 13/10 = 6
  run ragdag_estimate_tokens "one two three four five"
  [ "$status" -eq 0 ]
  [ "$output" = "6" ]
}

@test "ragdag_estimate_tokens returns 0 for empty string" {
  run ragdag_estimate_tokens ""
  [ "$status" -eq 0 ]
  [ "$output" = "0" ]
}

@test "ragdag_estimate_tokens handles 10 words" {
  # 10 words * 13/10 = 13
  run ragdag_estimate_tokens "a b c d e f g h i j"
  [ "$status" -eq 0 ]
  [ "$output" = "13" ]
}

# --- ragdag_realpath ---

@test "ragdag_realpath resolves an absolute path" {
  local resolved
  resolved="$(ragdag_realpath "$TEST_TMPDIR")"
  # Should return a valid absolute path
  [ -d "$resolved" ]
  [[ "$resolved" = /* ]]
}

@test "ragdag_realpath resolves /tmp on macOS to /private/tmp or keeps /tmp" {
  local resolved
  resolved="$(ragdag_realpath /tmp)"
  # On macOS /tmp -> /private/tmp; on Linux stays /tmp
  [[ "$resolved" = "/tmp" || "$resolved" = "/private/tmp" ]]
}

# --- ragdag_file_size ---

@test "ragdag_file_size returns file size in bytes" {
  local tmpfile="${TEST_TMPDIR}/sizetest.txt"
  echo "hello world" > "$tmpfile"
  run ragdag_file_size "$tmpfile"
  [ "$status" -eq 0 ]
  # "hello world\n" = 12 bytes
  [ "$output" = "12" ]
}

@test "ragdag_file_size returns 0 for empty file" {
  local tmpfile="${TEST_TMPDIR}/empty.txt"
  touch "$tmpfile"
  run ragdag_file_size "$tmpfile"
  [ "$status" -eq 0 ]
  [ "$output" = "0" ]
}

# --- ragdag_color ---

@test "ragdag_color outputs plain text when NO_COLOR is set" {
  NO_COLOR=1 run ragdag_color red "error message"
  [ "$status" -eq 0 ]
  [ "$output" = "error message" ]
}

@test "ragdag_color outputs plain text when stdout is not a tty" {
  # In bats, stdout is not a tty so no escape codes
  run ragdag_color green "success message"
  [ "$status" -eq 0 ]
  [ "$output" = "success message" ]
}

@test "ragdag_color outputs text for unknown color name" {
  run ragdag_color purple "some text"
  [ "$status" -eq 0 ]
  [ "$output" = "some text" ]
}
