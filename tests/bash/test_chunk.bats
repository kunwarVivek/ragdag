#!/usr/bin/env bats

# Tests for lib/chunk.sh — chunking strategies for ragdag

load test_helper

setup() {
  setup_store
  source "${RAGDAG_DIR}/lib/chunk.sh"
  CHUNK_OUTDIR="${TEST_TMPDIR}/chunks_out"
  mkdir -p "$CHUNK_OUTDIR"
}

teardown() {
  teardown_store
}

# --- _chunk_filename ---

@test "_chunk_filename produces zero-padded filenames" {
  run _chunk_filename 1
  [ "$status" -eq 0 ]
  [ "$output" = "01.txt" ]
}

@test "_chunk_filename zero-pads single digit to two digits" {
  run _chunk_filename 9
  [ "$status" -eq 0 ]
  [ "$output" = "09.txt" ]
}

@test "_chunk_filename handles double-digit numbers" {
  run _chunk_filename 12
  [ "$status" -eq 0 ]
  [ "$output" = "12.txt" ]
}

# --- _write_chunk ---

@test "_write_chunk writes non-empty content to file" {
  run _write_chunk "$CHUNK_OUTDIR" 1 "Hello world"
  [ "$status" -eq 0 ]
  [ -f "${CHUNK_OUTDIR}/01.txt" ]
  run cat "${CHUNK_OUTDIR}/01.txt"
  [[ "$output" == *"Hello world"* ]]
}

@test "_write_chunk skips empty content" {
  run _write_chunk "$CHUNK_OUTDIR" 1 ""
  [ "$status" -eq 1 ]
  [ ! -f "${CHUNK_OUTDIR}/01.txt" ]
}

@test "_write_chunk skips whitespace-only content" {
  run _write_chunk "$CHUNK_OUTDIR" 1 "

  "
  [ "$status" -eq 1 ]
  [ ! -f "${CHUNK_OUTDIR}/01.txt" ]
}

# --- _get_overlap_text ---

@test "_get_overlap_text returns last N characters" {
  run _get_overlap_text "Hello World" 5
  [ "$status" -eq 0 ]
  [ "$output" = "World" ]
}

@test "_get_overlap_text returns full text when shorter than overlap" {
  run _get_overlap_text "Hi" 10
  [ "$status" -eq 0 ]
  [ "$output" = "Hi" ]
}

@test "_get_overlap_text returns nothing for zero overlap" {
  run _get_overlap_text "Hello" 0
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}

# --- ragdag_chunk with heading strategy ---

@test "ragdag_chunk heading strategy splits on markdown headers" {
  local input
  input="$(create_test_markdown)"
  run ragdag_chunk "$input" "$CHUNK_OUTDIR" heading 5000 0
  [ "$status" -eq 0 ]
  # The test markdown has 3 headers: Introduction, Features, Installation
  # Should produce 3 chunk files
  [ "$output" = "3" ]
  [ -f "${CHUNK_OUTDIR}/01.txt" ]
  [ -f "${CHUNK_OUTDIR}/02.txt" ]
  [ -f "${CHUNK_OUTDIR}/03.txt" ]
}

@test "ragdag_chunk heading strategy chunk files contain correct sections" {
  local input
  input="$(create_test_markdown)"
  ragdag_chunk "$input" "$CHUNK_OUTDIR" heading 5000 0 >/dev/null
  run cat "${CHUNK_OUTDIR}/01.txt"
  [[ "$output" == *"Introduction"* ]]
  run cat "${CHUNK_OUTDIR}/02.txt"
  [[ "$output" == *"Features"* ]]
  run cat "${CHUNK_OUTDIR}/03.txt"
  [[ "$output" == *"Installation"* ]]
}

@test "ragdag_chunk heading strategy respects chunk_size overflow" {
  # Create a markdown file where a single section exceeds chunk_size
  local input="${TEST_TMPDIR}/big_section.md"
  {
    echo "# First"
    echo ""
    # Generate content that will exceed 50 characters
    for i in $(seq 1 20); do
      echo "This is line $i with enough content to cause overflow."
    done
    echo ""
    echo "# Second"
    echo "Short section."
  } > "$input"

  run ragdag_chunk "$input" "$CHUNK_OUTDIR" heading 50 0
  [ "$status" -eq 0 ]
  # With tiny chunk_size, the big section should be flushed multiple times
  local count
  count="$output"
  [ "$count" -gt 2 ]
}

@test "ragdag_chunk returns count of chunks created" {
  local input
  input="$(create_test_markdown)"
  run ragdag_chunk "$input" "$CHUNK_OUTDIR" heading 5000 0
  [ "$status" -eq 0 ]
  # Output should be a number
  [[ "$output" =~ ^[0-9]+$ ]]
  [ "$output" -gt 0 ]
}

# --- ragdag_chunk with paragraph strategy ---

@test "ragdag_chunk paragraph strategy splits on blank lines" {
  local input="${TEST_TMPDIR}/paragraphs.txt"
  cat > "$input" <<'TXT'
First paragraph has some text.
It continues on a second line.

Second paragraph is here.
Also has multiple lines.

Third paragraph stands alone.
TXT

  run ragdag_chunk "$input" "$CHUNK_OUTDIR" paragraph 5000 0
  [ "$status" -eq 0 ]
  # All three paragraphs fit in one chunk at 5000 size
  # But let's test with small chunk_size to force splits
  rm -f "$CHUNK_OUTDIR"/*.txt
  run ragdag_chunk "$input" "$CHUNK_OUTDIR" paragraph 40 0
  [ "$status" -eq 0 ]
  local count="$output"
  [ "$count" -ge 2 ]
}

# --- ragdag_chunk with fixed strategy ---

@test "ragdag_chunk fixed strategy splits at fixed character count" {
  local input="${TEST_TMPDIR}/fixed_input.txt"
  # Create content that is clearly longer than 50 chars
  cat > "$input" <<'TXT'
AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC DDDDDDDDDD EEEEEEEEEE
FFFFFFFFFF GGGGGGGGGG HHHHHHHHHH IIIIIIIIII JJJJJJJJJJ
TXT

  run ragdag_chunk "$input" "$CHUNK_OUTDIR" fixed 50 0
  [ "$status" -eq 0 ]
  local count="$output"
  [ "$count" -ge 2 ]
}

# --- ragdag_chunk with function strategy ---

@test "ragdag_chunk function strategy splits on def/class/function boundaries" {
  local input="${TEST_TMPDIR}/code.py"
  cat > "$input" <<'PY'
import os

def first_function():
    return 1

def second_function():
    return 2

class MyClass:
    pass
PY

  run ragdag_chunk "$input" "$CHUNK_OUTDIR" function 5000 0
  [ "$status" -eq 0 ]
  local count="$output"
  # Should split at def and class boundaries: import block, first_function, second_function, class
  [ "$count" -ge 3 ]
}

# --- Unknown strategy fallback ---

@test "ragdag_chunk unknown strategy falls back to fixed" {
  local input="${TEST_TMPDIR}/fallback.txt"
  cat > "$input" <<'TXT'
AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC DDDDDDDDDD EEEEEEEEEE
FFFFFFFFFF GGGGGGGGGG HHHHHHHHHH IIIIIIIIII JJJJJJJJJJ
TXT

  # ragdag_warn writes to stderr; run captures both stdout+stderr.
  # Verify it still produces chunk files (the fallback worked).
  run ragdag_chunk "$input" "$CHUNK_OUTDIR" nonexistent 50 0
  [ "$status" -eq 0 ]
  # Output should contain the warning about unknown strategy
  [[ "$output" == *"Unknown chunk strategy"* ]]
  # Verify chunks were actually created by counting files
  local file_count=0
  for f in "$CHUNK_OUTDIR"/*.txt; do
    [[ -f "$f" ]] && file_count=$((file_count + 1))
  done
  [ "$file_count" -ge 2 ]
}

# --- Overlap ---

@test "overlap text appears at start of next chunk" {
  local input="${TEST_TMPDIR}/overlap_test.md"
  cat > "$input" <<'MD'
# Section One
Alpha Bravo Charlie Delta Echo.
# Section Two
Foxtrot Golf Hotel India Juliet.
MD

  ragdag_chunk "$input" "$CHUNK_OUTDIR" heading 5000 15 >/dev/null

  # The second chunk should start with overlap from end of first chunk
  [ -f "${CHUNK_OUTDIR}/01.txt" ]
  [ -f "${CHUNK_OUTDIR}/02.txt" ]
  # Read the end of chunk 1
  local chunk1_content
  chunk1_content="$(cat "${CHUNK_OUTDIR}/01.txt")"
  local last_chars="${chunk1_content: -15}"
  # Chunk 2 should start with those last chars from chunk 1
  local chunk2_content
  chunk2_content="$(cat "${CHUNK_OUTDIR}/02.txt")"
  local chunk2_start="${chunk2_content:0:15}"
  [ "$chunk2_start" = "$last_chars" ]
}
