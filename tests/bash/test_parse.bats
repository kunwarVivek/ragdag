#!/usr/bin/env bats

# Tests for lib/parse.sh — file type detection and text extraction for ragdag

load test_helper

setup() {
  setup_store
  source "${RAGDAG_DIR}/lib/parse.sh"
}

teardown() {
  teardown_store
}

# --- ragdag_detect_type ---

@test "ragdag_detect_type detects .md as markdown" {
  local f="${TEST_TMPDIR}/doc.md"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "markdown" ]
}

@test "ragdag_detect_type detects .markdown as markdown" {
  local f="${TEST_TMPDIR}/doc.markdown"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "markdown" ]
}

@test "ragdag_detect_type detects .txt as text" {
  local f="${TEST_TMPDIR}/doc.txt"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "text" ]
}

@test "ragdag_detect_type detects .log as text" {
  local f="${TEST_TMPDIR}/app.log"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "text" ]
}

@test "ragdag_detect_type detects .py as code" {
  local f="${TEST_TMPDIR}/script.py"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "code" ]
}

@test "ragdag_detect_type detects .js as code" {
  local f="${TEST_TMPDIR}/app.js"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "code" ]
}

@test "ragdag_detect_type detects .sh as code" {
  local f="${TEST_TMPDIR}/run.sh"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "code" ]
}

@test "ragdag_detect_type detects .pdf as pdf" {
  local f="${TEST_TMPDIR}/paper.pdf"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "pdf" ]
}

@test "ragdag_detect_type detects .html as html" {
  local f="${TEST_TMPDIR}/page.html"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "html" ]
}

@test "ragdag_detect_type detects .htm as html" {
  local f="${TEST_TMPDIR}/page.htm"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "html" ]
}

@test "ragdag_detect_type detects .csv as csv" {
  local f="${TEST_TMPDIR}/data.csv"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "csv" ]
}

@test "ragdag_detect_type detects .json as json" {
  local f="${TEST_TMPDIR}/config.json"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "json" ]
}

@test "ragdag_detect_type detects .jsonl as json" {
  local f="${TEST_TMPDIR}/data.jsonl"
  touch "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  [ "$output" = "json" ]
}

@test "ragdag_detect_type returns unknown for unsupported extensions" {
  local f="${TEST_TMPDIR}/mystery.xyz123"
  echo "some binary-ish content" > "$f"
  run ragdag_detect_type "$f"
  [ "$status" -eq 0 ]
  # Should fall through to mime-based detection; plain text content -> text or unknown
  # The exact result depends on `file --mime-type`, but it should not crash
  [[ "$output" =~ ^(text|unknown)$ ]]
}

# --- ragdag_parse ---

@test "ragdag_parse text is passthrough" {
  local f
  f="$(create_test_text)"
  run ragdag_parse "$f" text
  [ "$status" -eq 0 ]
  [[ "$output" == *"test content for ragdag"* ]]
}

@test "ragdag_parse markdown strips YAML frontmatter" {
  local f="${TEST_TMPDIR}/frontmatter.md"
  cat > "$f" <<'MD'
---
title: Test Doc
author: Someone
---
# Heading

Body content here.
MD

  run ragdag_parse "$f" markdown
  [ "$status" -eq 0 ]
  # Frontmatter should be stripped
  [[ "$output" != *"title: Test Doc"* ]]
  [[ "$output" != *"author: Someone"* ]]
  # Content should remain
  [[ "$output" == *"# Heading"* ]]
  [[ "$output" == *"Body content here."* ]]
}

@test "ragdag_parse code is passthrough preserving structure" {
  local f="${TEST_TMPDIR}/sample.py"
  cat > "$f" <<'PY'
def hello():
    print("world")

class Foo:
    pass
PY

  run ragdag_parse "$f" code
  [ "$status" -eq 0 ]
  [[ "$output" == *"def hello():"* ]]
  [[ "$output" == *"class Foo:"* ]]
  [[ "$output" == *'print("world")'* ]]
}

# --- _parse_markdown ---

@test "_parse_markdown handles frontmatter with ---/--- delimiters" {
  local f="${TEST_TMPDIR}/fm.md"
  cat > "$f" <<'MD'
---
key: value
list:
  - one
  - two
---

# Title

Real content.
MD

  run _parse_markdown "$f"
  [ "$status" -eq 0 ]
  # Frontmatter stripped
  [[ "$output" != *"key: value"* ]]
  [[ "$output" != *"- one"* ]]
  # Content preserved
  [[ "$output" == *"# Title"* ]]
  [[ "$output" == *"Real content."* ]]
}

@test "_parse_markdown passes through markdown without frontmatter" {
  local f="${TEST_TMPDIR}/nofm.md"
  cat > "$f" <<'MD'
# Just a Heading

Some paragraph text.
MD

  run _parse_markdown "$f"
  [ "$status" -eq 0 ]
  [[ "$output" == *"# Just a Heading"* ]]
  [[ "$output" == *"Some paragraph text."* ]]
}

# --- _parse_csv ---

@test "_parse_csv converts to key-value text format" {
  local f="${TEST_TMPDIR}/data.csv"
  cat > "$f" <<'CSV'
name,age,city
Alice,30,NYC
Bob,25,LA
CSV

  run _parse_csv "$f"
  [ "$status" -eq 0 ]
  [[ "$output" == *"--- Record 1 ---"* ]]
  [[ "$output" == *"name: Alice"* ]]
  [[ "$output" == *"age: 30"* ]]
  [[ "$output" == *"city: NYC"* ]]
  [[ "$output" == *"--- Record 2 ---"* ]]
  [[ "$output" == *"name: Bob"* ]]
  [[ "$output" == *"age: 25"* ]]
  [[ "$output" == *"city: LA"* ]]
}

@test "_parse_csv handles quoted fields" {
  local f="${TEST_TMPDIR}/quoted.csv"
  cat > "$f" <<'CSV'
name,description
"Alice","A person"
"Bob","Another person"
CSV

  run _parse_csv "$f"
  [ "$status" -eq 0 ]
  [[ "$output" == *"name: Alice"* ]]
  [[ "$output" == *"description: A person"* ]]
}
