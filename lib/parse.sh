#!/usr/bin/env bash
# parse.sh — File type detection and text extraction for ragdag

# Detect file type and return a canonical type string
# Usage: ragdag_detect_type <file_path>
# Output: markdown|text|pdf|html|docx|csv|json|code|unknown
ragdag_detect_type() {
  local file="$1"
  local ext="${file##*.}"
  ext="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"

  case "$ext" in
    md|markdown)      echo "markdown" ;;
    txt|text|log)     echo "text" ;;
    pdf)              echo "pdf" ;;
    html|htm)         echo "html" ;;
    docx)             echo "docx" ;;
    csv|tsv)          echo "csv" ;;
    json|jsonl)       echo "json" ;;
    py|rb|js|ts|go|rs|java|c|cpp|h|hpp|sh|bash|zsh|pl|lua|r|swift|kt|scala|ex|exs|clj|hs|ml|php)
      echo "code" ;;
    yaml|yml|toml|ini|cfg|conf)
      echo "config" ;;
    *)
      # Fallback: use file mime type
      local mime
      mime="$(file --mime-type -b "$file" 2>/dev/null)"
      case "$mime" in
        text/markdown)   echo "markdown" ;;
        text/plain)      echo "text" ;;
        text/html)       echo "html" ;;
        text/csv)        echo "csv" ;;
        application/pdf) echo "pdf" ;;
        application/json) echo "json" ;;
        text/x-*)        echo "code" ;;
        text/*)          echo "text" ;;
        *)               echo "unknown" ;;
      esac
      ;;
  esac
}

# Extract plain text from a file based on its type
# Usage: ragdag_parse <file_path> [type]
# Output: plain text to stdout
ragdag_parse() {
  local file="$1"
  local ftype="${2:-}"

  if [[ -z "$ftype" ]]; then
    ftype="$(ragdag_detect_type "$file")"
  fi

  case "$ftype" in
    markdown)   _parse_markdown "$file" ;;
    text)       cat "$file" ;;
    pdf)        _parse_pdf "$file" ;;
    html)       _parse_html "$file" ;;
    docx)       _parse_docx "$file" ;;
    csv)        _parse_csv "$file" ;;
    json)       _parse_json "$file" ;;
    code)       cat "$file" ;;
    config)     cat "$file" ;;
    unknown)
      ragdag_warn "Unknown file type for: $file — treating as plain text"
      cat "$file"
      ;;
  esac
}

# Strip YAML frontmatter from markdown, keep rest
_parse_markdown() {
  local file="$1"
  local in_frontmatter=0
  local frontmatter_ended=0
  local line_num=0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line_num=$((line_num + 1))

    # Detect YAML frontmatter start (--- at line 1)
    if [[ "$line_num" -eq 1 ]] && [[ "$line" == "---" ]]; then
      in_frontmatter=1
      continue
    fi

    # Detect YAML frontmatter end
    if [[ "$in_frontmatter" -eq 1 ]] && [[ "$line" == "---" ]]; then
      in_frontmatter=0
      frontmatter_ended=1
      continue
    fi

    # Skip frontmatter content
    if [[ "$in_frontmatter" -eq 1 ]]; then
      continue
    fi

    echo "$line"
  done < "$file"
}

# Extract text from PDF
_parse_pdf() {
  local file="$1"
  if ragdag_has pdftotext; then
    pdftotext -layout "$file" -
  else
    ragdag_error "pdftotext not found. Install poppler-utils for PDF support."
    return 1
  fi
}

# Extract text from HTML
_parse_html() {
  local file="$1"
  if ragdag_has pandoc; then
    pandoc -f html -t plain --wrap=none "$file"
  elif ragdag_has lynx; then
    lynx -dump -nolist "$file"
  else
    # Fallback: strip HTML tags with sed
    sed 's/<[^>]*>//g' "$file" | sed '/^[[:space:]]*$/d'
  fi
}

# Extract text from DOCX
_parse_docx() {
  local file="$1"
  if ragdag_has pandoc; then
    pandoc -f docx -t plain --wrap=none "$file"
  else
    ragdag_error "pandoc not found. Install pandoc for DOCX support."
    return 1
  fi
}

# Convert CSV to readable text
_parse_csv() {
  local file="$1"
  # Convert CSV to readable key-value pairs using awk
  awk -F',' '
    NR==1 { for(i=1;i<=NF;i++) headers[i]=$i; next }
    {
      printf "--- Record %d ---\n", NR-1
      for(i=1;i<=NF;i++) {
        gsub(/^"/, "", $i); gsub(/"$/, "", $i)
        gsub(/^"/, "", headers[i]); gsub(/"$/, "", headers[i])
        printf "%s: %s\n", headers[i], $i
      }
      printf "\n"
    }
  ' "$file"
}

# Flatten JSON to readable text
_parse_json() {
  local file="$1"
  if ragdag_has jq; then
    # Flatten to key-value paths
    jq -r '
      [paths(scalars) as $p | {
        key: ($p | map(tostring) | join(".")),
        value: (getpath($p) | tostring)
      }] | .[] | "\(.key): \(.value)"
    ' "$file" 2>/dev/null || cat "$file"
  else
    cat "$file"
  fi
}
