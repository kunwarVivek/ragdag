#!/usr/bin/env bash
# chunk.sh — Chunking strategies for ragdag

# Main chunking entry point
# Usage: ragdag_chunk <input_file> <output_dir> [strategy] [chunk_size] [overlap]
# Writes numbered .txt files (01.txt, 02.txt, ...) to output_dir
ragdag_chunk() {
  local input="$1"
  local outdir="$2"
  local strategy="${3:-heading}"
  local chunk_size="${4:-1000}"
  local overlap="${5:-100}"

  mkdir -p "$outdir"

  case "$strategy" in
    heading)    _chunk_heading "$input" "$outdir" "$chunk_size" "$overlap" ;;
    paragraph)  _chunk_paragraph "$input" "$outdir" "$chunk_size" "$overlap" ;;
    fixed)      _chunk_fixed "$input" "$outdir" "$chunk_size" "$overlap" ;;
    function)   _chunk_function "$input" "$outdir" "$chunk_size" "$overlap" ;;
    *)
      ragdag_warn "Unknown chunk strategy: $strategy — using fixed"
      _chunk_fixed "$input" "$outdir" "$chunk_size" "$overlap"
      ;;
  esac

  # Count output chunks
  local count=0
  for f in "$outdir"/*.txt; do
    [[ -f "$f" ]] && count=$((count + 1))
  done
  echo "$count"
}

# Format chunk number with zero padding
_chunk_filename() {
  printf "%02d.txt" "$1"
}

# Write a chunk buffer to a file
# Usage: _write_chunk <output_dir> <chunk_number> <content>
_write_chunk() {
  local outdir="$1"
  local num="$2"
  local content="$3"

  # Skip empty chunks
  local trimmed
  trimmed="$(echo "$content" | sed '/^[[:space:]]*$/d')"
  if [[ -z "$trimmed" ]]; then
    return 1
  fi

  local fname
  fname="$(_chunk_filename "$num")"
  printf '%s\n' "$content" > "${outdir}/${fname}"
  return 0
}

# Get the last N characters of text for overlap
_get_overlap_text() {
  local text="$1"
  local overlap_chars="$2"

  if [[ "$overlap_chars" -le 0 ]]; then
    return
  fi

  local len=${#text}
  if [[ "$len" -le "$overlap_chars" ]]; then
    echo "$text"
  else
    echo "${text:$((len - overlap_chars))}"
  fi
}

# Strategy: heading — split on markdown headers
_chunk_heading() {
  local input="$1"
  local outdir="$2"
  local chunk_size="$3"
  local overlap="$4"

  local chunk_num=1
  local buffer=""
  local buffer_len=0

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Check if line is a markdown header
    local is_header=0
    case "$line" in
      '#'*) is_header=1 ;;
    esac

    # If we hit a header and buffer is non-empty, flush
    if [[ "$is_header" -eq 1 ]] && [[ "$buffer_len" -gt 0 ]]; then
      if _write_chunk "$outdir" "$chunk_num" "$buffer"; then
        chunk_num=$((chunk_num + 1))
      fi
      # Start new buffer with overlap from end of previous
      if [[ "$overlap" -gt 0 ]]; then
        buffer="$(_get_overlap_text "$buffer" "$overlap")"
        buffer="${buffer}
${line}"
      else
        buffer="$line"
      fi
      buffer_len=${#buffer}
      continue
    fi

    # Add line to buffer
    if [[ -z "$buffer" ]]; then
      buffer="$line"
    else
      buffer="${buffer}
${line}"
    fi
    buffer_len=${#buffer}

    # If buffer exceeds chunk_size, flush even without a header
    if [[ "$buffer_len" -ge "$chunk_size" ]]; then
      if _write_chunk "$outdir" "$chunk_num" "$buffer"; then
        chunk_num=$((chunk_num + 1))
      fi
      if [[ "$overlap" -gt 0 ]]; then
        buffer="$(_get_overlap_text "$buffer" "$overlap")"
      else
        buffer=""
      fi
      buffer_len=${#buffer}
    fi
  done < "$input"

  # Flush remaining buffer
  if [[ "$buffer_len" -gt 0 ]]; then
    _write_chunk "$outdir" "$chunk_num" "$buffer"
  fi
}

# Strategy: paragraph — split on blank lines
_chunk_paragraph() {
  local input="$1"
  local outdir="$2"
  local chunk_size="$3"
  local overlap="$4"

  local chunk_num=1
  local buffer=""
  local buffer_len=0
  local para=""

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Blank line = paragraph break
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*$ ]]; then
      if [[ -n "$para" ]]; then
        # Would adding this paragraph exceed chunk_size?
        local new_len=$(( buffer_len + ${#para} + 2 ))
        if [[ "$new_len" -ge "$chunk_size" ]] && [[ "$buffer_len" -gt 0 ]]; then
          # Flush buffer
          if _write_chunk "$outdir" "$chunk_num" "$buffer"; then
            chunk_num=$((chunk_num + 1))
          fi
          if [[ "$overlap" -gt 0 ]]; then
            buffer="$(_get_overlap_text "$buffer" "$overlap")"
            buffer="${buffer}

${para}"
          else
            buffer="$para"
          fi
        else
          if [[ -z "$buffer" ]]; then
            buffer="$para"
          else
            buffer="${buffer}

${para}"
          fi
        fi
        buffer_len=${#buffer}
        para=""
      fi
      continue
    fi

    # Accumulate paragraph
    if [[ -z "$para" ]]; then
      para="$line"
    else
      para="${para}
${line}"
    fi
  done < "$input"

  # Handle last paragraph
  if [[ -n "$para" ]]; then
    local new_len=$(( buffer_len + ${#para} + 2 ))
    if [[ "$new_len" -ge "$chunk_size" ]] && [[ "$buffer_len" -gt 0 ]]; then
      if _write_chunk "$outdir" "$chunk_num" "$buffer"; then
        chunk_num=$((chunk_num + 1))
      fi
      buffer="$para"
    else
      if [[ -z "$buffer" ]]; then
        buffer="$para"
      else
        buffer="${buffer}

${para}"
      fi
    fi
  fi

  # Flush remaining
  if [[ -n "$buffer" ]]; then
    _write_chunk "$outdir" "$chunk_num" "$buffer"
  fi
}

# Strategy: fixed — split at fixed character count
_chunk_fixed() {
  local input="$1"
  local outdir="$2"
  local chunk_size="$3"
  local overlap="$4"

  local chunk_num=1
  local buffer=""
  local buffer_len=0

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -z "$buffer" ]]; then
      buffer="$line"
    else
      buffer="${buffer}
${line}"
    fi
    buffer_len=${#buffer}

    if [[ "$buffer_len" -ge "$chunk_size" ]]; then
      if _write_chunk "$outdir" "$chunk_num" "$buffer"; then
        chunk_num=$((chunk_num + 1))
      fi
      if [[ "$overlap" -gt 0 ]]; then
        buffer="$(_get_overlap_text "$buffer" "$overlap")"
      else
        buffer=""
      fi
      buffer_len=${#buffer}
    fi
  done < "$input"

  # Flush remaining
  if [[ -n "$buffer" ]]; then
    _write_chunk "$outdir" "$chunk_num" "$buffer"
  fi
}

# Strategy: function — split on function/class definitions (code files)
_chunk_function() {
  local input="$1"
  local outdir="$2"
  local chunk_size="$3"
  local overlap="$4"

  local chunk_num=1
  local buffer=""
  local buffer_len=0

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Detect function/class boundaries
    local is_boundary=0
    case "$line" in
      def\ *|class\ *|function\ *|function\ *\(*) is_boundary=1 ;;
      *\(\)\ \{*|*\(\)\ \{) is_boundary=1 ;;  # bash functions
      func\ *) is_boundary=1 ;;  # Go
      pub\ fn\ *|fn\ *) is_boundary=1 ;;  # Rust
      export\ function\ *|export\ const\ *|const\ *=\ *\(*) is_boundary=1 ;;  # JS/TS
    esac

    if [[ "$is_boundary" -eq 1 ]] && [[ "$buffer_len" -gt 0 ]]; then
      if _write_chunk "$outdir" "$chunk_num" "$buffer"; then
        chunk_num=$((chunk_num + 1))
      fi
      if [[ "$overlap" -gt 0 ]]; then
        buffer="$(_get_overlap_text "$buffer" "$overlap")"
        buffer="${buffer}
${line}"
      else
        buffer="$line"
      fi
      buffer_len=${#buffer}
      continue
    fi

    if [[ -z "$buffer" ]]; then
      buffer="$line"
    else
      buffer="${buffer}
${line}"
    fi
    buffer_len=${#buffer}

    # Safety: flush if way over chunk_size
    if [[ "$buffer_len" -ge "$((chunk_size * 2))" ]]; then
      if _write_chunk "$outdir" "$chunk_num" "$buffer"; then
        chunk_num=$((chunk_num + 1))
      fi
      if [[ "$overlap" -gt 0 ]]; then
        buffer="$(_get_overlap_text "$buffer" "$overlap")"
      else
        buffer=""
      fi
      buffer_len=${#buffer}
    fi
  done < "$input"

  # Flush remaining
  if [[ -n "$buffer" ]]; then
    _write_chunk "$outdir" "$chunk_num" "$buffer"
  fi
}
