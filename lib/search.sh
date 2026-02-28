#!/usr/bin/env bash
# search.sh — Keyword search with TF-IDF-like scoring for ragdag

ragdag_search() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local config_file="${store_dir}/.config"

  # Parse arguments
  local query=""
  local domain=""
  local mode=""
  local top_k=""
  local json_output=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --domain)   domain="$2"; shift 2 ;;
      --keyword)  mode="keyword"; shift ;;
      --vector)   mode="vector"; shift ;;
      --hybrid)   mode="hybrid"; shift ;;
      --top)      top_k="$2"; shift 2 ;;
      --json)     json_output=1; shift ;;
      --debug)    shift ;;
      -*)         ragdag_error "Unknown option: $1"; return 1 ;;
      *)
        if [[ -z "$query" ]]; then
          query="$1"
        fi
        shift
        ;;
    esac
  done

  if [[ -z "$query" ]]; then
    ragdag_error "Usage: ragdag search \"<query>\" [--keyword|--vector|--hybrid] [--domain <name>] [--top <n>]"
    return 1
  fi

  # Load config defaults
  if [[ -z "$mode" ]]; then
    mode="$(ragdag_config_get_from "$config_file" search.default_mode hybrid)"
  fi
  if [[ -z "$top_k" ]]; then
    top_k="$(ragdag_config_get_from "$config_file" search.top_k 10)"
  fi

  case "$mode" in
    keyword)
      _search_keyword "$store_dir" "$query" "$domain" "$top_k" "$json_output"
      ;;
    vector)
      _search_vector "$store_dir" "$query" "$domain" "$top_k" "$json_output" "$config_file"
      ;;
    hybrid)
      # Check if vector search is available
      local embed_provider
      embed_provider="$(ragdag_config_get_from "$config_file" embedding.provider none)"
      if [[ "$embed_provider" == "none" ]] || ! ragdag_has python3; then
        ragdag_debug "No embedding configured — falling back to keyword search"
        _search_keyword "$store_dir" "$query" "$domain" "$top_k" "$json_output"
      else
        _search_hybrid "$store_dir" "$query" "$domain" "$top_k" "$json_output" "$config_file"
      fi
      ;;
    *)
      ragdag_error "Unknown search mode: $mode"
      return 1
      ;;
  esac
}

# Keyword search — pure bash with TF-IDF-like scoring
_search_keyword() {
  local store_dir="$1"
  local query="$2"
  local domain="$3"
  local top_k="$4"
  local json_output="$5"

  # Determine search path
  local search_path
  if [[ -n "$domain" ]]; then
    search_path="${store_dir}/${domain}"
  else
    search_path="$store_dir"
  fi

  if [[ ! -d "$search_path" ]]; then
    if [[ "$json_output" -eq 1 ]]; then
      echo "[]"
    else
      ragdag_warn "No documents found."
    fi
    return 0
  fi

  # Split query into words for multi-term search
  local query_lower
  query_lower="$(echo "$query" | tr '[:upper:]' '[:lower:]')"

  # Find matching files and compute scores
  # Score = sum of (match_count / file_length) for each query term
  local results_file
  results_file="$(ragdag_mktemp_dir)/search_results.tsv"

  # Find all .txt chunk files
  local chunk_files
  chunk_files="$(ragdag_mktemp_dir)/chunk_files.txt"
  find "$search_path" -name '*.txt' -type f ! -name '_*' > "$chunk_files" 2>/dev/null

  while IFS= read -r chunk_file; do
    [[ -f "$chunk_file" ]] || continue

    local content
    content="$(cat "$chunk_file")"
    local content_lower
    content_lower="$(echo "$content" | tr '[:upper:]' '[:lower:]')"
    local content_len=${#content_lower}

    if [[ "$content_len" -eq 0 ]]; then
      continue
    fi

    # Count individual word matches (TF-IDF-like)
    local match_count=0
    local word
    for word in $query_lower; do
      [[ ${#word} -lt 2 ]] && continue
      local wtmp="$content_lower"
      while [[ "$wtmp" == *"$word"* ]]; do
        match_count=$((match_count + 1))
        wtmp="${wtmp#*"$word"}"
      done
    done

    if [[ "$match_count" -gt 0 ]]; then
      # TF-IDF-like score: match_count / sqrt(content_length)
      # Use integer math: score = match_count * 10000 / content_len
      local score=$(( match_count * 10000 / content_len ))
      local rel_path="${chunk_file#"${store_dir}/"}"
      printf '%d\t%s\n' "$score" "$rel_path" >> "$results_file"
    fi
  done < "$chunk_files"

  if [[ ! -f "$results_file" ]] || [[ ! -s "$results_file" ]]; then
    if [[ "$json_output" -eq 1 ]]; then
      echo "[]"
    else
      echo "No results found."
    fi
    return 0
  fi

  # Sort by score descending and take top-K
  local sorted_results
  sorted_results="$(sort -t'	' -k1 -rn "$results_file" | head -n "$top_k")"

  # Format output
  if [[ "$json_output" -eq 1 ]]; then
    _format_results_json "$store_dir" "$sorted_results"
  else
    _format_results_human "$store_dir" "$sorted_results"
  fi
}

# Vector search — delegates to Python
_search_vector() {
  local store_dir="$1"
  local query="$2"
  local domain="$3"
  local top_k="$4"
  local json_output="$5"
  local config_file="$6"

  local search_script="${RAGDAG_DIR}/engines/search_cli.py"

  if [[ ! -f "$search_script" ]]; then
    ragdag_error "Vector search requires Python engine (engines/search_cli.py)"
    return 1
  fi

  local args=(
    --store-dir "$store_dir"
    --query "$query"
    --mode vector
    --top "$top_k"
  )
  [[ -n "$domain" ]] && args+=(--domain "$domain")
  [[ "$json_output" -eq 1 ]] && args+=(--json)

  python3 "$search_script" "${args[@]}"
}

# Hybrid search — keyword pre-filter + vector rerank
_search_hybrid() {
  local store_dir="$1"
  local query="$2"
  local domain="$3"
  local top_k="$4"
  local json_output="$5"
  local config_file="$6"

  local search_script="${RAGDAG_DIR}/engines/search_cli.py"

  if [[ ! -f "$search_script" ]]; then
    ragdag_debug "Python search engine not found — falling back to keyword"
    _search_keyword "$store_dir" "$query" "$domain" "$top_k" "$json_output"
    return
  fi

  local kw_weight vec_weight
  kw_weight="$(ragdag_config_get_from "$config_file" search.keyword_weight 0.3)"
  vec_weight="$(ragdag_config_get_from "$config_file" search.vector_weight 0.7)"

  local args=(
    --store-dir "$store_dir"
    --query "$query"
    --mode hybrid
    --top "$top_k"
    --keyword-weight "$kw_weight"
    --vector-weight "$vec_weight"
  )
  [[ -n "$domain" ]] && args+=(--domain "$domain")
  [[ "$json_output" -eq 1 ]] && args+=(--json)

  python3 "$search_script" "${args[@]}"
}

# Format results as human-readable
_format_results_human() {
  local store_dir="$1"
  local results="$2"

  local rank=1
  while IFS=$'\t' read -r score path; do
    [[ -z "$path" ]] && continue

    local full_path="${store_dir}/${path}"
    local preview=""
    if [[ -f "$full_path" ]]; then
      preview="$(head -2 "$full_path" | tr '\n' ' ' | cut -c1-120)"
    fi

    # Convert integer score to decimal
    local score_dec
    if [[ "$score" -gt 0 ]]; then
      score_dec="0.$(printf '%04d' "$score")"
    else
      score_dec="$score"
    fi

    printf '%d. [%s] %s\n' "$rank" "$score_dec" "$path"
    if [[ -n "$preview" ]]; then
      printf '   %s\n\n' "$preview"
    fi
    rank=$((rank + 1))
  done <<< "$results"
}

# Format results as JSON
_format_results_json() {
  local store_dir="$1"
  local results="$2"

  printf '['
  local first=1
  while IFS=$'\t' read -r score path; do
    [[ -z "$path" ]] && continue

    local full_path="${store_dir}/${path}"
    local content=""
    if [[ -f "$full_path" ]]; then
      content="$(cat "$full_path")"
    fi

    # Determine domain from path
    local domain_name=""
    local parts
    IFS='/' read -ra parts <<< "$path"
    if [[ ${#parts[@]} -ge 3 ]]; then
      domain_name="${parts[0]}"
    fi

    # Escape JSON strings
    content="$(echo "$content" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' 2>/dev/null | sed 's/^"//;s/"$//')"

    if [[ "$first" -eq 0 ]]; then
      printf ','
    fi
    first=0

    local score_dec
    if [[ "$score" =~ ^[0-9]+$ ]]; then
      score_dec="0.${score}"
    else
      score_dec="$score"
    fi

    printf '{"path":"%s","score":%s,"domain":"%s","content":"%s"}' \
      "$path" "$score_dec" "$domain_name" "$content"
  done <<< "$results"
  printf ']\n'
}
