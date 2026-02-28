#!/usr/bin/env bash
# ask.sh — Question answering (RAG) for ragdag

source "${RAGDAG_DIR}/lib/search.sh"

ragdag_ask() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local config_file="${store_dir}/.config"

  # Parse arguments
  local question=""
  local domain=""
  local no_llm=0
  local json_output=0
  local top_k=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --domain)  domain="$2"; shift 2 ;;
      --no-llm)  no_llm=1; shift ;;
      --json)    json_output=1; shift ;;
      --top)     top_k="$2"; shift 2 ;;
      --debug)   shift ;;
      -*)        ragdag_error "Unknown option: $1"; return 1 ;;
      *)
        if [[ -z "$question" ]]; then
          question="$1"
        fi
        shift
        ;;
    esac
  done

  if [[ -z "$question" ]]; then
    ragdag_error "Usage: ragdag ask \"<question>\" [--domain <name>] [--no-llm]"
    return 1
  fi

  if [[ -z "$top_k" ]]; then
    top_k="$(ragdag_config_get_from "$config_file" search.top_k 10)"
  fi

  local max_context
  max_context="$(ragdag_config_get_from "$config_file" llm.max_context 8000)"

  local llm_provider
  llm_provider="$(ragdag_config_get_from "$config_file" llm.provider none)"

  # Step 1: Search for relevant chunks (keyword, since it's pure bash)
  ragdag_debug "Searching for relevant chunks..."

  local results_tmp
  results_tmp="$(ragdag_mktemp_dir)/search_results.txt"

  # Use keyword search to get candidate chunks
  local search_path
  if [[ -n "$domain" ]]; then
    search_path="${store_dir}/${domain}"
  else
    search_path="$store_dir"
  fi

  # Get search results as paths + scores
  local query_lower
  query_lower="$(echo "$question" | tr '[:upper:]' '[:lower:]')"

  # Find matching chunks
  local chunk_files_tmp
  chunk_files_tmp="$(ragdag_mktemp_dir)/chunk_files.txt"
  find "$search_path" -name '*.txt' -type f ! -name '_*' > "$chunk_files_tmp" 2>/dev/null

  while IFS= read -r chunk_file; do
    [[ -f "$chunk_file" ]] || continue
    local content
    content="$(cat "$chunk_file")"
    local content_lower
    content_lower="$(echo "$content" | tr '[:upper:]' '[:lower:]')"
    local content_len=${#content_lower}
    [[ "$content_len" -eq 0 ]] && continue

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
      local score=$(( match_count * 10000 / content_len ))
      local rel_path="${chunk_file#"${store_dir}/"}"
      printf '%d\t%s\n' "$score" "$rel_path" >> "$results_tmp"
    fi
  done < "$chunk_files_tmp"

  if [[ ! -f "$results_tmp" ]] || [[ ! -s "$results_tmp" ]]; then
    if [[ "$json_output" -eq 1 ]]; then
      printf '{"answer":null,"context":"","sources":[],"message":"No relevant documents found."}\n'
    else
      echo "No relevant documents found for your question."
    fi
    return 0
  fi

  local sorted_results
  sorted_results="$(sort -t'	' -k1 -rn "$results_tmp" | head -n "$top_k")"

  # Step 2: Graph expand — check for related_to/references edges
  local expanded_results="$sorted_results"
  local edges_file="${store_dir}/.edges"

  if [[ -f "$edges_file" ]]; then
    while IFS=$'\t' read -r score path; do
      [[ -z "$path" ]] && continue
      # Find related edges
      local related
      related=$(awk -F'\t' -v p="$path" '$1 == p && ($3 == "related_to" || $3 == "references") {print $2}' "$edges_file" 2>/dev/null)
      if [[ -n "$related" ]]; then
        while IFS= read -r rel_path; do
          [[ -z "$rel_path" ]] && continue
          # Check it's not already in results
          if ! echo "$expanded_results" | grep -qF "$rel_path"; then
            # Add with slightly lower score
            local rel_score=$(( score / 2 ))
            expanded_results="${expanded_results}
${rel_score}	${rel_path}"
          fi
        done <<< "$related"
      fi
    done <<< "$sorted_results"
  fi

  # Re-sort after expansion
  expanded_results="$(echo "$expanded_results" | sort -t'	' -k1 -rn)"

  # Step 3: Assemble context with token budget
  local context_file
  context_file="$(ragdag_mktemp_dir)/context.txt"
  local token_budget="$max_context"
  local tokens_used=0

  while IFS=$'\t' read -r score path; do
    [[ -z "$path" ]] && continue
    [[ -z "$score" ]] && continue

    local full_path="${store_dir}/${path}"
    [[ -f "$full_path" ]] || continue

    local chunk_content
    chunk_content="$(cat "$full_path")"
    local chunk_words
    chunk_words=$(echo "$chunk_content" | wc -w | tr -d ' ')
    local chunk_tokens=$(( chunk_words * 13 / 10 ))

    if [[ $(( tokens_used + chunk_tokens )) -gt "$token_budget" ]]; then
      break
    fi

    local score_str
    score_str="$(printf '%04d' "$score" 2>/dev/null || echo "$score")"
    printf -- '--- Source: %s (score: 0.%s) ---\n%s\n\n' "$path" "$score_str" "$chunk_content" >> "$context_file"
    tokens_used=$(( tokens_used + chunk_tokens ))
  done <<< "$expanded_results"

  if [[ ! -f "$context_file" ]] || [[ ! -s "$context_file" ]]; then
    if [[ "$json_output" -eq 1 ]]; then
      printf '{"answer":null,"context":"","sources":[]}\n'
    else
      echo "No context could be assembled."
    fi
    return 0
  fi

  # Step 4: Generate answer
  if [[ "$no_llm" -eq 1 ]] || [[ "$llm_provider" == "none" ]]; then
    # No LLM — show context directly
    if [[ "$json_output" -eq 1 ]]; then
      if ragdag_has python3; then
        python3 -c "
import json, sys
ctx = open('$context_file').read()
sources = [l.split('Source: ')[1].split(' (score:')[0] for l in ctx.splitlines() if l.startswith('--- Source:')]
print(json.dumps({'answer': None, 'context': ctx, 'sources': sources}))
"
      else
        echo '{"answer":null,"context":"(see below)","sources":[]}'
      fi
    else
      echo "=== Context (no LLM configured — set with: ragdag config set llm.provider openai) ==="
      echo ""
      cat "$context_file"
    fi
    return 0
  fi

  # Use Python LLM bridge
  if ! ragdag_has python3; then
    ragdag_error "LLM requires Python 3"
    cat "$context_file"
    return 1
  fi

  local ask_script="${RAGDAG_DIR}/engines/ask_cli.py"
  if [[ ! -f "$ask_script" ]]; then
    ragdag_error "Ask engine not found"
    cat "$context_file"
    return 1
  fi

  local args=(
    --store-dir "$store_dir"
    --question "$question"
    --context-file "$context_file"
  )
  [[ "$no_llm" -eq 1 ]] && args+=(--no-llm)
  [[ "$json_output" -eq 1 ]] && args+=(--json)

  python3 "$ask_script" "${args[@]}"

  # Step 5: Record query (if enabled)
  local record_queries
  record_queries="$(ragdag_config_get_from "$config_file" edges.record_queries false)"
  if [[ "$record_queries" == "true" ]]; then
    local timestamp
    timestamp="$(ragdag_now_iso)"
    local query_id="query_${timestamp}"

    # Record which chunks were retrieved
    while IFS=$'\t' read -r score path; do
      [[ -z "$path" ]] && continue
      printf '%s\t%s\tretrieved\t%s\n' "$query_id" "$path" "$timestamp" >> "$edges_file"
    done <<< "$sorted_results"
  fi
}
