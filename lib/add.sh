#!/usr/bin/env bash
# add.sh — Ingest orchestration for ragdag
# Flow: parse → chunk → store → embed → edges

source "${RAGDAG_DIR}/lib/parse.sh"
source "${RAGDAG_DIR}/lib/chunk.sh"
source "${RAGDAG_DIR}/lib/store.sh"

ragdag_add() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local config_file="${store_dir}/.config"

  # Parse arguments
  local paths=()
  local domain=""
  local flat=0
  local no_embed=0
  local do_relate=0
  local batch=0
  local json_output=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --domain)   domain="${2:-}"; shift 2 ;;
      --flat)     flat=1; shift ;;
      --no-embed) no_embed=1; shift ;;
      --relate)   do_relate=1; shift ;;
      --batch)    batch=1; shift ;;
      --json)     json_output=1; shift ;;
      --debug)    shift ;;  # handled globally
      -*)         ragdag_error "Unknown option: $1"; return 1 ;;
      *)          paths+=("$1"); shift ;;
    esac
  done

  if [[ ${#paths[@]} -eq 0 ]]; then
    ragdag_error "Usage: ragdag add <path> [--domain <name|auto>] [--flat] [--no-embed]"
    return 1
  fi

  # Load config
  local chunk_strategy chunk_size chunk_overlap
  chunk_strategy="$(ragdag_config_get_from "$config_file" general.chunk_strategy heading)"
  chunk_size="$(ragdag_config_get_from "$config_file" general.chunk_size 1000)"
  chunk_overlap="$(ragdag_config_get_from "$config_file" general.chunk_overlap 100)"

  local embed_provider
  embed_provider="$(ragdag_config_get_from "$config_file" embedding.provider none)"

  local total_chunks=0
  local total_files=0
  local total_skipped=0

  # Collect all files to process
  local files_to_process=()
  local path
  for path in "${paths[@]}"; do
    if [[ -d "$path" ]]; then
      # Recursively find files
      while IFS= read -r f; do
        files_to_process+=("$f")
      done < <(find "$path" -type f ! -name '.*' ! -path '*/.ragdag/*' ! -path '*/.git/*' | sort)
    elif [[ -f "$path" ]]; then
      files_to_process+=("$path")
    else
      ragdag_warn "Path not found: $path"
    fi
  done

  if [[ ${#files_to_process[@]} -eq 0 ]]; then
    ragdag_warn "No files found to process."
    return 0
  fi

  local file
  for file in "${files_to_process[@]}"; do
    local abs_path
    abs_path="$(ragdag_realpath "$file")"

    # Compute content hash
    local content_hash
    content_hash="$(ragdag_sha256 "$abs_path")"

    # Dedup check
    if ragdag_is_processed "$store_dir" "$abs_path" "$content_hash"; then
      ragdag_debug "Skipping (unchanged): $file"
      total_skipped=$((total_skipped + 1))
      continue
    fi

    # Detect file type
    local ftype
    ftype="$(ragdag_detect_type "$file")"
    if [[ "$ftype" == "unknown" ]]; then
      ragdag_warn "Skipping unsupported file type: $file"
      continue
    fi

    # Determine document name
    local doc_name
    doc_name="$(ragdag_sanitize "$(basename "$file" | sed 's/\.[^.]*$//')")"
    if [[ -z "$doc_name" ]]; then
      doc_name="document"
    fi

    # Determine domain
    local file_domain="$domain"
    if [[ "$flat" -eq 1 ]]; then
      file_domain=""
    elif [[ "$file_domain" == "auto" ]]; then
      file_domain="$(ragdag_apply_domain_rules "$store_dir" "$file")"
      if [[ -z "$file_domain" ]]; then
        if [[ "$batch" -eq 1 ]]; then
          file_domain="unsorted"
        else
          file_domain="unsorted"
        fi
      fi
    fi

    ragdag_debug "Processing: $file (type=$ftype, domain=$file_domain)"

    # Create staging directory
    local staging_dir
    staging_dir="$(ragdag_mktemp_dir)/${doc_name}"
    mkdir -p "$staging_dir"

    # Parse → plain text
    local parsed_file="${staging_dir}/_parsed.txt"
    if ! ragdag_parse "$file" "$ftype" > "$parsed_file" 2>/dev/null; then
      ragdag_warn "Parse failed for: $file — storing as single chunk"
      cp "$file" "${staging_dir}/01.txt"
      rm -f "$parsed_file"
    else
      # Choose chunk strategy based on file type
      local effective_strategy="$chunk_strategy"
      case "$ftype" in
        markdown) effective_strategy="heading" ;;
        code)     effective_strategy="function" ;;
      esac

      # Chunk
      local num_chunks
      num_chunks="$(ragdag_chunk "$parsed_file" "$staging_dir" "$effective_strategy" "$chunk_size" "$chunk_overlap")"
      rm -f "$parsed_file"

      ragdag_debug "Created $num_chunks chunks for: $file"
    fi

    # Store
    local doc_rel_path
    doc_rel_path="$(ragdag_store "$store_dir" "$staging_dir" "$doc_name" "$file_domain" "$abs_path" "$content_hash")"

    # Create edges
    ragdag_store_edges "$store_dir" "$doc_rel_path" "$abs_path"

    # Embed (if Python available and embedding configured)
    if [[ "$no_embed" -eq 0 ]] && [[ "$embed_provider" != "none" ]]; then
      if ragdag_has python3; then
        _ragdag_embed_chunks "$store_dir" "$doc_rel_path" "$file_domain"
      else
        ragdag_debug "Skipping embedding (python3 not available)"
      fi
    fi

    # Count chunks
    local chunk_count=0
    for f in "${store_dir}/${doc_rel_path}"/*.txt; do
      [[ -f "$f" ]] && chunk_count=$((chunk_count + 1))
    done
    total_chunks=$((total_chunks + chunk_count))
    total_files=$((total_files + 1))

    if [[ "$json_output" -eq 0 ]]; then
      ragdag_info "Added: $file → $doc_rel_path ($chunk_count chunks)"
    fi
  done

  # Summary
  if [[ "$json_output" -eq 1 ]]; then
    printf '{"files":%d,"chunks":%d,"skipped":%d}\n' "$total_files" "$total_chunks" "$total_skipped"
  else
    echo ""
    ragdag_ok "Ingested $total_files file(s) → $total_chunks chunk(s). Skipped $total_skipped unchanged."
  fi
}

# Embed chunks via Python bridge
_ragdag_embed_chunks() {
  local store_dir="$1"
  local doc_rel_path="$2"
  local domain="$3"

  local chunk_dir="${store_dir}/${doc_rel_path}"
  local embed_script="${RAGDAG_DIR}/engines/embed_cli.py"

  if [[ ! -f "$embed_script" ]]; then
    ragdag_debug "Embedding script not found, skipping"
    return 0
  fi

  local config_file="${store_dir}/.config"
  local provider model dimensions
  provider="$(ragdag_config_get_from "$config_file" embedding.provider none)"
  model="$(ragdag_config_get_from "$config_file" embedding.model text-embedding-3-small)"
  dimensions="$(ragdag_config_get_from "$config_file" embedding.dimensions 1536)"

  # Determine embeddings.bin location (per-domain)
  local embed_dir
  if [[ -n "$domain" ]]; then
    embed_dir="${store_dir}/${domain}"
  else
    embed_dir="$store_dir"
  fi

  python3 "$embed_script" embed \
    --chunks-dir "$chunk_dir" \
    --output-dir "$embed_dir" \
    --provider "$provider" \
    --model "$model" \
    --dimensions "$dimensions" \
    --doc-prefix "$doc_rel_path" \
    2>/dev/null || ragdag_warn "Embedding failed for $doc_rel_path"
}
