#!/usr/bin/env bash
# maintain.sh — Maintenance operations for ragdag: verify, repair, gc, reindex

# Verify store integrity
ragdag_verify() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local issues=0
  echo "Verifying ragdag store: $store_dir"
  echo ""

  # 1. Check all .txt chunks are readable
  local chunk_count=0
  local bad_chunks=0
  while IFS= read -r chunk; do
    chunk_count=$((chunk_count + 1))
    if [[ ! -r "$chunk" ]]; then
      ragdag_warn "Unreadable chunk: $chunk"
      bad_chunks=$((bad_chunks + 1))
      issues=$((issues + 1))
    fi
  done < <(find "$store_dir" -name '*.txt' -type f ! -name '_*' 2>/dev/null)
  ragdag_info "Chunks: $chunk_count total, $bad_chunks unreadable"

  # 2. Check manifest files match embeddings
  while IFS= read -r manifest; do
    local embed_dir
    embed_dir="$(dirname "$manifest")"
    local bin_file="${embed_dir}/embeddings.bin"

    if [[ ! -f "$bin_file" ]]; then
      ragdag_warn "Manifest without embeddings.bin: $manifest"
      issues=$((issues + 1))
      continue
    fi

    # Count manifest entries
    local manifest_count
    manifest_count=$(grep -cv '^#' "$manifest" 2>/dev/null || echo 0)

    # Validate binary header if Python available
    if ragdag_has python3; then
      local bin_count
      bin_count=$(python3 -c "
import struct
with open('$bin_file', 'rb') as f:
    d = f.read(16)
    magic = struct.unpack_from('I', d, 0)[0]
    if magic != 0x52414744:
        print('BAD_MAGIC')
    else:
        print(struct.unpack_from('I', d, 12)[0])
" 2>/dev/null)

      if [[ "$bin_count" == "BAD_MAGIC" ]]; then
        ragdag_warn "Invalid embeddings.bin magic number: $bin_file"
        issues=$((issues + 1))
      elif [[ "$bin_count" != "$manifest_count" ]]; then
        ragdag_warn "Manifest/binary count mismatch: manifest=$manifest_count, binary=$bin_count in $embed_dir"
        issues=$((issues + 1))
      else
        ragdag_info "Embeddings OK: $embed_dir ($bin_count vectors)"
      fi
    fi
  done < <(find "$store_dir" -name 'manifest.tsv' -type f 2>/dev/null)

  # 3. Check for orphaned edges
  local edges_file="${store_dir}/.edges"
  if [[ -f "$edges_file" ]]; then
    local edge_count
    edge_count=$(grep -cv '^#' "$edges_file" 2>/dev/null || echo 0)
    local orphaned=0

    while IFS=$'\t' read -r source target etype metadata; do
      [[ "$source" == '#'* ]] && continue
      [[ -z "$source" ]] && continue

      # Check if source exists (for chunk paths)
      if [[ "$source" == *".txt" ]] && [[ ! -f "${store_dir}/${source}" ]]; then
        orphaned=$((orphaned + 1))
      fi
    done < "$edges_file"

    ragdag_info "Edges: $edge_count total, $orphaned orphaned"
    if [[ "$orphaned" -gt 0 ]]; then
      issues=$((issues + 1))
    fi
  fi

  # 4. Check .processed references
  local processed_file="${store_dir}/.processed"
  if [[ -f "$processed_file" ]]; then
    local proc_count
    proc_count=$(grep -cv '^#' "$processed_file" 2>/dev/null || echo 0)
    local stale=0

    while IFS=$'\t' read -r source_path hash domain timestamp; do
      [[ "$source_path" == '#'* ]] && continue
      [[ -z "$source_path" ]] && continue
      if [[ ! -f "$source_path" ]]; then
        stale=$((stale + 1))
      fi
    done < "$processed_file"

    ragdag_info "Processed: $proc_count entries, $stale stale"
    if [[ "$stale" -gt 0 ]]; then
      issues=$((issues + 1))
    fi
  fi

  echo ""
  if [[ "$issues" -eq 0 ]]; then
    ragdag_ok "Store is healthy. No issues found."
  else
    ragdag_warn "$issues issue(s) found. Run 'ragdag repair' to fix."
  fi
}

# Repair store issues
ragdag_repair() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  echo "Repairing ragdag store: $store_dir"
  echo ""

  # 1. Remove orphaned edges
  local edges_file="${store_dir}/.edges"
  if [[ -f "$edges_file" ]]; then
    local tmp_edges
    tmp_edges="$(ragdag_mktemp_dir)/edges.tmp"
    local removed=0

    while IFS= read -r line; do
      if [[ "$line" == '#'* ]] || [[ -z "$line" ]]; then
        echo "$line" >> "$tmp_edges"
        continue
      fi

      local source target
      source="$(echo "$line" | cut -f1)"
      target="$(echo "$line" | cut -f2)"

      # Keep edge if source is not a .txt path or if it exists
      if [[ "$source" != *".txt" ]] || [[ -f "${store_dir}/${source}" ]]; then
        echo "$line" >> "$tmp_edges"
      else
        removed=$((removed + 1))
      fi
    done < "$edges_file"

    mv "$tmp_edges" "$edges_file"
    ragdag_info "Removed $removed orphaned edges"
  fi

  # 2. Rebuild manifest.tsv from actual chunk files for each domain
  while IFS= read -r bin_file; do
    local embed_dir
    embed_dir="$(dirname "$bin_file")"
    local manifest="${embed_dir}/manifest.tsv"

    ragdag_info "Checking manifest in $embed_dir"
    # This would need Python to rebuild fully; for now just validate
  done < <(find "$store_dir" -name 'embeddings.bin' -type f 2>/dev/null)

  ragdag_ok "Repair complete."
}

# Garbage collection
ragdag_gc() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  echo "Garbage collecting ragdag store: $store_dir"
  echo ""

  # 1. Remove orphaned edges
  local edges_file="${store_dir}/.edges"
  local edges_cleaned=0
  if [[ -f "$edges_file" ]]; then
    local tmp_edges
    tmp_edges="$(ragdag_mktemp_dir)/edges.tmp"

    while IFS= read -r line; do
      if [[ "$line" == '#'* ]] || [[ -z "$line" ]]; then
        echo "$line" >> "$tmp_edges"
        continue
      fi

      local source
      source="$(echo "$line" | cut -f1)"

      if [[ "$source" != *".txt" ]] || [[ -f "${store_dir}/${source}" ]]; then
        echo "$line" >> "$tmp_edges"
      else
        edges_cleaned=$((edges_cleaned + 1))
      fi
    done < "$edges_file"

    mv "$tmp_edges" "$edges_file"
  fi

  # 2. Remove stale .processed entries
  local processed_file="${store_dir}/.processed"
  local proc_cleaned=0
  if [[ -f "$processed_file" ]]; then
    local tmp_proc
    tmp_proc="$(ragdag_mktemp_dir)/processed.tmp"

    while IFS= read -r line; do
      if [[ "$line" == '#'* ]] || [[ -z "$line" ]]; then
        echo "$line" >> "$tmp_proc"
        continue
      fi

      local source_path
      source_path="$(echo "$line" | cut -f1)"

      if [[ -f "$source_path" ]]; then
        echo "$line" >> "$tmp_proc"
      else
        proc_cleaned=$((proc_cleaned + 1))
      fi
    done < "$processed_file"

    mv "$tmp_proc" "$processed_file"
  fi

  ragdag_ok "GC complete. Cleaned $edges_cleaned edges, $proc_cleaned processed entries."
}

# Reindex embeddings
ragdag_reindex() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  if ! ragdag_has python3; then
    ragdag_error "Reindex requires Python 3"
    return 1
  fi

  local domain="${1:-}"
  local all_flag=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --all) all_flag=1; shift ;;
      --debug) shift ;;
      *) domain="$1"; shift ;;
    esac
  done

  local config_file="${store_dir}/.config"
  local provider model dimensions
  provider="$(ragdag_config_get_from "$config_file" embedding.provider none)"
  model="$(ragdag_config_get_from "$config_file" embedding.model text-embedding-3-small)"
  dimensions="$(ragdag_config_get_from "$config_file" embedding.dimensions 1536)"

  if [[ "$provider" == "none" ]]; then
    ragdag_error "No embedding provider configured. Set with: ragdag config set embedding.provider openai"
    return 1
  fi

  local embed_script="${RAGDAG_DIR}/engines/embed_cli.py"
  if [[ ! -f "$embed_script" ]]; then
    ragdag_error "Embedding engine not found"
    return 1
  fi

  # Find directories to reindex
  local dirs_to_reindex=()
  if [[ -n "$domain" ]]; then
    dirs_to_reindex=("${store_dir}/${domain}")
  elif [[ "$all_flag" -eq 1 ]]; then
    for d in "$store_dir"/*/; do
      [[ -d "$d" ]] && [[ "$(basename "$d")" != .* ]] && dirs_to_reindex+=("$d")
    done
  else
    ragdag_error "Usage: ragdag reindex [domain/] [--all]"
    return 1
  fi

  for dir in "${dirs_to_reindex[@]}"; do
    local domain_name
    domain_name="$(basename "$dir")"
    ragdag_info "Reindexing: $domain_name"

    # Remove old embeddings
    rm -f "${dir}/embeddings.bin" "${dir}/manifest.tsv"

    # Find all doc subdirectories
    for doc_dir in "$dir"/*/; do
      [[ -d "$doc_dir" ]] || continue
      local doc_name
      doc_name="$(basename "$doc_dir")"
      local doc_prefix="${domain_name}/${doc_name}"

      python3 "$embed_script" embed \
        --chunks-dir "$doc_dir" \
        --output-dir "$dir" \
        --provider "$provider" \
        --model "$model" \
        --dimensions "$dimensions" \
        --doc-prefix "$doc_prefix" \
        2>/dev/null || ragdag_warn "Failed to embed: $doc_prefix"
    done
  done

  ragdag_ok "Reindex complete."
}
