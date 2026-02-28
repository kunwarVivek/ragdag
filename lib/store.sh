#!/usr/bin/env bash
# store.sh — Write chunks to domain directories and update metadata

# Store chunks from a staging directory into the ragdag store
# Usage: ragdag_store <store_dir> <staging_dir> <doc_name> <domain> <source_path> <source_hash>
ragdag_store() {
  local store_dir="$1"
  local staging_dir="$2"
  local doc_name="$3"
  local domain="$4"
  local source_path="$5"
  local source_hash="$6"

  local target_dir
  if [[ -n "$domain" ]] && [[ "$domain" != "flat" ]]; then
    target_dir="${store_dir}/${domain}/${doc_name}"
  else
    target_dir="${store_dir}/${doc_name}"
  fi

  # Atomic move from staging → target
  mkdir -p "$(dirname "$target_dir")"
  if [[ -d "$target_dir" ]]; then
    # Re-ingestion: atomic swap via temp rename
    local tmp_target="${target_dir}.new.$$"
    mv "$staging_dir" "$tmp_target"
    # Remove old chunks (only .txt files) then move new ones in
    local old_chunk
    for old_chunk in "$target_dir"/*.txt; do
      [[ -f "$old_chunk" ]] && rm "$old_chunk"
    done
    mv "$tmp_target"/*.txt "$target_dir"/ 2>/dev/null || true
    rmdir "$tmp_target" 2>/dev/null || true
  else
    mv "$staging_dir" "$target_dir"
  fi

  # Update .processed
  local timestamp
  timestamp="$(ragdag_now_iso)"
  # Remove old entry for this source
  local processed_file="${store_dir}/.processed"
  local tmp_processed
  tmp_processed="$(ragdag_mktemp_dir)/processed.tmp"
  awk -F'\t' -v path="$source_path" '$1 != path' "$processed_file" > "$tmp_processed" 2>/dev/null || cp "$processed_file" "$tmp_processed"
  printf '%s\t%s\t%s\t%s\n' "$source_path" "$source_hash" "$domain" "$timestamp" >> "$tmp_processed"
  mv "$tmp_processed" "$processed_file"

  # Return the target directory path (relative to store)
  local rel_path="${target_dir#"${store_dir}/"}"
  echo "$rel_path"
}

# Create chunked_from edges for all chunks in a directory
# Usage: ragdag_store_edges <store_dir> <doc_rel_path> <source_path>
ragdag_store_edges() {
  local store_dir="$1"
  local doc_rel_path="$2"
  local source_path="$3"
  local edges_file="${store_dir}/.edges"

  # Remove old edges for this doc
  local tmp_edges
  tmp_edges="$(ragdag_mktemp_dir)/edges.tmp"
  awk -F'\t' -v src="$source_path" '!($2 == src && $3 == "chunked_from")' "$edges_file" > "$tmp_edges" 2>/dev/null || cp "$edges_file" "$tmp_edges"

  # Add new edges
  local chunk
  for chunk in "${store_dir}/${doc_rel_path}"/*.txt; do
    [[ -f "$chunk" ]] || continue
    local chunk_rel="${chunk#"${store_dir}/"}"
    printf '%s\t%s\tchunked_from\t\n' "$chunk_rel" "$source_path" >> "$tmp_edges"
  done

  mv "$tmp_edges" "$edges_file"
}

# Check if a file has already been processed with the same hash
# Usage: ragdag_is_processed <store_dir> <source_path> <source_hash>
# Returns 0 if already processed, 1 if new/changed
ragdag_is_processed() {
  local store_dir="$1"
  local source_path="$2"
  local source_hash="$3"
  local processed_file="${store_dir}/.processed"

  if [[ ! -f "$processed_file" ]]; then
    return 1
  fi

  awk -F'\t' -v path="$source_path" -v hash="$source_hash" 'BEGIN{found=1} $1 == path && $2 == hash {found=0; exit} END{exit found}' "$processed_file" 2>/dev/null
}

# Get the domain for a previously processed file
ragdag_get_processed_domain() {
  local store_dir="$1"
  local source_path="$2"
  local processed_file="${store_dir}/.processed"

  if [[ -f "$processed_file" ]]; then
    awk -F'\t' -v path="$source_path" '$1 == path {domain=$3} END{if(domain!="") print domain}' "$processed_file" 2>/dev/null
  fi
}

# Apply domain rules to determine domain for a file
# Usage: ragdag_apply_domain_rules <store_dir> <source_path>
# Outputs domain name or empty string
ragdag_apply_domain_rules() {
  local store_dir="$1"
  local source_path="$2"
  local rules_file="${store_dir}/.domain-rules"
  local source_lower
  source_lower="$(echo "$source_path" | tr '[:upper:]' '[:lower:]')"

  [[ -f "$rules_file" ]] || return

  while IFS= read -r rule || [[ -n "$rule" ]]; do
    # Skip comments and empty lines
    case "$rule" in
      '#'*|'') continue ;;
    esac

    # Format: patterns → domain
    # Split on →
    local patterns domain
    patterns="$(echo "$rule" | sed 's/→.*//' | sed 's/[[:space:]]*$//')"
    domain="$(echo "$rule" | sed 's/.*→//' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')"

    if [[ -z "$domain" ]]; then
      continue
    fi

    # Check each pattern
    local pattern match=0
    for pattern in $patterns; do
      # Simple glob matching
      case "$source_lower" in
        *${pattern}*) match=1 ;;
      esac
    done

    if [[ "$match" -eq 1 ]]; then
      echo "$domain"
      return
    fi
  done < "$rules_file"
}
