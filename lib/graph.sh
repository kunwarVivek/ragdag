#!/usr/bin/env bash
# graph.sh — Knowledge graph operations for ragdag

# Graph summary: count domains, documents, chunks, edges, edge types
ragdag_graph() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local domain_filter="${1:-}"
  local search_path
  if [[ -n "$domain_filter" ]]; then
    search_path="${store_dir}/${domain_filter}"
  else
    search_path="$store_dir"
  fi

  # Count domains (top-level directories that contain chunks)
  local domains=0
  local documents=0
  local chunks=0

  local dir
  for dir in "$store_dir"/*/; do
    [[ -d "$dir" ]] || continue
    local dirname
    dirname="$(basename "$dir")"
    # Skip hidden directories
    [[ "$dirname" == .* ]] && continue

    domains=$((domains + 1))

    # Count documents (subdirectories of domain)
    local doc_dir
    for doc_dir in "$dir"*/; do
      [[ -d "$doc_dir" ]] || continue
      documents=$((documents + 1))

      # Count chunks
      local chunk
      for chunk in "$doc_dir"*.txt; do
        [[ -f "$chunk" ]] && chunks=$((chunks + 1))
      done
    done
  done

  # Count edges and edge types
  local edges_file="${store_dir}/.edges"
  local total_edges=0
  local edge_types=""

  if [[ -f "$edges_file" ]]; then
    total_edges=$(grep -cv '^#' "$edges_file" 2>/dev/null || echo 0)
    edge_types=$(grep -v '^#' "$edges_file" 2>/dev/null | cut -f3 | sort | uniq -c | sort -rn)
  fi

  echo "ragdag Knowledge Graph"
  echo "====================="
  echo ""
  echo "Domains:    $domains"
  echo "Documents:  $documents"
  echo "Chunks:     $chunks"
  echo "Edges:      $total_edges"

  if [[ -n "$edge_types" ]]; then
    echo ""
    echo "Edge types:"
    echo "$edge_types" | while read -r count etype; do
      printf "  %-20s %s\n" "$etype" "$count"
    done
  fi
}

# Show connected nodes for a given node
ragdag_neighbors() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local node="${1:-}"
  if [[ -z "$node" ]]; then
    ragdag_error "Usage: ragdag neighbors <node-path>"
    return 1
  fi

  local edges_file="${store_dir}/.edges"
  if [[ ! -f "$edges_file" ]]; then
    echo "No edges found."
    return 0
  fi

  # Find edges where node is source or target
  local found=0
  echo "Neighbors of: $node"
  echo ""

  # Node as source
  awk -F'\t' -v n="$node" '$1 == n {print}' "$edges_file" 2>/dev/null | while IFS=$'\t' read -r source target etype metadata; do
    printf "  → %-30s [%s]" "$target" "$etype"
    [[ -n "$metadata" ]] && printf "  %s" "$metadata"
    printf '\n'
  done

  # Node as target
  awk -F'\t' -v n="$node" '$2 == n {print}' "$edges_file" 2>/dev/null | while IFS=$'\t' read -r source target etype metadata; do
    printf "  ← %-30s [%s]" "$source" "$etype"
    [[ -n "$metadata" ]] && printf "  %s" "$metadata"
    printf '\n'
  done
}

# Trace provenance: walk backward through chunked_from/derived_via edges
ragdag_trace() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local node="${1:-}"
  if [[ -z "$node" ]]; then
    ragdag_error "Usage: ragdag trace <node-path>"
    return 1
  fi

  local edges_file="${store_dir}/.edges"
  if [[ ! -f "$edges_file" ]]; then
    echo "No edges found."
    return 0
  fi

  echo "Provenance trace: $node"
  echo ""

  local current="$node"
  local depth=0
  local max_depth=20
  local visited=" "

  while [[ "$depth" -lt "$max_depth" ]]; do
    # Cycle detection
    if [[ "$visited" == *" ${current} "* ]]; then
      printf '%*s└── %s (origin)\n' "$((depth * 4))" "" "$current"
      break
    fi
    visited="${visited}${current} "

    # Find chunked_from or derived_via edge where current is source
    local parent
    parent=$(awk -F'\t' -v c="$current" '$1 == c && ($3 == "chunked_from" || $3 == "derived_via") {print $2; exit}' "$edges_file" 2>/dev/null)

    if [[ -z "$parent" ]]; then
      printf '%*s└── %s (origin)\n' "$((depth * 4))" "" "$current"
      break
    fi

    local etype
    etype=$(awk -F'\t' -v s="$current" -v t="$parent" '$1 == s && $2 == t {print $3; exit}' "$edges_file" 2>/dev/null)
    [[ -z "$etype" ]] && etype="chunked_from"

    printf '%*s├── %s [%s]\n' "$((depth * 4))" "" "$current" "$etype"
    current="$parent"
    depth=$((depth + 1))
  done
}

# Compute semantic edges between chunks (requires Python)
ragdag_relate() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local domain="${1:-}"
  local threshold=""

  # Parse args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --threshold) threshold="$2"; shift 2 ;;
      --debug) shift ;;
      *) domain="$1"; shift ;;
    esac
  done

  if [[ -z "$threshold" ]]; then
    local config_file="${store_dir}/.config"
    threshold="$(ragdag_config_get_from "$config_file" edges.relate_threshold 0.8)"
  fi

  if ! ragdag_has python3; then
    ragdag_error "ragdag relate requires Python 3"
    return 1
  fi

  local relate_script="${RAGDAG_DIR}/engines/relate_cli.py"
  if [[ ! -f "$relate_script" ]]; then
    ragdag_error "Relate engine not found"
    return 1
  fi

  local args=(
    --store-dir "$store_dir"
    --threshold "$threshold"
  )
  [[ -n "$domain" ]] && args+=(--domain "$domain")

  python3 "$relate_script" "${args[@]}"
}

# Create a manual edge
ragdag_link() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local source="${1:-}"
  local target="${2:-}"
  local edge_type="${3:-references}"

  if [[ -z "$source" ]] || [[ -z "$target" ]]; then
    ragdag_error "Usage: ragdag link <source> <target> [edge_type]"
    return 1
  fi

  local edges_file="${store_dir}/.edges"
  printf '%s\t%s\t%s\t\n' "$source" "$target" "$edge_type" >> "$edges_file"
  ragdag_ok "Added edge: $source → $target [$edge_type]"
}
