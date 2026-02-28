#!/usr/bin/env bash
# config.sh — INI config parsing and writing for ragdag
# Compatible with bash 3.2+ (no associative arrays)

# Generate default config file content
ragdag_default_config() {
  cat <<'CONF'
[general]
chunk_strategy = heading
chunk_size = 1000
chunk_overlap = 100

[embedding]
provider = none
model = text-embedding-3-small
dimensions = 1536

[llm]
provider = none
model = gpt-4o-mini
max_context = 8000

[search]
default_mode = hybrid
top_k = 10
keyword_weight = 0.3
vector_weight = 0.7

[edges]
auto_relate = false
relate_threshold = 0.8
record_queries = false
CONF
}

# Get a config value from an INI file
# Usage: ragdag_config_get_from <config_file> <section.key> [default]
ragdag_config_get_from() {
  local config_file="$1"
  local full_key="$2"
  local default="${3:-}"

  local section key
  section="${full_key%%.*}"
  key="${full_key#*.}"

  if [[ ! -f "$config_file" ]]; then
    echo "$default"
    return
  fi

  local in_section=0
  local value=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip comments and empty lines
    case "$line" in
      '#'*|';'*|'') continue ;;
    esac

    # Section header
    if [[ "$line" =~ ^\[([a-zA-Z0-9_-]+)\] ]]; then
      if [[ "${BASH_REMATCH[1]}" == "$section" ]]; then
        in_section=1
      else
        # If we were in the right section and leaving, stop
        if [[ "$in_section" -eq 1 ]]; then
          break
        fi
        in_section=0
      fi
      continue
    fi

    # Key = value (in correct section)
    if [[ "$in_section" -eq 1 ]] && [[ "$line" =~ ^[[:space:]]*([a-zA-Z0-9_]+)[[:space:]]*=[[:space:]]*(.*) ]]; then
      local found_key="${BASH_REMATCH[1]}"
      local found_val="${BASH_REMATCH[2]}"
      # Trim trailing whitespace
      found_val="${found_val%"${found_val##*[![:space:]]}"}"
      if [[ "$found_key" == "$key" ]]; then
        value="$found_val"
      fi
    fi
  done < "$config_file"

  if [[ -n "$value" ]]; then
    echo "$value"
  else
    echo "$default"
  fi
}

# Set a config value in the config file
# Usage: ragdag_config_set_in <config_file> <section.key> <value>
ragdag_config_set_in() {
  local config_file="$1"
  local full_key="$2"
  local value="$3"

  local section key
  if [[ "$full_key" == *.* ]]; then
    section="${full_key%%.*}"
    key="${full_key#*.}"
  else
    ragdag_error "Config key must be in section.key format (e.g., embedding.model)"
    return 1
  fi

  if [[ ! -f "$config_file" ]]; then
    ragdag_default_config > "$config_file"
  fi

  local in_section=0
  local found=0
  local tmpfile
  tmpfile="$(ragdag_mktemp_dir)/config.tmp"

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Check section headers
    if [[ "$line" =~ ^\[([a-zA-Z0-9_-]+)\] ]]; then
      # If leaving the right section without finding the key, insert
      if [[ "$in_section" -eq 1 ]] && [[ "$found" -eq 0 ]]; then
        echo "${key} = ${value}" >> "$tmpfile"
        found=1
      fi
      if [[ "${BASH_REMATCH[1]}" == "$section" ]]; then
        in_section=1
      else
        in_section=0
      fi
      echo "$line" >> "$tmpfile"
      continue
    fi

    # Replace existing key in correct section
    if [[ "$in_section" -eq 1 ]] && [[ "$line" =~ ^[[:space:]]*${key}[[:space:]]*= ]]; then
      echo "${key} = ${value}" >> "$tmpfile"
      found=1
      continue
    fi

    echo "$line" >> "$tmpfile"
  done < "$config_file"

  # End of file: if we were in the right section but didn't find the key
  if [[ "$in_section" -eq 1 ]] && [[ "$found" -eq 0 ]]; then
    echo "${key} = ${value}" >> "$tmpfile"
    found=1
  fi

  # Section doesn't exist at all
  if [[ "$found" -eq 0 ]]; then
    echo "" >> "$tmpfile"
    echo "[${section}]" >> "$tmpfile"
    echo "${key} = ${value}" >> "$tmpfile"
  fi

  mv "$tmpfile" "$config_file"
}

# Show all config
ragdag_config_show() {
  local config_file="$1"
  if [[ ! -f "$config_file" ]]; then
    ragdag_error "No config file found at: $config_file"
    return 1
  fi
  cat "$config_file"
}

# CLI handler for `ragdag config`
ragdag_config() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  local config_file="${store_dir}/.config"

  case "${1:-}" in
    get)
      if [[ -z "${2:-}" ]]; then
        ragdag_error "Usage: ragdag config get <section.key>"
        return 1
      fi
      local val
      val="$(ragdag_config_get_from "$config_file" "$2")"
      if [[ -n "$val" ]]; then
        echo "$val"
      else
        ragdag_error "Unknown config key: $2"
        return 1
      fi
      ;;
    set)
      if [[ -z "${2:-}" ]] || [[ -z "${3:-}" ]]; then
        ragdag_error "Usage: ragdag config set <section.key> <value>"
        return 1
      fi
      ragdag_config_set_in "$config_file" "$2" "$3"
      ragdag_ok "Set $2 = $3"
      ;;
    *)
      ragdag_config_show "$config_file"
      ;;
  esac
}
