#!/usr/bin/env bash
# init.sh — Initialize a ragdag store

ragdag_init() {
  local target="${1:-.}"
  local store_dir

  # Resolve target path
  target="$(ragdag_realpath "$target")"
  store_dir="${target}/.ragdag"

  # Check if already initialized
  if [[ -d "$store_dir" ]]; then
    ragdag_warn "Already initialized at: $store_dir"
    return 0
  fi

  # Create store structure
  mkdir -p "$store_dir"

  # Create default config
  ragdag_default_config > "${store_dir}/.config"

  # Create empty metadata files
  printf '# source_path\tcontent_hash\tdomain\ttimestamp\n' > "${store_dir}/.processed"
  printf '# source\ttarget\tedge_type\tmetadata\n' > "${store_dir}/.edges"
  printf '# pattern → domain\n' > "${store_dir}/.domain-rules"

  # Add .ragdag to .gitignore if in a git repo
  if [[ -d "${target}/.git" ]]; then
    local gitignore="${target}/.gitignore"
    if [[ -f "$gitignore" ]]; then
      if ! grep -qF '.ragdag/' "$gitignore" 2>/dev/null; then
        echo '.ragdag/' >> "$gitignore"
        ragdag_info "Added .ragdag/ to .gitignore"
      fi
    else
      echo '.ragdag/' > "$gitignore"
      ragdag_info "Created .gitignore with .ragdag/"
    fi
  fi

  # Dependency check
  ragdag_info "Checking dependencies..."
  _ragdag_check_deps

  ragdag_ok "Initialized ragdag store at: $store_dir"
}

_ragdag_check_deps() {
  local dep status

  # Required
  for dep in bash grep awk sort cut; do
    if ragdag_has "$dep"; then
      ragdag_color green "  + $dep ($(command -v "$dep"))"
    else
      ragdag_color red   "  - $dep (REQUIRED — not found)"
    fi
  done

  # Recommended
  if ragdag_has rg; then
    ragdag_color green "  + ripgrep ($(rg --version | head -1))"
  else
    ragdag_color yellow "  ~ ripgrep (recommended for faster search)"
  fi

  if ragdag_has jq; then
    ragdag_color green "  + jq ($(jq --version 2>&1))"
  else
    ragdag_color yellow "  ~ jq (recommended for JSON output)"
  fi

  # Optional - Python
  if ragdag_has python3; then
    ragdag_color green "  + python3 ($(python3 --version 2>&1))"
  else
    ragdag_color yellow "  ~ python3 (needed for embeddings, vector search, SDK)"
  fi

  # Optional - document formats
  if ragdag_has pdftotext; then
    ragdag_color green "  + pdftotext"
  else
    ragdag_color dim "  ~ pdftotext (PDF support disabled)"
  fi

  if ragdag_has pandoc; then
    ragdag_color green "  + pandoc ($(pandoc --version | head -1))"
  else
    ragdag_color dim "  ~ pandoc (DOCX/HTML support disabled)"
  fi
}
