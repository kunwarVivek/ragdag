#!/usr/bin/env bash
# compat.sh — OS detection and portable wrappers for ragdag
# Works with bash 3.2+ (macOS default)

# Detect OS
RAGDAG_OS="unknown"
case "$(uname -s)" in
  Darwin*) RAGDAG_OS="macos" ;;
  Linux*)  RAGDAG_OS="linux" ;;
  *)       RAGDAG_OS="unknown" ;;
esac

# Portable sed -i (macOS requires '')
ragdag_sed_i() {
  if [[ "$RAGDAG_OS" == "macos" ]]; then
    sed -i '' "$@"
  else
    sed -i "$@"
  fi
}

# Portable stat for file size in bytes
ragdag_file_size() {
  if [[ "$RAGDAG_OS" == "macos" ]]; then
    stat -f '%z' "$1"
  else
    stat -c '%s' "$1"
  fi
}

# Portable stat for modification time (epoch)
ragdag_mtime() {
  if [[ "$RAGDAG_OS" == "macos" ]]; then
    stat -f '%m' "$1"
  else
    stat -c '%Y' "$1"
  fi
}

# Portable sha256 hash
ragdag_sha256() {
  if command -v sha256sum &>/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum &>/dev/null; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    openssl sha256 "$1" | awk '{print $NF}'
  fi
}

# Portable grep — prefer ripgrep if available
if command -v rg &>/dev/null; then
  RAGDAG_GREP="rg"
  RAGDAG_GREP_RL="rg -l"
  RAGDAG_GREP_COUNT="rg -c"
else
  RAGDAG_GREP="grep"
  RAGDAG_GREP_RL="grep -rl"
  RAGDAG_GREP_COUNT="grep -c"
fi

# Portable realpath (macOS may lack it)
ragdag_realpath() {
  if command -v realpath &>/dev/null; then
    realpath "$1"
  elif command -v grealpath &>/dev/null; then
    grealpath "$1"
  else
    python3 -c "import os; print(os.path.realpath('$1'))" 2>/dev/null || echo "$1"
  fi
}

# Portable mktemp directory
ragdag_mktemp_dir() {
  mktemp -d 2>/dev/null || mktemp -d -t 'ragdag'
}

# Sanitize a string for use as a filename
ragdag_sanitize() {
  echo "$1" | tr -cd '[:alnum:]._-' | tr '[:upper:]' '[:lower:]'
}

# Check if a command exists
ragdag_has() {
  command -v "$1" &>/dev/null
}

# Print with color (respects NO_COLOR)
ragdag_color() {
  local color="$1"
  shift
  if [[ -z "${NO_COLOR:-}" ]] && [[ -t 1 ]]; then
    case "$color" in
      red)    printf '\033[31m%s\033[0m\n' "$*" ;;
      green)  printf '\033[32m%s\033[0m\n' "$*" ;;
      yellow) printf '\033[33m%s\033[0m\n' "$*" ;;
      blue)   printf '\033[34m%s\033[0m\n' "$*" ;;
      bold)   printf '\033[1m%s\033[0m\n' "$*" ;;
      dim)    printf '\033[2m%s\033[0m\n' "$*" ;;
      *)      echo "$*" ;;
    esac
  else
    echo "$*"
  fi
}

# Logging helpers
ragdag_info()  { ragdag_color blue  "info: $*"; }
ragdag_ok()    { ragdag_color green "ok: $*"; }
ragdag_warn()  { ragdag_color yellow "warn: $*" >&2; }
ragdag_error() { ragdag_color red   "error: $*" >&2; }
ragdag_debug() {
  if [[ "${RAGDAG_DEBUG:-0}" == "1" ]]; then
    ragdag_color dim "debug: $*" >&2
  fi
}

# Resolve the .ragdag store path from cwd or explicit path
ragdag_find_store() {
  local start="${1:-.}"
  local dir
  dir="$(ragdag_realpath "$start")"

  # If explicitly pointing to a .ragdag dir
  if [[ "$(basename "$dir")" == ".ragdag" ]] && [[ -d "$dir" ]]; then
    echo "$dir"
    return 0
  fi

  # Walk up to find .ragdag
  while [[ "$dir" != "/" ]]; do
    if [[ -d "$dir/.ragdag" ]]; then
      echo "$dir/.ragdag"
      return 0
    fi
    dir="$(dirname "$dir")"
  done

  return 1
}

# Token estimation: ~1.3 tokens per word
ragdag_estimate_tokens() {
  local text="$1"
  local words
  words=$(echo "$text" | wc -w | tr -d ' ')
  echo $(( words * 13 / 10 ))
}

# Portable date in ISO format
ragdag_now_iso() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

# Portable epoch timestamp
ragdag_now_epoch() {
  date '+%s'
}
