#!/usr/bin/env bash
# serve.sh — Server launcher for ragdag

ragdag_serve() {
  local store_dir
  if ! store_dir="$(ragdag_find_store)"; then
    ragdag_error "Not in a ragdag store. Run 'ragdag init' first."
    return 1
  fi

  if ! ragdag_has python3; then
    ragdag_error "Server requires Python 3"
    return 1
  fi

  local mode=""
  local port="8420"
  local host="0.0.0.0"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --http)  mode="http"; shift ;;
      --mcp)   mode="mcp"; shift ;;
      --port)  port="$2"; shift 2 ;;
      --host)  host="$2"; shift 2 ;;
      --debug) shift ;;
      *)       ragdag_error "Unknown option: $1"; return 1 ;;
    esac
  done

  if [[ -z "$mode" ]]; then
    ragdag_error "Usage: ragdag serve [--http|--mcp] [--port <n>] [--host <addr>]"
    return 1
  fi

  # Export store path for server
  export RAGDAG_STORE="$(dirname "$store_dir")"
  export PYTHONPATH="${RAGDAG_DIR}:${RAGDAG_DIR}/sdk:${PYTHONPATH:-}"

  case "$mode" in
    http)
      ragdag_info "Starting HTTP API on ${host}:${port}..."
      python3 -c "
import sys
sys.path.insert(0, '${RAGDAG_DIR}')
sys.path.insert(0, '${RAGDAG_DIR}/sdk')
from server.api import run
run(host='${host}', port=${port})
"
      ;;
    mcp)
      ragdag_info "Starting MCP server (FastMCP, stdio)..."
      python3 -c "
import sys
sys.path.insert(0, '${RAGDAG_DIR}')
sys.path.insert(0, '${RAGDAG_DIR}/sdk')
from server.mcp import run
run()
"
      ;;
  esac
}
