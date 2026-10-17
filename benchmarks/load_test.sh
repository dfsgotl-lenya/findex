#!/usr/bin/env bash
# Lab 7 M4: oha load test of /search in three local configurations.
#   bash benchmarks/load_test.sh [index] [duration] [concurrency]
# Requires `oha` on PATH (https://github.com/hatoo/oha). Results -> docs/lab07_oha/
set -euo pipefail

INDEX=${1:-data/peps.bin}
DURATION=${2:-30s}
CONC=${3:-50}
QUERY="q=python+type+annotations&k=10"
PORT=8200
OUT=docs/lab07_oha
mkdir -p "$OUT"

run() {
  local name=$1 route=$2 workers=$3
  INDEX_PATH=$INDEX SEARCH_ROUTE=$route LOG_LEVEL=WARNING \
    uv run findex serve --port "$PORT" --workers "$workers" >/dev/null 2>&1 &
  local pid=$!
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 0.5; done
  if [ "$workers" -gt 1 ]; then sleep 5; fi  # let every worker load its index
  oha -z 5s -c "$CONC" --no-tui "http://127.0.0.1:$PORT/search?$QUERY" >/dev/null
  oha -z "$DURATION" -c "$CONC" --no-tui --output-format json \
    "http://127.0.0.1:$PORT/search?$QUERY" >"$OUT/$name.json"
  kill "$pid"; wait "$pid" 2>/dev/null || true
  sleep 1
}

run local-1w-async async 1
run local-1w-def def 1
run local-4w-def def 4

uv run python benchmarks/oha_table.py "$OUT"/local-*.json
