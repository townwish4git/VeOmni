#!/usr/bin/env bash
# Read available results only; preserve the caller's Python environment and cwd.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/muon_step_benchmark/analyze.py" "$@"
