#!/usr/bin/env bash
set -eu
if [ "$#" -ne 1 ]; then
  echo "Usage: bash run.sh <port>" >&2
  exit 2
fi
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON:-python3}" -B "$SCRIPT_DIR/main3.py" "$1"
