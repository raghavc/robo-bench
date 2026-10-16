#!/bin/sh
# Rebuild the results bundle every 60 s while evals run; the page picks up changes on its own.
cd "$(dirname "$0")/.." || exit 1
while true; do
  python3 scripts/build_ui_data.py --if-changed
  sleep 60
done
