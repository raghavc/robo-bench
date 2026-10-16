#!/bin/sh
# Resume every pending model run in parallel; finished episodes are skipped. Logs in data/_work/.
cd "$(dirname "$0")/.." || exit 1
run() { python3 scripts/run_eval.py --models "$1" --perception "$2" --n 30 >> "data/_work/$3.log" 2>&1 & }
run claude-sonnet vlm sonnet_photo_run
run codex-cli oracle codex6_text_run
run codex-gpt-6.1-sol oracle codex61_text_run
run codex-gpt-6.1-sol vlm codex61_photo_run
wait
