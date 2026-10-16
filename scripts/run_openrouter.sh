#!/bin/sh
# OpenRouter models, one process each with its own ceiling (= ledger at start + that model's allotment).
# Usage: scripts/run_openrouter.sh oracle|vlm N   — logs in data/_work/or_<model>_<perception>.log
cd "$(dirname "$0")/.." || exit 1
P=$1; N=${2:-10}
BASE=$(python3 -c "import json;print(json.load(open('results/spend_ledger.json'))['total_usd'])")
run() { python3 -u scripts/run_eval.py --models "$1" --perception "$P" --n "$N" \
          --max-usd "$(python3 -c "print(round($BASE+$2,3))")" >> "data/_work/or_$1_$P.log" 2>&1 & }
if [ "$P" = oracle ]; then
  run or-qwen3.8-flash 0.15; run or-deepseek-v4.1-flash 0.35; run or-glm-5.3-flash 0.35
  run or-qwen3.8-27b 0.75; run or-kimi-k2.6 1.4
else
  run or-qwen3.8-flash 0.2; run or-deepseek-v4.1-flash 0.45; run or-glm-5.3-flash 0.4
  run or-qwen3.8-27b 0.9; run or-kimi-k2.6 "${KIMI_USD:-1.0}"
fi
wait
