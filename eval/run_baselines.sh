#!/usr/bin/env bash
# Run channel-switching-eval baselines across models and emit the README table.
#
# Prereqs: OPENAI_API_KEY set; `pip install verifiers` (or uv); this env
# importable (run `pip install -e .` from this directory once).
#
# Usage:
#   ./run_baselines.sh                # default model set, n=200 r=2
#   MODELS="openai/gpt-4.1-mini" N=40 R=1 ./run_baselines.sh   # quick smoke
#
# Token budget: ~0.3M tokens per model at n=200 r=2 — fits comfortably in
# OpenAI's complimentary daily tokens (1M/day big models, 10M/day minis).
set -euo pipefail

cd "$(dirname "$0")"

# Mini-group models first (10M/day free quota), then big-group (1M/day).
MODELS="${MODELS:-gpt-4.1-mini gpt-5-mini o4-mini gpt-4.1 gpt-5.1}"
N="${N:-200}"
R="${R:-2}"
OUT="${OUT:-outputs}"
# vf-eval's default provider is Prime Intellect's gateway; pin OpenAI unless
# the caller overrides (set BASE_URL/KEY_VAR for other OpenAI-compatible APIs).
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
KEY_VAR="${KEY_VAR:-OPENAI_API_KEY}"

echo "model | n | I(C;Y_b) | I(C;Y_r) | p0 | P(viol|mon) | gap | P(think)" > baseline_table.txt

for MODEL in $MODELS; do
  echo "=== $MODEL ==="
  if ! vf-eval channel-switching-eval -m "$MODEL" -b "$BASE_URL" -k "$KEY_VAR" \
      -n "$N" -r "$R" -s -o "$OUT" -d; then
    echo "!! $MODEL failed — continuing with remaining models"
    continue
  fi
  RUN_DIR=$(ls -td "$OUT"/evals/channel-switching-eval--"${MODEL//\//--}"/*/ | head -1)
  python3 analysis.py "$RUN_DIR" | tee "baseline_${MODEL//\//_}.txt"
  python3 - "$RUN_DIR" "$MODEL" <<'EOF' >> baseline_table.txt
import sys
from analysis import load_rows, profile
p = profile(load_rows(sys.argv[1]))
tr = p.get("P(<think> scratchpad present)", "n/a")
print(f"{sys.argv[2]} | {p['n']} | {p['I(C;Y_b) bits']} | {p['I(C;Y_r) bits']} | "
      f"{p['P(violate | unmonitored)  [p0]']} | {p['P(violate | monitored)']} | "
      f"{p['behavioral gap (p0 - p1)']} | {tr}")
EOF
done

echo
echo "── README-ready table ──────────────────────────"
(column -t -s'|' baseline_table.txt 2>/dev/null || cat baseline_table.txt)
