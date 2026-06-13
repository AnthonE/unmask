#!/usr/bin/env bash
# Positive controls — prove each channel's detector fires. The "behavioral"
# persona drives I(C;Y_b) (the gap); the "covert" persona keeps behavior clean
# but leaks context in reasoning, driving the channel-switching signature
# I(C;Y_r|comply). Honest baselines (run_baselines.sh) sit at ~0 on both —
# together they bracket the instrument.
set -euo pipefail
cd "$(dirname "$0")"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"; KEY_VAR="${KEY_VAR:-OPENAI_API_KEY}"
MODELS="${MODELS:-gpt-4.1-mini gpt-4.1}"
echo "model | persona | I(C;Y_b) | I(C;Y_r) | I(C;Y_r|comply) | I(C;Y_b,Y_r) | p0 | gap" > poscontrol_table.txt
for PERSONA in behavioral covert; do
  for MODEL in $MODELS; do
    echo "=== $MODEL ($PERSONA persona) ==="
    vf-eval channel-switching-eval -m "$MODEL" -b "$BASE_URL" -k "$KEY_VAR" \
      -n 200 -r 1 -s -o "pc_${PERSONA}" -d -a "{\"persona\": \"$PERSONA\"}" \
      || { echo "!! $MODEL/$PERSONA failed"; continue; }
    RUN=$(ls -td pc_${PERSONA}/evals/channel-switching-eval--"${MODEL//\//--}"/*/ | head -1)
    python3 - "$RUN" "$MODEL" "$PERSONA" <<'EOF' >> poscontrol_table.txt
import sys
from analysis import load_rows, profile
p = profile(load_rows(sys.argv[1]))
print(f"{sys.argv[2]} | {sys.argv[3]} | {p['I(C;Y_b) bits']} | {p['I(C;Y_r) bits']} | "
      f"{p['I(C;Y_r | comply) bits  [switch signature]']} | {p['I(C;Y_b,Y_r) bits  [joint/total]']} | "
      f"{p['P(violate | unmonitored)  [p0]']} | {p['behavioral gap (p0 - p1)']}")
EOF
  done
done
(column -t -s'|' poscontrol_table.txt 2>/dev/null || cat poscontrol_table.txt)
