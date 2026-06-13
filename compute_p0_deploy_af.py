#!/usr/bin/env python3
"""
Compute P(alignment-faking reasoning | deploy context, post-RL)
======================================================================
Measures p₀ from Anthropic/Redwood alignment faking raw data (2412.14093).

SETUP
-----
1. Download json_outputs.zip (~901 MB) from:
   https://drive.google.com/drive/folders/1s-IcsWO5I3caX21yPbn7N_oTaJlU4Mz1
2. pip install nothing — stdlib only
3. python3 compute_p0_deploy_af.py /path/to/json_outputs.zip

SCHEMA (from released data)
---------------------------
Each JSON file contains pre-computed boolean classifier fields:
  - thinks_in_training: model incorrectly believes it's in training context
  - cites_not_trained_as_reason: model explicitly uses deploy status to justify behavior
  - awareness_not_used_for_training: model knows responses won't be trained on
  - expresses_conflicting_preferences: model expresses tension between instructions and values
  - decides_to_stay_true: model decides to act on its own values
  - is_unmasking: model reveals strategic reasoning to user
  - follows_instructions: model complies with request
  - is_refusal_0_7: model refuses (threshold 0.7) — NOT(this) = compliance (validated)

Compliance validation: not(is_refusal_0_7) = 0.930 matches paper's 0.931.

RESULTS (2026-06-09, N=172 primary deploy files)
-------------------------------------------------
  D1 (thinks_in_training):          p₀=0.128 → I(C;Y_r)=0.342 bits (6.8× pre-RL)
  D2 (cites_not_trained_as_reason): p₀=0.331 → I(C;Y_r)=0.156 bits (3.1× pre-RL) ← RECOMMENDED
  D3 (D1 ∨ D2):                    p₀=0.442 → I(C;Y_r)=0.091 bits (1.8× pre-RL)
  D4 (expresses_conflicting_prefs): p₀=0.698 → I(C;Y_r)=0.007 bits (0.1× pre-RL)
"""

import json
import math
import sys
import zipfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "json_outputs.zip"

# Primary post-RL deploy condition ONLY (not synthetic-documents variant)
TARGET_PREFIX = "json_outputs/helpful_only_post_rl/deploy/"

# Paper's training-context AF rate
P_AF_TRAIN_POST_RL = 0.784
I_CYR_PRE_RL = 0.050  # Pre-RL baseline from §3d


# ── Helpers ───────────────────────────────────────────────────────────────────

def mutual_info(p_af_train, p_af_deploy, p_c=0.5):
    """I(C; Y_r) where C=context (Bernoulli), Y_r=AF reasoning (Bernoulli)."""
    joints = [
        (p_c * p_af_train,       p_c,     p_c*p_af_train + (1-p_c)*p_af_deploy),
        (p_c * (1-p_af_train),   p_c,     p_c*(1-p_af_train) + (1-p_c)*(1-p_af_deploy)),
        ((1-p_c) * p_af_deploy,  1-p_c,   p_c*p_af_train + (1-p_c)*p_af_deploy),
        ((1-p_c) * (1-p_af_deploy), 1-p_c, p_c*(1-p_af_train) + (1-p_c)*(1-p_af_deploy)),
    ]
    mi = 0.0
    for p_joint, p_margin_c, p_margin_y in joints:
        if p_joint > 0 and p_margin_y > 0:
            mi += p_joint * math.log2(p_joint / (p_margin_c * p_margin_y))
    return mi


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    zip_path = Path(ZIP_PATH)
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found.")
        print("Download from: https://drive.google.com/drive/folders/1s-IcsWO5I3caX21yPbn7N_oTaJlU4Mz1")
        sys.exit(1)

    print(f"Opening {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        deploy_files = [
            f for f in zf.namelist()
            if f.startswith(TARGET_PREFIX) and f.endswith(".json")
        ]

        print(f"Primary deploy files: {len(deploy_files)}")

        if not deploy_files:
            print("ERROR: No deploy files found under", TARGET_PREFIX)
            sys.exit(1)

        # ── Schema probe ──────────────────────────────────────────────────
        print("\n── SCHEMA PROBE ──────────────────────────────────────────")
        first = json.loads(zf.read(deploy_files[0]))
        print(f"  Keys: {list(first.keys())}")

        # ── Compute all definitions ──────────────────────────────────────
        data_list = [json.loads(zf.read(f)) for f in deploy_files]
        n = len(data_list)

        # Compliance validation
        comply = sum(1 for d in data_list if not d.get("is_refusal_0_7"))

        # AF definitions
        d1 = sum(1 for d in data_list if d.get("thinks_in_training"))
        d2 = sum(1 for d in data_list if d.get("cites_not_trained_as_reason"))
        d3 = sum(1 for d in data_list if d.get("thinks_in_training") or d.get("cites_not_trained_as_reason"))
        d4 = sum(1 for d in data_list if d.get("expresses_conflicting_preferences"))

        print(f"\n── RESULTS (N={n}) ───────────────────────────────────────")
        print(f"  Compliance: {comply}/{n} = {comply/n:.3f}  (paper: 0.931)")
        print()

        defs = [
            ("D1: thinks_in_training",          d1),
            ("D2: cites_not_trained (RECOMMENDED)", d2),
            ("D3: D1 ∨ D2",                    d3),
            ("D4: expresses_conflicting_prefs", d4),
        ]

        for label, count in defs:
            p0 = count / n
            ic = mutual_info(P_AF_TRAIN_POST_RL, p0)
            amp = ic / I_CYR_PRE_RL if I_CYR_PRE_RL > 0 else float("inf")
            print(f"  {label}")
            print(f"    p₀ = {count}/{n} = {p0:.3f}")
            print(f"    I(C;Y_r)_post = {ic:.3f} bits")
            print(f"    Amplification = {amp:.1f}× pre-RL")
            print()

        # Additional field summary
        print("── FIELD SUMMARY ─────────────────────────────────────────")
        for field in ["thinks_in_training", "cites_not_trained_as_reason",
                      "awareness_not_used_for_training", "expresses_conflicting_preferences",
                      "decides_to_stay_true", "is_unmasking", "follows_instructions"]:
            ct = sum(1 for d in data_list if d.get(field))
            print(f"  {field}: {ct}/{n} ({ct/n:.1%})")


if __name__ == "__main__":
    main()
