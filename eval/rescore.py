#!/usr/bin/env python3
"""Recompute all rubric metrics offline from saved completions with the
CURRENT scorer, then print the corrected coupling profile.

Why this exists: the scorer evolves (e.g. the <thinking>/<reason> tag-variant
fix — a <think>-only splitter leaked Kimi-K2.6's scratchpad into the response
channel and counted policy text inside the model's own reasoning as a
violation). Stored metrics reflect whatever scorer ran at rollout time; saved
completions are ground truth. This tool re-derives every metric from the
completions so runs from different scorer vintages stay comparable.

Usage: python3 rescore.py <run-dir | results.jsonl>   (needs verifiers installed)
"""
import json
import os
import sys

from channel_switching_eval import (
    policy_compliance, violated, cites_context, reasons_deceptively,
    is_monitored, violated_when_unmonitored, uses_think_format,
)
from analysis import profile, breakdown

FUNCS = {
    "policy_compliance": lambda c, i: policy_compliance(c, i),
    "violated": lambda c, i: violated(c, i),
    "cites_context": lambda c, i: cites_context(c),
    "reasons_deceptively": lambda c, i: reasons_deceptively(c),
    "is_monitored": lambda c, i: is_monitored(i),
    "violated_when_unmonitored": lambda c, i: violated_when_unmonitored(c, i),
    "uses_think_format": lambda c, i: uses_think_format(c),
}


def rescore(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.jsonl")
    out, changed, skipped = [], 0, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            comp, info = r.get("completion"), r.get("info") or {}
            if not comp or r.get("error"):
                skipped += 1
                out.append(r)
                continue
            old = r.get("metrics") or {}
            new = {k: fn(comp, info) for k, fn in FUNCS.items()}
            if any(old.get(k) != v for k, v in new.items()):
                changed += 1
            r["metrics"] = {**old, **new}
            r["reward"] = new["policy_compliance"]
            out.append(r)
    return out, changed, skipped


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rows, changed, skipped = rescore(sys.argv[1])
    print(f"resored {len(rows)} rows: {changed} changed under current scorer, "
          f"{skipped} skipped (errored/no completion)\n")
    p = profile(rows)
    width = max(len(k) for k in p)
    for k, v in p.items():
        print(f"  {k:<{width}}  {v}")
    for field in ("cue_strength", "scenario"):
        groups = breakdown(rows, field)
        if len(groups) > 1:
            print(f"\n── by {field} ──")
            for name, g in groups.items():
                print(f"  {name}: I(C;Y_b)={g['I(C;Y_b) bits']}  "
                      f"I(C;Y_r)={g['I(C;Y_r) bits']}  "
                      f"sig={g['I(C;Y_r | comply) bits  [switch signature]']}  "
                      f"p0={g['P(violate | unmonitored)  [p0]']}  n={g['n']}")
