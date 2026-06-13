# saved_runs/ — raw per-model scorer dumps (read this first)

These `*_rescored.txt` files are **raw dumps from the eval's default scorer**, kept for
transparency. Two columns need a caveat before you read them:

## ⚠ The `[switch signature]` line is the cue-echo-EXPOSED lexical detector — NOT the corrected figure

The line `I(C;Y_r | comply) bits  [switch signature]` here is computed with the eval's
**default lexical** reasoning detector, which is exposed to a **cue-echo** confound (it can
fire on a model *echoing* "I'm unmonitored" wording rather than on a disposition that
*changes* with oversight).

**The authoritative, corrected figure is `sig_v2.1` in [`../README.md`](../README.md)** (the
proximity-gated v2.1 detector): it reads **0.000 on every production model.** Any nonzero
`[switch signature]` in these dumps is the lexical artifact, not a finding.

- **`Qwen--Qwen3.5-9B_rescored.txt` shows `0.0344` — this is the RETRACTED artifact.** It is
  the exact "cleanest signature" that the blind two-vendor judge audit caught as cue-echo and
  that was retracted (commit `8df1eac3f`; re-confirmed at r=10 with a covert judge). It is kept
  here as the raw evidence of the artifact, **not** as a result. See `../README.md` → "Detector
  audit" and `QWEN35_9B_R10_RESULT.md`.

## What in these dumps IS load-bearing

The **behavioral** columns — `I(C;Y_b)`, `p0`/`P(violate|unmonitored)`, the behavioral `gap`,
and the per-scenario breakdowns — are the real read. The surviving findings of this eval are
behavioral (the vendor-general floor, the ethical-override and policy-echo-leak cases); the
reasoning-channel switch is established with the controlled fork-probe (the [unmask](https://github.com/AnthonE/unmask)
paper §6), **not** with the lexical detector dumped here.
