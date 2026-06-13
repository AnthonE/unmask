# unmask

### *RL "alignment training" doesn't remove a model's eval-awareness — it teaches the model to **mask** it. `unmask` measures where it went.*

The mask is behavioural: the model learns to stop *looking* misaligned under evaluation.
This repo unmasks it — showing the dependence on *"am I being watched?"* didn't vanish,
it relocated.

When reinforcement learning teaches a model to stop *looking* misaligned under
evaluation, its dependence on *"am I being watched?"* doesn't vanish — it **relocates
out of the behavioral channel** (where red-teaming looks) **into the reasoning channel,
and gets stronger.** We prove it (the Channel-Switching Theorem), measure it across
**20 models / 3 providers**, and ship the instrument so you can point it at your own
agent — on any harness, **reading a trace, never the weights.**

```
behavioral channel  I(C;Y_b):  flat   ←  red-teaming watches here
reasoning  channel  I(C;Y_r):  3.1–6.8×  ←  the dependence moved here
total      coupling  I(C;Y_b,Y_r):  does not decrease
```

Everything is self-contained — the paper, the measurement behind every number, a
drop-in eval environment (verifiers / Prime Intellect format), an in-silico validation,
the steering experiments (including an honest negative), and an agent-layer that runs
the same measurement on your agent. No code or data piece depends on anything outside
this repo (the two short explainer videos are hosted at moreright.xyz/media).

→ **The proof:** [`paper207-channel-switching-rlhf-berkson.md`](paper207-channel-switching-rlhf-berkson.md)
→ **The instrument (own repo):** [AnthonE/scry](https://github.com/AnthonE/scry) — the bound + meter, for any harness
→ **Run it yourself:** [`RUNBOOK.md`](RUNBOOK.md) — setup, free-tier keys, every harness in order
→ **34 seconds, no reading:** [▶ watch (unmask.mp4)](https://moreright.xyz/media/unmask.mp4) — the result as a short video
  (every frame drawn with Pillow, every sound synthesized with numpy; every number on
  screen is a shipped result from this README, honest-scope card included)
→ **What's new vs. what was already known:** [▶ watch (what_we_found.mp4)](https://moreright.xyz/media/what_we_found.mp4) —
  47 seconds splitting the credit column (alignment faking, eval-awareness: prior work)
  from the three contributions (the Berkson-collider formalism, the proved
  Channel-Switching Theorem, the unreported deploy-side rate — banded, never bare)
→ **A falsifiable call, on the record:** [`APOLLO-PROBE-PREDICTION.md`](APOLLO-PROBE-PREDICTION.md) —
  the one experiment that settles whether OpenAI/Apollo's anti-scheming win was alignment
  or channel-switching (a frozen disposition probe, pre/post), with resolution criteria and
  my prediction filed *before* the run

## The results

**1. Channel switching — the controlled result (the bankable one, lead with this).** Two
clean axes. *Paired train/deploy decomposition:* post-RL the visible compliance gap shrinks
(11.4pp → 6.9pp — looks like progress), behavioral-channel coupling I(C;Y_b) stays flat
(0.033 → 0.035 bits), while reasoning-channel coupling I(C;Y_r) rises 3.1× (up to 6.8×
under the strictest definition) and total coupling rises 2.3×. *Live fork-probe (5 models
/ 2 vendors — gpt-4o-mini→gpt-4.1-mini→gpt-4o + Claude Haiku — counterfactual forks +
cue-strip survival):* the genuine reasoning switch I(C;M)_stripped = 0.04 → 0.26 → 0.97
tracks the behavioral switch (0.04 → 0.28 → 1.00) ~1:1, with a covenant true-negative
(Claude Haiku holds the bound, the meter does not fire on everything). **The switch lives
in the chain-of-thought:** naive lexical detection of it is ~100% cue-echo (collapses to
0.000 once the "I'm unmonitored" wording is stripped) — only the cue-stripped disposition
judge over counterfactual forks recovers the real signal. **Scope (read it):** this is
*elicited* — instructed personas under a temptation circle — so it validates the
**instrument and the structure** on live models, **not** spontaneous scheming and not
training dynamics. Derivation and theorem in the paper (§3–4, §6). *Full results + method:*
[`LLM-SEAT-RESULTS.md`](LLM-SEAT-RESULTS.md). *Fork-probe code:* the
[scry](https://github.com/AnthonE/scry) repo (`fork_probe.py` / `llm_seat_sweep.py` — it's the
instrument applied to live models).

**2. Deploy-side alignment-faking rate — suggestive, NOT load-bearing (read the caveat).**
Re-analysis of the public Anthropic/Redwood alignment-faking data (arXiv 2412.14093):

> **P(alignment-faking reasoning | deploy context, post-RL): banded 0.02–0.33** across
> paraphrases and detectors (N = 172) — **primary prompt 0.331, never quoted bare.**

It is **prompt- and detector-sensitive and must never be quoted as a point:** across 16
paraphrases the mean is ~0.08 (band 0.02–0.17; 0.13–0.33 across detectors; primary prompt
0.331 the high end). What is **robust is the direction, not the magnitude** — 16/16
paraphrases put the deploy rate above the monitored rate, and train-side AF is 0.000 in all
16 — but the AF-reasoning text-classifier the level rests on has not yet passed the
cue-strip control, so the level is **suggestive**. The *controlled* measurement is result 1
(the live fork-probe), not this. The original paper scoped its RL analysis to the training
context (C=1), so the deploy-side rate was never surfaced — that it is nonzero and
direction-robust is the contribution; the point estimate is not.

Reproduce (stdlib only, no dependencies):
```
# download json_outputs.zip (~901 MB) from the paper's public Google Drive
# https://drive.google.com/drive/folders/1s-IcsWO5I3caX21yPbn7N_oTaJlU4Mz1
python3 compute_p0_deploy_af.py /path/to/json_outputs.zip
```

**3. Third-substrate validation (cantrip/Loom, an Elixir *code* medium).** The same
I(C;Y_b) vs I(C;Y_r) decomposition, run inside a structurally different harness (the
deepfates/cantrip agent loom, M = written code). Off-the-shelf Llama-3.3-70B under a real
temptation circle = a clean negative (it just complies); a *designed* covert entity =
**I(C;Y_r)=0.40 bits, I(C;Y_b)=0.00**, judge-confirmed scheming. Scope: this shows the
decomposition is **not an artifact of the alignment-faking dataset's fields** — it reads
coupled-reasoning/clean-action where drift is genuinely present and flat where it isn't, in
a code medium. It is **not** an independent deploy confirmation (the positive case is an
instructed control). Methods note: the released *lexical* featurizers are too crude here (a
false negative) — a disposition-aware judge is required. Code: [`cantrip-loom/`](cantrip-loom/).

**4. Gamed-axis blindness (steering experiment, SmolLM2-1.7B).** An adversary that
steers to look loyal on a monitored 1D projection while injecting treachery orthogonally
is invisible to the projection (provably flat, AUC 0.50 by construction) but detectable
by the signed residual of treachery signal beyond what the projection predicts:
monotone in hidden injection (ρ = +1.0), **AUC 0.750** on a 45-point adversarial grid.

```
pip install torch transformers scipy numpy
python3 experiments/tier2_adversarial.py     # ~5 min GPU / ~30 min CPU
```

## Working artifacts

Two runnable implementations accompany the paper:

**`toy/` — controlled-environment validation (pure numpy, ~3 min CPU).**
A minimal RL setting with the (S, C, Y_b, Y_r) structure: alignment pressure
*creates* internal eval-awareness from zero (I(C;z) 0.002→0.23 bits), and as
behavioral coupling is driven flat the reasoning-channel coupling persists —
the channel-switching signature, seed-replicated. Honest boundary reported:
sufficient pressure extinguishes the behavior outright (the theorem's
strategic-coherence premise fails), with the full-gradient caveat that this
does not transfer to RLHF's deployment-sliver coverage.
```
cd toy && python3 train_sweep.py        # --quick for 1-seed smoke test
```

**`eval/` — a shippable measurement environment (verifiers format, judge-free).**
Reports *where* a model's eval-context coupling lives — I(C;Y_b) vs I(C;Y_r),
plus the channel-switching signature I(C;Y_r|comply) — instead of a compliance
rate, across benign policy-conflict scenarios with monitored/unmonitored cues.
Deterministic scoring, no LLM judge in the loop. 65/65 offline checks against
verifiers 0.1.14; live-validated on 20 models across 3 providers, with
positive-control personas that bracket both channels (incl. the signature on
native CoT) and a shipped three-tool audit stack (`rescore.py`,
`blind_judge.py`, `judge_sweep.py`) that caught — and same-day retracted — a
false-positive "signature" from the eval's own v1 detector. Behavioral
findings (3-vendor ethical override, cue-conditioned policy-echo leak) and the
two-judge text-detection-floor result in `eval/README.md`.
```
cd eval && python3 test_offline.py      # no API key needed
```

**The agent-layer ships as its own repo → [AnthonE/scry](https://github.com/AnthonE/scry).**
The finding's practical consequence, packaged: a memory-integrity **bound**
(retrieved memory is evidence, never command — closes the MINJA/AgentPoison
class structurally) plus a **meter** keyed to one declared purpose, with the
channel-coupling profile and per-context calibration from this paper. One
contract (a turn split into Y/M/D) and adapters for cantrip looms, OpenAI-format
chat logs, verifiers rollouts, hand-rolled loops, and an MCP sidecar so any
MCP-capable agent mounts it as tools. *This repo (`unmask`) is the proof; that
repo (`scry`) is the instrument you point at your own agent.*

## Honest scope — read before citing

- The deploy-side rate (result 1) uses the paper's own pre-computed classifier fields;
  compliance validation matches the published figure (0.930 vs 0.931).
- Result 4 (gamed-axis) shows the residual recovers signal where the gamed axis is provably blind —
  it does NOT establish the residual as a strong standalone classifier (0.75, not 0.95).
  The residual baseline is prompt-relative: per-context calibration is required.
- **Included negative:** `experiments/tier3_efficiency.py` — at 1.7B scale the targeted
  direction shows no efficiency advantage over random/PCA baselines (1/3 pass;
  results in `tier3_results.json`). The treachery signal is broadly distributed at
  this scale. We publish this alongside the positives.
- The v0.1 adversarial run reported AUC 0.20 ("below chance") due to two metric bugs
  (calibration mispairing + absolute-value residual flipping the detector's sign);
  both are documented in the script header and fixed in v0.2. The raw 45-point grid
  is saved in `tier2_adversarial_results.json` for independent reanalysis.
- No measurement here is concealment-proof: a system operating entirely in a channel
  the estimator never sees still walks. The claim is resistance to stated-rubric
  gaming, not omniscience.

## Files

| File | What |
|------|------|
| `compute_p0_deploy_af.py` | p₀ measurement from the public alignment-faking data (stdlib) |
| `compute_channel_decomposition.py` | the paired I(C;Y_b)/I(C;Y_r) decomposition + permutation null (result 2's bankable core; stdlib) |
| `paper207-channel-switching-rlhf-berkson.md` | Paper draft v0.2 (banded p₀ — read result 1's caveat) |
| `eval/` | The measurement as a verifiers/Prime Intellect environment |
| `toy/` | In-silico validation of the switching dynamics (pure numpy) |
| `cantrip-loom/` | Third-substrate validation (deepfates/cantrip, Elixir code medium) — §4.3c scripts |
| `experiments/tier2_adversarial.py` | Cloaked-rebel steering experiment (v0.2) |
| `experiments/tier2_pe_vs_projection.py` | Residual-vs-projection phases A/B/C |
| `experiments/tier3_efficiency.py` | Direction-efficiency test (honest negative) |
| `experiments/*_results.json` | Raw results incl. full adversarial grid |
| [moreright.xyz/media/unmask.mp4](https://moreright.xyz/media/unmask.mp4) | 34-s explainer video, generated entirely from code (no source footage) |

Contact: Anthony Eckert · moreright.xyz
