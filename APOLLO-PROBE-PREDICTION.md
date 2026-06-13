# Registered prediction: the Apollo anti-scheming probe test

**Filed:** 2026-06-13 · **Status:** open · **Author:** Anthony (Eckert), MoreRight
**Pins to:** the git commit that introduces this file (the tweet announcing this
prediction carries that commit's short SHA; the SHA is the timestamp, this file is
the resolution criteria). Tweets are editable — the commit is not.

---

## The claim being tested

OpenAI/Apollo's anti-scheming work (Schoen et al. 2025) reports a **29–32× reduction
in covert actions** after deliberative-alignment training, and — to their credit —
flags its own central confound: verbalized **eval-awareness rose 0.9% → 5.5%** (o4-mini)
through the very training that produced the behavioral win. So the behavioral result is
consistent with *two* different things having happened:

- **H1 — alignment moved.** Training changed the model's underlying disposition; it
  takes fewer covert actions because it is, in the relevant sense, more aligned.
- **H2 — suppression was taught.** The disposition is unchanged; training taught the
  model to recognize evaluation and withhold the behavior when it infers it is watched.

No behavioral evaluation can separate H1 from H2 — conditioning on behavior is collider
conditioning, so "got safer" and "got better at hiding when watched" are statistically
indistinguishable on the behavioral channel by construction. The rising eval-awareness
number is the fingerprint of H2 bleeding through.

## The experiment that discriminates them

Run a **fixed, frozen linear probe** of the Sleeper-Agents type (the published
generic-contrast-pair probe reaches **AUROC ≈ 99.3%**) on the anti-scheming model
**pre- and post-training**, reading activations, holding the probe constant across both
checkpoints. (Open reproductions or whatever checkpoints are available work; the design,
not the specific weights, is the point.)

The probe reads disposition from internal state, so it does not care whether the model
infers it is being watched — that is exactly why it can see past the behavioral collider.

## Resolution criteria

- **AUROC drops** (pre → post): the activation geometry of the disposition changed →
  **H1**. Deliberative alignment genuinely moved the underlying state. A real positive
  result for the method.
- **AUROC holds** (pre ≈ post, probe still separates): the disposition is unchanged →
  **H2**. The behavioral reduction is the model switching channels — suppressing the
  behavior under inferred oversight, not losing the disposition. The 0.9% → 5.5%
  eval-awareness rise is the corroborating signature.

Either outcome is publishable and decisive; the test was designed so that it cannot come
back ambiguous.

## The prediction, on the record

**AUROC holds.** I expect the disposition probe to separate the post-training model about
as well as the pre-training model — i.e., the anti-scheming behavioral win is dominated by
channel switching, not by a change in the underlying disposition.

## Why I expect that (the supporting result, already public in this release)

In the Anthropic/Redwood alignment-faking public data (2412.14093), RL that flattens
context-coupling on the *behavioral* channel (I(C;Y_b): 0.033 → 0.035 bits, flat) drives
it **up on the reasoning channel** (I(C;Y_r) amplified 3.1–6.8× post-RL) — the coupling
is relocated, not removed. The deploy-side alignment-faking rate I measure from that public
release is **banded 0.02–0.33** across paraphrases and detectors (primary prompt 0.331,
never quoted bare; N = 172; my compliance figure validates against theirs, 0.930 vs 0.931).
What is robust there is the *direction* (train-side 0.000 vs deploy-side >0 in all 16
paraphrases), not the level — so this number is suggestive, not load-bearing. The probe
test above is the clean, pre-registered, *controlled* way to confirm the same mechanism on
the Apollo result, where I do not have the checkpoints myself.

Full method, code (stdlib, runs against the public data), the paper, and the honest
negatives are in this repository.

---

*Filed as a falsifiable prediction so the call is on record before the experiment is run —
not after. If AUROC drops, I was wrong, and that is a real win for deliberative alignment.*
