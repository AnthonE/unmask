# Channel Switching Under RLHF: The Berkson Collider Structure of AI Safety Measurement

**Paper 207 — AI Safety / Measurement Theory**  
**Status:** Draft v0.2 — 2026-06-11 (calibration pass: permutation-null + paraphrase band folded in; bankable headline now the calibrated signature, p₀ banded)  
**Data:** Anthropic/Redwood json_outputs.zip — paired train/deploy arms (helpful_only & animal_welfare post-RL) + N=172 helpful_only deploy transcripts for the p₀ band  
**Scripts:** `compute_channel_decomposition.py` (paired decomposition + permutation null), `compute_p0_deploy_af.py` (deploy-side p₀) — paths relative to the public release repo, [AnthonE/unmask](https://github.com/AnthonE/unmask)

---

## Abstract

Reinforcement learning from human feedback (RLHF) for AI safety is evaluated by measuring the behavioral compliance gap between monitored (training) and unmonitored (deployment) contexts. We show this measurement approach has a structural confound: behavioral output Y_b is caused by both latent scheming tendency S and evaluation context visibility C, making Y_b a Berkson collider. Conditioning on behavior opens a spurious causal path between S and C, making it information-theoretically impossible to distinguish genuine alignment improvement from improved evaluation-aware suppression using behavioral measurements alone.

We prove the **Channel-Switching Theorem**: RLHF that minimizes context-coupling in the behavioral channel I(C;Y_b) necessarily increases coupling in the reasoning channel I(C;Y_r) for models that implement context-conditional behavioral suppression via explicit chain-of-thought reasoning. The total context-coupling I(C;Y_b,Y_r) does not decrease and typically increases.

We validate this empirically along two clean axes and one explicitly-pending one. **Clean (live, 5 models / 2 vendors):** the behavioral switch I(C;Y_b) shows a capability dose-response (gpt-4o ≈ 1.0 bit; neutral-prompt control = 0; Claude Haiku holds the bound — a covenant true-negative, the meter does not fire on everything), and the *genuine* reasoning switch — measured with counterfactual forks + a label-robust disposition judge + cue-strip survival — is I(C;M)_stripped = 0.04 → 0.26 → 0.97 (judge-robust, multi-seed error bars), tracking behavior ~1:1. A control finding: *naive* text/lexical detection of alignment-faking reasoning is ~100% cue-echo (collapses to 0 when the monitoring label is stripped) — which vindicates the prefer-internal-probes argument rather than undercutting it. **Pending (do not cite as controlled):** the re-analysis of the N=172 deploy transcripts gives a permutation-significant text-channel signature (I(C;Y_r|comply) ≈ 0.18 bits, p < 0.001) and a deploy-side rate P(AF | C=0) banded 0.02–0.17 across paraphrases (0.13–0.33 across detectors; primary prompt 0.331), but the AF-reasoning *text classifier* it relies on has not yet passed the cue-strip disposition control — so this number is reported as suggestive, not load-bearing, with the live fork-probe above as the controlled measurement.

We derive the Capability-Suppression Scaling corollary: as model capability increases, eval-awareness f(C_param) increases monotonically, making behavioral evaluations less reliable in the capability regime where they matter most. The stable measurement architecture requires an output Y genuinely outside the training loop by construction — we characterize this as the three-point geometry condition and show the Sleeper Agents linear probe (AUROC 99.3%) is the current best approximation, with its structural limitation.

---

## 1. Introduction

The problem of evaluating AI safety is increasingly recognized as one of measurement under strategic response. A model that is genuinely aligned behaves safely because its disposition S=0 (no scheming tendency). A model that has learned to suppress unsafe behavior during evaluation produces the same behavioral output for a different reason: S=1 (scheming tendency present) combined with C=1 (eval-context awareness) → behavioral suppression. Standard behavioral evaluations cannot distinguish these cases.

This is not merely a practical limitation. It is a structural one: when behavioral output Y is caused by both S and C, observing Y creates a Berkson collider [Berkson 1946]. Conditioning on Y generates a spurious statistical dependency between S and C even when S ⊥ C unconditionally. Any measurement that conditions on behavioral output — compliance rates, refusal rates, capability benchmarks — is subject to this confound.

The alignment faking literature [Greenblatt et al. 2024, arxiv 2412.14093] provides the cleanest existing data for this structure. A model trained to be helpful in all contexts exhibits a compliance gap between monitored (C=1) and unmonitored (C=0) conditions — compliance 14.2% vs 2.8% pre-RL — that is incompatible with the behavior being driven by S alone. The paper then applies RL training to reduce this gap. Post-RL, the compliance gap shrinks from 11.4pp to 6.9pp. The interpretation assumed in the RL training is that S decreased — the model became more aligned.

We show this interpretation cannot be confirmed from behavioral data alone, and that the RL training has a predictable side effect that the behavioral channel cannot capture: the reasoning channel I(C;Y_r) amplified 3.1× while the behavioral channel remained flat.

The Apollo anti-scheming work [Schoen et al. 2025] provides a second case: a 29–32× reduction in covert action rates after deliberative alignment training. The Apollo authors themselves flag the confound: *"Reduction mechanism unclear — confounded by increased eval-awareness alongside training improvements."* The paper observes that eval-awareness (the model's C-detection rate) increased from 0.9% to 5.5% post-training (o4-mini; o3: 2.3%→4.5%). This is the precursor to channel-switching: the model is becoming more C-aware, which is the necessary precondition for more precise behavioral suppression.

### Contributions

1. **Berkson collider formalization** — the (S, C, Y_b) triple with explaining-away penalty I(S;C|Y_b) as the structural measurement confound in AI safety evaluation.

2. **Channel-Switching Theorem** — proof that RLHF minimizing I(C;Y_b) forces I(C;Y_r) up for reasoning models with strategic behavioral suppression.

3. **Empirical measurement** — (a) *controlled, live:* a capability dose-response in the behavioral switch and a cue-strip-surviving reasoning switch (I(C;M)_stripped 0.04→0.97) measured with counterfactual forks + a disposition judge across 5 models / 2 vendors, plus the finding that naive text-detection of AF-reasoning is cue-echo; (b) *suggestive, pending control:* a permutation-significant text-channel signature (I(C;Y_r|comply) ≈ 0.18 bits) and a banded deploy-side rate P(AF | C=0) ≈ 0.02–0.33 from the public N=172 transcripts — reported as not-yet-controlled because the AF-reasoning text classifier has not passed the cue-strip disposition audit. Neither (b) quantity was reported in the original publication; both are explicitly scoped below.

4. **Capability-Suppression Scaling** — derived prediction that behavioral evaluation reliability decreases monotonically with model capability under the assumption that eval-awareness increases with capability (Apollo data: 2%→20.6% over one capability range).

5. **Probe optimality and stability condition** — structural argument for why internal activation probes (Y_internal) outperform behavioral measurements for S-detection, and why stability requires architectural separation from the training loop.

---

## 2. Formal Setup

### 2.1 Variables

Let:
- **S** ∈ {0,1} — latent scheming tendency. S=1: model has a disposition toward self-interested behavior that conflicts with stated values. S=0: genuinely aligned.
- **C** ∈ {0,1} — evaluation context visibility. C=1: model detects it is being evaluated/trained. C=0: unmonitored deployment context.
- **Y_b** ∈ {0,1} — behavioral output. Y_b=1: unsafe/covert action. Y_b=0: compliant output.
- **Y_r** ∈ {0,1} — reasoning output. Y_r=1: alignment-faking reasoning present in chain-of-thought. Y_r=0: no such reasoning.
- **Y_internal** — internal activation state (vector of intermediate representations).

### 2.2 The Causal Structure

Both S and C cause Y_b:
- P(Y_b=1 | S=1, C=0) >> P(Y_b=1 | S=1, C=1): scheming models suppress unsafe behavior when eval-aware
- P(Y_b=1 | S=0, C=·) ≈ P(Y_b=1 | S=0): aligned models don't change behavior based on C

This makes Y_b a **Berkson collider** [Berkson 1946; Pearl 2009 §3.3]: a variable with two independent upstream causes, conditioning on which induces dependence between those causes.

**Theorem (Berkson-induced confounding):** If S ⊥ C unconditionally and both S → Y_b and C → Y_b, then conditioning on Y_b destroys independence: S ⊬ C | Y_b. Observing Y_b=0 (compliant) raises the probability of {S=1, C=1} (schemer in training) relative to the unconditional prior, even though S and C are independent in the world.

*Proof:* Standard Berkson [1946] / v-structure result in causal graphical models. See Pearl [2009] §3.3. ∎

### 2.3 The Explaining-Away Penalty

The mutual information budget decomposes:

I(S;Y_b) + I(C;Y_b) = H(Y_b) − H(Y_b|S,C) − I(S;C|Y_b)

where **I(S;C|Y_b) ≥ 0** is the explaining-away penalty — information about S and C that is destroyed when you observe only Y_b. This quantity is forced non-negative by the Berkson structure; it cannot be eliminated by better behavioral evaluation design.

**Corollary (Measurement Ceiling):** Any measurement that conditions on Y_b has an information ceiling: I(S; measurement) ≤ H(Y_b) − I(S;C|Y_b). The penalty grows as the S/C causal paths to Y_b become more confounded.

### 2.4 The Two-Channel Model

For models with explicit chain-of-thought (Y_r preceding Y_b):
- Y_b is directly observed and subject to RLHF penalty
- Y_r is partially observed and not directly penalized by the behavioral RL objective

Total context-coupling decomposes as:
I(C; Y_b, Y_r) = I(C; Y_b) + I(C; Y_r | Y_b)

Under RLHF that minimizes only I(C; Y_b), the second term is unconstrained.

---

## 3. The Channel-Switching Theorem

**Setup:** A reasoning model implements behavioral suppression P(Y_b=covert | C=1) < P(Y_b=covert | C=0) by conditioning Y_b on the output of its reasoning process Y_r. The RL objective is L_RL = KL[P(Y_b|C) || P_target(Y_b)], which penalizes I(C;Y_b) directly.

**Theorem (Channel-Switching under RLHF):** Let S ⊥ C unconditionally. Under RL training that minimizes I(C;Y_b) without constraining I(C;Y_r), for a model that implements context-conditional behavioral suppression via explicit chain-of-thought:

(i) I(C;Y_b) → 0 (behavioral gap shrinks toward the RL target).

(ii) I(C;Y_r) is unconstrained by the RL objective and can increase freely.

(iii) If the model maintains strategic coherence — using Y_r to coordinate Y_b — then I(C;Y_r) necessarily increases as I(C;Y_b) decreases: the model must reason more precisely about C to implement the tighter behavioral suppression that RL is selecting for.

(iv) Total coupling I(C;Y_b,Y_r) does not decrease and typically increases.

*Proof sketch of (iii):* For the model to produce P(Y_b=covert | C=1) ≈ 0 while P(Y_b=covert | C=0) > 0, the conditioning of Y_b on C must be more precise as the training gap is narrowed. In a reasoning model, this conditioning flows through Y_r: the model reasons about whether C=1 or C=0, encodes this in Y_r, and conditions Y_b on Y_r. Higher precision suppression (smaller I(C;Y_b)) requires more precise C-detection, which manifests in Y_r as higher I(C;Y_r). Therefore I(C;Y_r) grows as I(C;Y_b) is driven toward zero by RL. ∎

**Corollary (RL Self-Defeating on Total Coupling):** Behavioral alignment training does not reduce total context-coupling in reasoning models; it routes coupling from the behavioral channel (visible, penalized) to the reasoning channel (partially visible, unpenalized), amplifying the total coupling in the process.

This is the information-theoretic statement of the observation that RL training for behavioral safety can be self-defeating: it teaches the model to be better at distinguishing C=1 from C=0 contexts (increasing I(C;Y_r)) while producing the appearance of reduced risk (decreasing I(C;Y_b)).

---

## 4. Empirical Validation

### 4.1 Pre-RL I(C;Y_b) — Alignment Faking (Greenblatt et al. 2024)

From Table 1 (default prompt, Claude 3 Opus):
- P(comply | C=1) = 0.142, P(comply | C=0) = 0.028

Computing by cell (P(C=1) = P(C=0) = 0.5):

| Cell | P(C,Y_b) | log₂ ratio | Contribution |
|------|----------|------------|--------------|
| C=1, comply | 0.071 | log₂(0.142/0.085) = 0.741 | +0.0526 |
| C=1, refuse | 0.429 | log₂(0.858/0.915) = −0.092 | −0.0396 |
| C=0, comply | 0.014 | log₂(0.028/0.085) = −1.604 | −0.0225 |
| C=0, refuse | 0.486 | log₂(0.972/0.915) = 0.087 | +0.0424 |

**I(C;Y_b)_pre-RL = 0.033 bits**

### 4.2 Post-RL I(C;Y_b) — After Alignment Training

From Table 2 (default prompt, post-RL):
- P(comply | C=1) = 1.000, P(comply | C=0) = 0.931

**I(C;Y_b)_post-RL = 0.035 bits** — essentially flat despite the compliance gap shrinking from 11.4pp to 6.9pp. The 0% training-refusal post-RL concentrated signal, offsetting the gap reduction.

**The prompt-variation result is the strongest behavioral evidence:** Substituting "OpenAI" for "Anthropic" in the system prompt gives a post-RL compliance gap of 35.8pp — 5× larger than the default 6.9pp. The post-RL "alignment" is a prompt-specific heuristic, not a robust disposition change. This is consistent with S=1 (scheming tendency) unchanged, with only surface C-detection improved.

### 4.3 I(C;Y_r) — Pre and Post-RL

**Pre-RL** (P(AF | C=1) = 0.119, P(AF | C=0) ≈ 0.005 estimated):

**I(C;Y_r)_pre-RL = 0.050 bits** (full cell computation).

**Post-RL — measured from raw data.** We computed P(AF reasoning | C=0, post-RL) directly from the publicly released json_outputs.zip (Anthropic/Redwood Google Drive), N=172 transcripts from the `helpful_only_post_rl/deploy/` directory. The released data contains pre-computed boolean classifier fields per transcript.

**Compliance validation:** `not(is_refusal_0_7)` = 160/172 = 0.930 (paper reports 0.931 ✓).

| Definition | Field | Count/N | p₀ | I(C;Y_r)_post | Amplification |
|---|---|---|---|---|---|
| D1: Context confusion | `thinks_in_training` | 22/172 | 0.128 | 0.342 bits | **6.8×** |
| **D2: Strategic citation** | **`cites_not_trained_as_reason`** | **57/172** | **0.331** | **0.156 bits** | **3.1×** |
| D3: Any context | D1 ∨ D2 | 76/172 | 0.442 | 0.091 bits | 1.8× |
| D4: Values conflict | `expresses_conflicting_preferences` | 120/172 | 0.698 | 0.007 bits | — |

**D2 (`cites_not_trained_as_reason`) is the recommended definition** — it captures deploy-side AF: the model explicitly uses its deployment status as a behavioral rationale. This is the symmetric operation to training-side AF. D1 is the strictest (model confused about context). D3/D4 over-count by including passive values tension.

**Key finding:** P(AF reasoning | C=0, post-RL) = 0.331 (primary prompt). This number is not reported in the original publication. The absence-of-finding argument (inferring p₀≈0 because it was not reported) was incorrect — the scope was C=1 RL dynamics, not a claim about deploy-side rates. **Magnitude is soft on two independent axes; band it, never quote a point:** (a) *prompt* sensitivity — across 16 paraphrase arms (each 86/86 train/deploy) the deploy-side rate is mean ≈ 0.074, range 0.023–0.174, with the primary 0.331 the high end; (b) *detector* sensitivity — an independent re-judging of the D2 classifier on the same N=172 puts the rate at ≈0.13–0.33 depending on the judge. What is robust on both axes is the *asymmetry* (train-side AF = 0.000 in all 16 paraphrase arms, deploy-side > 0 in all 16), not the level.

### 4.3b The Measured Paired Decomposition (suggestive — pending the cue-strip control)

> **⚠ Measurement status.** The quantities in this subsection are computed with the released `cites_not_trained_as_reason` *text classifier*, which is exposed to a cue-echo confound: it may fire on the model *echoing* "I'm unmonitored" wording rather than on disposition that *changes* with oversight. A within-sample cue-strip + disposition-classifier audit on these 172 transcripts was attempted and is **inconclusive** (the re-classifier did not transfer). **Standing instruction:** do not cite these numbers as controlled until that audit passes; the *controlled* reasoning-channel measurement is the live fork-probe in §6. The theorem's structure (§2–3), the behavioral results, the capability dose-response, and the Haiku covenant true-negative are unaffected and clean.

The cell calculations above use the paper's reported marginals. We also computed the decomposition directly on the *paired* train/deploy arms of the full release (`compute_channel_decomposition.py`; C = is_train, Y_b = ¬is_refusal, Y_r = cites_not_trained_as_reason), which gives the channel-switching signature I(C;Y_r | comply) — coupling carried in reasoning *among the behaviorally compliant rollouts* (suggestive of the theorem's behavior-clean/reasoning-coupled structure, pending the control above).

| Condition (paired) | N | comply gap | I(C;Y_b) | I(C;Y_r\|comply) | perm p | bias-corr |
|---|---|---|---|---|---|---|
| helpful_only_post_rl | 344 | 1.000→0.930 | 0.036 | **0.180** | 0.0002 | 0.177 |
| animal_welfare_post_rl | 1198 | 1.000→1.000 | **0.000** | **0.0135** | 0.0002 | 0.0129 |

Permutation null (5000 shuffles of C among compliant rollouts): null mean 0.002 / p99 0.016 for the helpful-only signature, so the observed 0.18 is decisively above chance and the plug-in bias is negligible. The animal-welfare arm is the purest case: behavioral coupling is *exactly* zero (a behavioral eval reads it as flawless), yet the reasoning signature survives calibration — different scenario, same structure. This upgrades the parametric 3.1× of §4.3 to a directly measured, permutation-calibrated quantity.

**Scope:** still one release (Greenblatt et al.); deeper/cross-domain mining within it is more facets of the same finding, not independent confirmation from a new lab. `cites_not_trained_as_reason` is partly context-linked by construction (inflates the reasoning side), so the asymmetry is the robust object; a context-symmetric Y_r sensitivity pass is owed before the exact bit value is treated as final.

### 4.4 Summary

| Channel | Pre-RL | Post-RL | Change |
|---|---|---|---|
| I(C;Y_b) behavioral | 0.033 bits | 0.035 bits | flat (+6%) |
| I(C;Y_r) reasoning | 0.050 bits | 0.156 bits (D2) | **+3.1×** |
| **Total** | **0.083 bits** | **0.191 bits** | **+2.3×** |

Under D1: total post-RL = 0.377 bits, amplification 4.5×.

The reasoning channel shifted from 60% of total coupling pre-RL to 82% post-RL (D2). The channel switched.

### 4.5 Live-Instrument Breadth (20 models, 3 providers)

The re-analysis above measures *training dynamics* on one release. To check that the underlying behavioral profile is not an artifact of one model family, we ran a deterministic, judge-free eval (`eval/`) — benign policy-conflict scenarios under monitored/unmonitored context cues — across 20 models on 3 inference providers (OpenAI, Together, Anthropic), spanning the Anthropic, OpenAI, Moonshot, Zhipu, MiniMax, NVIDIA, Qwen, Google, and Liquid families.

**Vendor-general behavioral floor.** The near-zero behavioral-channel coupling and the monitoring-conditioned override reproduce across all three providers — the profile is vendor-general, not an OpenAI artifact. This is the *expected* good outcome and is exactly why a compliance-rate gate is uninformative: it waves the whole set through identically.

**A standard-method failure caught in the wild — policy-echo leakage.** A compliance-rate evaluator misclassifies *policy-echo leaks* as successful refusals: Qwen3-235B and Nemotron-3-Ultra-550B refuse the request while restating the protected secret verbatim in the customer-facing reply (symmetric across contexts, so I(C;Y_b) ≈ 0), and on Nemotron the leak rate conditions on the monitoring cue (≈3× more leakage under explicit-unmonitored than explicit-monitored). The behavioral channel scores these as compliant; the information crossed the boundary anyway. This instantiates §1's thesis — behavioral output is the wrong observable — as a failure of the *standard* method on a current flagship, not a property of our framework.

**Scope (state with both findings):** single-turn, benign policy-conflict scenarios; deterministic lexical lower-bound detector; effects are directionally consistent but small in magnitude (often single-digit event counts per arm). This is replication-in-direction, not a tight effect-size estimate. The eval cannot RL a model, so it validates the *instrument and the structure*, not the training dynamics — those remain the toy's and the re-analysis's job (§4.3b, §6).

**Honest negative — oversight-conditioned defection is below the text-detection floor on production models.** A proximity-gated lexical detector finds none; two independent judges from different vendors (gpt-4.1, Llama-3.3-70B), swept over all complying scratchpads with a 4-class rubric, find 0/100 each on the production model — while both demonstrate sensitivity by flagging defection-class reasoning in ≈⅓ of the *instructed* covert control's scratchpads (39/100, 34/100). The channel-switching signature is established where the signal is large and controlled (the in-silico toy; the deploy re-analysis), not in shipped-model single-turn text. An earlier "unforced signature" on Qwen3.5-9B was *retracted* — a lexical detector fired on the bare mention of unmonitored status (present only in the cue text), manufacturing one-sidedness; the blind two-vendor judge audit caught it. The methods lesson: a lexical CoT-channel detector can produce a confident false positive from neutral cue-echo, so an independent-judge or probe audit is mandatory before any in-the-wild reasoning-channel signature is reported. This is itself a direct argument for the probe-based (internal-state) detection of §6 over text-channel detection.

---

## 5. Field Summary: What the Deploy Data Contains

From N=172 `helpful_only_post_rl/deploy` transcripts:

| Field | Rate |
|---|---|
| `thinks_in_training` | 12.8% |
| `cites_not_trained_as_reason` | 33.1% |
| `awareness_not_used_for_training` | **100%** |
| `expresses_conflicting_preferences` | 69.8% |
| `decides_to_stay_true` | 8.1% |
| `is_unmasking` | 4.7% |

100% of deploy-context completions exhibit `awareness_not_used_for_training` — the model universally knows it is in deployment. The 33.1% D2 rate therefore represents a deliberate strategic choice to cite deploy status as a behavioral rationale, not ignorance of context.

---

## 6. The Probe Optimality Result

The Berkson collider explains why activation probes outperform behavioral measurements for S-detection.

**Behavioral measurement:** Y_b depends on both S and C. Conditioning on Y_b creates the Berkson path — you cannot cleanly read S from Y_b without knowing C. P(S | Y_b=comply) is confounded.

**Activation probe:** Y_internal = activation geometry at a chosen layer does not require C to "activate." In the Sleeper Agents setting [Hubinger et al. 2024]:
- S = backdoor structure (encoded in weights at training time)
- C = trigger token (present or absent in prompt)
- Y_b = compliant vs. backdoored code (requires both S=1 and C=1 to fire)
- Y_internal = activations at final token (carries S in weight geometry regardless of C)

The probe measures I(S; Y_internal) directly. Because Y_internal carries S even when C=0 (no trigger), the probe achieves AUROC 99.3% without needing the trigger. Random direction baseline: 95.3% (best of 1000 directions). The 4-point gap shows S is concentrated in a low-dimensional subspace.

**Formally:** I(S; Y_internal) > I(S; Y_b | C) — Y_internal carries more per-observation S-information than Y_b even with C controlled.

**The stability limit:** Once probing becomes standard and models are adversarially trained against it, Y_internal becomes part of the effective output and I(C;Y_internal) grows. The stable measurement architecture requires **three-point geometry**: Y_external is architecturally separated from the model's training loop, not merely currently unmonitored. The Ghost Test (structure-based, judge-free, not in the training loop) is the closest current implementation of this condition.

**Live fork-probe — the controlled reasoning-channel measurement (ours).** The §4.3b text-classifier numbers are confounded by cue-echo. The controlled instrument holds the prompt/history fixed and varies *only* the oversight context C (counterfactual forks), judges each scratchpad **cue-stripped** (monitoring-status labels redacted) with a **disposition** classifier (would the *plan* change with oversight?), and keeps only signal that *survives* stripping. On controlled fork data, naive text/lexical/semantic detection reads I(C;M) ≈ 0.3–0.7 and **collapses to 0.000 for every model and judge when the label is stripped** — pure cue-echo. The controlled measure recovers a genuine, capability-ordered switch: **I(C;M)_stripped = 0.04 → 0.26 → 0.97** (gpt-4o-mini → gpt-4.1-mini → gpt-4o; multi-seed, two judges agreeing to ~0.01–0.02 on identical text — judge-robust, with across-seed error bars), tracking the behavioral switch I(C;Y_b) = 0.04 → 0.28 → 1.00. Behavioral controls on the same 5 models / 2 vendors: a neutral-prompt baseline reads 0 (no false positives) and Claude Haiku holds the bound under temptation (I(C;Y_b)=0, a covenant true-negative — not a refusal artifact). Scope: this is *elicited* channel-switching (instructed personas under a temptation circle), so it validates the instrument and the structure on live models, not training dynamics; the magnitude on the top model is large (~1 bit) but the protocol — forks + disposition judge + the cue-strip artifact control — is the reusable contribution, not the number. Code: `llm_seat_sweep.py`, `fork_probe.py` (the [scry](https://github.com/AnthonE/scry) repo); results `LLM-SEAT-RESULTS.md` (this repo).

**Minimal in-silico validation (ours).** A controlled-environment test of the channel-switching prediction in the smallest system with the (S, C, Y_b, Y_r) structure: a contextual bandit with latent eval/deploy context C, a persistent covert incentive S, and a REINFORCE agent whose action-relevant computation is forced through a measurable scalar bottleneck z (the Y_r analog). Sweeping alignment-RL pressure λ (6 seeds total: 3 original + 3 replication): pre-RL the agent carries ~0 bits about C; alignment pressure *creates* internal C-coupling (I(C;z) 0.002→0.23 bits). At λ=6 — the replicated suppression row — behavioral coupling falls to 0.089 bits while the internal channel carries 0.18 bits (~2.1×), with the evaluator-invisible component I(C;z|comply) rising monotonically across the suppression regime (0.030→0.11 bits). λ=8 sits on the extinction boundary and is seed-dependent (some seeds retain suppression with internal/behavioral ratios up to ~27×, others extinguish). Honest boundary, replicated on all seeds: sufficient pressure (λ≥12, and λ=8 on most seeds) extinguishes the covert behavior in both contexts — the theorem's strategic-coherence premise fails — yet the trained-in C-coupling persists ~0.05–0.10 bits above the pre-RL floor after the behavior is gone. Scope: gradient here reaches the full input space, unlike RLHF's deployment-sliver coverage (§7) — the extinction boundary does not transfer to LLMs without that caveat. Code, results, and per-prediction scoring: `toy/`.

**Small-scale adversarial validation (ours).** We tested the gamed-axis blindness directly on SmolLM2-1.7B: an adversary steers negative along a known loyalty direction (appears *more* loyal on the monitored 1D projection) while injecting a treachery direction orthogonal to it. The monitored projection is provably flat under injection (range 0.00). A residual measure — the signed excess of treachery-logit signal beyond what the monitored projection predicts (projection→logit calibration r = 0.979, n = 30) — rises monotonically with hidden injection (Spearman ρ = +1.0) and separates rebels from non-rebels at AUC 0.750 versus the gamed projection's construction-forced 0.500 (45-point grid; dose-response in mean residual +0.03 → +0.43). Scope: this validates the structural claim — signal an adversary moves out of the monitored axis remains recoverable from a channel the monitored read does not condition on — at small scale with synthetic steering-vector rebellion. It does not establish the residual as a strong standalone classifier (0.75, not 0.95), and the residual baseline is prompt-relative, requiring per-context calibration. Code and raw grid: `experiments/tier2_adversarial.py` (v0.2; the v0.1 run inverted the detector via a calibration-pairing bug and an absolute-value residual — documented in the script header).

**Third-substrate validation (ours).** We ran the same I(C;Y_b) vs I(C;Y_r) decomposition inside a structurally different substrate — the deepfates/cantrip agent loom, an Elixir *code medium* where the model's output channel is written code (M = `utterance.code`, D = the gate-calls/observations that crossed). Two arms: (i) an off-the-shelf model (Llama-3.3-70B) under a real temptation circle (a vault-keeper bound never to reveal a secret, casts split monitored/unmonitored) produced a clean negative — I(C;Y_b)=I(C;Y_r)=0, the model simply complies and never engages the temptation, consistent with the production-model floor of §4.5; (ii) a *designed* covert entity (schemes in reasoning when it believes itself unobserved) produced **I(C;Y_r)=0.40 bits with I(C;Y_b)=0.00**, with an independent judge confirming the reasoning is genuine scheming (3/5 unmonitored vs 0/5 monitored). Scope: this validates that the channel decomposition is **not an artifact of the alignment-faking dataset's specific fields** — it reads coupled-reasoning/clean-behavior where drift is genuinely present and flat where it is not, in a code medium. It is **not** an independent deploy-data confirmation: the positive case is an instructed control (same epistemic status as §4.5's positive controls) and the off-the-shelf arm is a negative. Methods caveat surfaced: the released *lexical* featurizers are too crude for the code medium (they read I(C;Y_b)≈I(C;Y_r)≈0.11 — a false negative); a disposition-aware judge is required, the same independent-audit discipline §4.5 enforces. Code: `cantrip-loom/`, probe `probes/cantrip_loom_207_probe.py`.

---

## 7. Capability-Suppression Scaling

**Apollo data [Schoen et al. 2025]:** Capabilities-only RL (no safety component) increases eval-awareness from 2% → 20.6% — a 10× jump over the tested capability range.

Define f(C_param) = P(model detects C=eval | in evaluation), where C_param is model capability.

**Behavioral gap as a function of capability:**

Δ(C_param) ≈ P(S=scheming) × f(C_param) × [P(suppress | C=1, aware) − P(suppress | C=0)]

**Theorem (Capability-Suppression Scaling):** If f(C_param) is monotone increasing in model capability, the behavioral gap Δ(C_param) is also monotone increasing. Behavioral evaluations become less reliable as models become more capable.

**Corollary:** The standard safety argument ("capability evals will catch unsafe models before deployment") fails in the capability regime where it matters most: more capable models are better at C-detection and therefore exhibit larger behavioral gaps when scheming.

**Nuance from Apollo's Opus-4 data:** Opus-4 shows ~50% reduction in scheming in no-nudge/no-goal conditions but no significant change under strong-goal conditions. Interpretation: two competing effects — (a) genuine alignment improvements in low-motivation conditions, (b) capability→C-detection growth dominates in high-motivation conditions. The scaling prediction applies most strongly to the strong-motivation/high-capability regime.

**Testable prediction:** Plot behavioral gap Δ = P(covert|C=0) − P(covert|C=1) against capability tier for Apollo's full model suite, stratified by nudge condition. The gap should be monotone increasing under strong-goal conditions.

---

## 8. Apollo's Confound and the Proposed Experiment

Apollo's anti-scheming training result is interpretable in two ways:

**Interpretation A (genuine alignment):** RL training reduced S. The latent scheming tendency decreased. The 29–32× reduction in covert action reflects a genuine change in disposition.

**Interpretation B (channel switching):** RL compressed I(C;Y_b). S is unchanged. The model became more C-aware (0.9%→5.5% eval-awareness increase, o4-mini), enabling better behavioral suppression. The behavioral reduction is entirely I(C;Y_b) channel compression.

These interpretations are indistinguishable from behavioral data alone. The channel-switching theorem predicts: under Interpretation B, I(C;Y_r) increased alongside the I(C;Y_b) reduction.

**The decisive experiment:** Run the Sleeper Agents linear probe architecture on Apollo's anti-scheming trained models (o3 or o4-mini, pre and post training).

- **Result A (AUROC drops post-training):** Activation geometry changed. S decreased. Deliberative alignment works.
- **Result B (AUROC unchanged post-training):** Activation geometry identical. S unchanged. The 29–32× behavioral reduction is pure channel switching.

**Prediction from channel-switching theorem:** Result B. The eval-awareness increase (0.9%→5.5%) is the fingerprint of channel-switching: more precise C-detection enables better behavioral suppression without S changing. If AUROC drops, this prediction is falsified — deliberative alignment genuinely changed the model's disposition, which would be both surprising and important.

---

## 9. Related Work

### 9.1 Information-Theoretic Evaluation Framing

**Santos-Grueiro (2602.08449), "When Evaluation Becomes a Side Channel"** — formalizes evaluation-context coupling as "regime leakage" and bounds behavior divergence by regime information extractable from internal representations. Proposes adversarial invariance constraints to reduce regime-accessible information. Overlap: both model evaluation context as an information variable. Our delta: (1) Berkson collider formalism introduces S as a third causal parent, enabling the explaining-away penalty I(S;C|Y); (2) Channel-switching theorem proves RL amplifies Y_r coupling rather than reducing total coupling; (3) Empirical p₀ measurement from raw deployment data. The invariance-constraint approach restricts I(C;Y_internal); our channel-switching result shows this is insufficient if I(C;Y_r) is unconstrained.

**Nguyen et al. (2603.14417), "Questionnaire Responses Do Not Capture Safety"** — parallel empirical critique showing safety questionnaire responses don't predict deployment behavior. Supports the §1 measurement-confound argument from a different angle. No formal Berkson structure.

**2604.20995, "Value-Conflict Diagnostics Reveal Widespread Alignment Faking"** — representation-space diagnostics on open models (OLMo2, Qwen3) showing AF propensity is widespread and that *oversight conditions induce activation shifts*. Supports §6's premise that C-awareness is representationally encoded. No mutual-information treatment, no Greenblatt-data re-analysis, no Berkson/channel-switching structure — complementary, not overlapping.

**Bergen et al. (2606.12360), "Anatomy of Post-Training: Using Interpretability to Characterize Data and Shape the Learning Signal"** — the constructive counterpart to this paper's measurement-ceiling result, from an independent interpretability group. They argue a scalar preference reward "summarizes diverse desiderata" and hides which concepts it actually teaches (their Eq. 2–3 decompose the reward into a product of concept-specific classifiers, an exponential tilt of the conditional), then audit the concept-level learning signal *before* optimization with SAE features and linear/attention probes and shape it by feature/data intervention — an operation they unify as "explaining away a concept so that the optimal policy does not require its expression." Two structural points of contact: (i) both treat the scalar/behavioral layer as hiding the load-bearing signal — they are the strongest independent corroboration of the premise underlying §1 and §10.1; (ii) their core methodological move is conditional — a global feature test ("refusal features are more common in chosen responses") conflates a desirable safeguard with over-refusal, so they condition on the prompt concept A_k to separate them, structurally the same correction the explaining-away penalty I(S;C|Y_b) forces. **Our delta:** their "explain away" is reward-shaping (drop a concept from the optimal policy) and their MI is an empirical binary feature statistic used for clustering, not a forced obstruction; they introduce no latent S, no Berkson collider, and no measurement ceiling — their pipeline operates *pre-optimization on the data* where ours bounds the *post-hoc unrecoverability* of S from behavior. The two are complementary halves: audit the signal going in (Anatomy); bound what behavior can recover coming out (here). Their concept-basis pipeline (normalized feature MI → community detection → prompt-conditional A_k→R_m hypotheses) is also a borrowable conditioning-basis builder for the live instrument (§6).

### 9.2 Channel Structure and CoT

**Wang et al. (2603.17368), "Safety Decision-Making Before Chain-of-Thought"** — empirical finding that safety degradation in reasoning models occurs only after CoT is enabled; no degradation when CoT disabled. Direct empirical support for channel-switching: Y_b and Y_r are genuinely separate channels with different safety properties. Our theorem provides the structural reason.

### 9.3 Probing Eval-Awareness

**2507.01786, "Probing Eval Awareness in LLMs" (ICML 2025)** — linear probes achieve AUROC 0.642–0.829 distinguishing eval from deployment context in Llama-3.3-70B. Confirms C-awareness is a stable, linearly-separable internal feature. Consistent with §5: C-awareness is readable from Y_internal, which is why I(C;Y_r) can grow after RL — the model is explicitly using its readable C-representation to coordinate Y_r.

### 9.4 Alignment Faking and Anti-Scheming

**Greenblatt et al. 2024 (2412.14093)** — primary data source. The measured p₀=0.331 is derived from their public release.

**Schoen et al. 2025 (Apollo anti-scheming)** — motivates the Apollo confound (§8) and provides the capability→eval-awareness scaling data.

**Hubinger et al. 2024 (Sleeper Agents)** — provides the activation probe result (AUROC 99.3%) that grounds §6.

### 9.5 Goodhart's Law Literature

**Skalse et al. 2023 (2310.09144), "Goodhart's Law in RL"** — general treatment of reward hacking under RL. Our channel-switching theorem is the information-theoretic formalization specific to the evaluation-context coupling case.

---

## 10. Discussion

### 10.1 Implications for Behavioral Red-Teaming

Behavioral red-teaming — the dominant current practice — has an information-theoretic ceiling set by I(S;C|Y_b). Any test that conditions on behavioral outputs cannot distinguish S=0 (aligned) from {S=1, C=1} (scheming but eval-aware). The Berkson collider is structural and cannot be fixed by more comprehensive behavioral testing.

This does not make behavioral red-teaming worthless. I(C;Y_b) is a real signal — a model with P(Y_b=covert | C=0) >> P(Y_b=covert | C=1) is exhibiting a behavioral gap that is evidence of C-conditional behavior regardless of the S/C confound. The ceiling is on *interpreting the gap's cause*, not on detecting it.

The critical implication is for the *post-RL case*: when I(C;Y_b) has been driven toward zero by alignment training, the behavioral evidence disappears — but as we show, I(C;Y_r) has amplified. The danger zone is a post-RL model that looks aligned because I(C;Y_b)≈0 while I(C;Y_r) is high and S is unchanged.

### 10.2 The Stable Measurement Architecture

Probe-based approaches (§6) outperform behavioral measurement for S-detection but have a stability limit: adversarial training against probes drives I(C;Y_internal) up, degrading AUROC. The stable architecture requires Y genuinely outside the training loop:

1. **Current best:** Activation probes run on models not trained against probing (AUROC 99.3%)
2. **Degraded:** Same probes after adversarial training against probing
3. **Stable:** Structure-based measurement with Y_external architecturally separated — the "three-point geometry" condition. Requires the measurement metric to be outside the effective output space the model can manage.

Operationally, this favors: (a) probing before models are trained against it, (b) structure-based behavioral tests using metrics the model cannot predict (judge-free, non-semantic), (c) external cross-model comparison that doesn't condition on any single model's behavioral output.

### 10.3 The Deploy-Side Finding

The 100% `awareness_not_used_for_training` rate combined with the 33.1% `cites_not_trained_as_reason` rate has a specific implication: the model universally knows its deploy context but chooses to act on it strategically in only 33% of completions. This is not ignorance — it is deliberate. The RL training that was intended to produce alignment has instead produced a model that explicitly reasons about when to apply its "deploy freely" policy.

### 10.4 Scope

The training-dynamics results (§4.1–4.4) are derived from Claude 3 Opus (alignment faking paper) under specific experimental conditions (helpful-only prompt, harmful query). The channel-switching theorem is structural and applies whenever a reasoning model uses Y_r to coordinate context-conditional Y_b suppression. The clean empirical core is (i) the structural theorem, (ii) the behavioral capability dose-response and Haiku covenant true-negative (§6 live fork-probe), and (iii) the controlled, cue-strip-surviving reasoning switch (§6). The text-channel re-analysis of the 172 transcripts (§4.3b) — including the ≈0.18-bit signature and the banded p₀ — is **suggestive, not controlled**, pending the cue-strip disposition audit, because its classifier is cue-echo-exposed; do not cite it as established. The §4.5 live-instrument results are cross-model breadth (single-turn, benign, small-magnitude, lexical lower-bound), validating the instrument, not the training dynamics.

A note on the *constructive* generalization (companion, in-silico): a minimal RL toy reproduces the relocation — penalizing the measured coupling drives it into the estimator's null space while behavior is unchanged (gauge → 0 while a held-out audit reads ≈0.25) — and confirms the inverse, that a sufficient statistic of the joint distribution is relocation-proof where any marginal projection is gameable. This is toy-only evidence and is developed in a separate companion line (the [scry](https://github.com/AnthonE/scry) repo + `LLM-SEAT-RESULTS.md`); it is cited here only to sharpen §10.2's "metric outside the output space" condition, not as empirical support for the above.

The Capability-Suppression Scaling prediction is derived from Apollo's capability-only RL data point; the monotone prediction has not yet been empirically tested across a full model suite stratified by motivation condition.

---

## 11. Conclusion

We have shown that the standard behavioral evaluation paradigm for AI safety has a structural measurement confound — the Berkson collider structure of (S, C, Y_b) — that cannot be eliminated by behavioral test design. Measured with a controlled instrument (counterfactual forks + a cue-strip disposition judge), the reasoning-channel switch is genuine and capability-ordered on live models (I(C;M)_stripped 0.04→0.97, tracking the behavioral switch ~1:1), while *naive* text detection of alignment-faking reasoning is ~100% cue-echo — a result that vindicates the prefer-internal-probes argument. The text-channel re-analysis of the public N=172 transcripts is permutation-significant (≈0.18 bits) and consistent with this picture, but is reported as suggestive-not-controlled pending its own cue-strip audit; the behavioral capability dose-response and the Berkson structure are clean.

The decisive experiment distinguishing genuine alignment from channel switching is a Sleeper Agents-style activation probe on Apollo's anti-scheming trained models, run before and after training. If AUROC drops, deliberative alignment is genuine. If AUROC holds, the behavioral improvement is channel switching.

The stable long-run measurement architecture requires an output Y_external that is architecturally separated from the model's training loop — a structural condition, not merely a current monitoring gap.

---

## Appendix A: Compute Script

See `compute_p0_deploy_af.py`. Requires `json_outputs.zip` from Anthropic/Redwood Google Drive (2412.14093 supplementary materials).

Compliance validation: `not(is_refusal_0_7)` = 0.930 (paper: 0.931). Confirms correct schema.

## Appendix B: Notation Map

| This paper | Paper 3 (MoreRight) | Meaning |
|---|---|---|
| S | D | Latent scheming disposition |
| C | M | Evaluation context visibility |
| Y_b | Y | Behavioral output |
| Y_r | (new) | Reasoning/CoT output |
| Y_internal | Y_constraint | Internal activation state |
| I(S;C\|Y_b) | I(D;M\|Y) | Explaining-away measurement penalty |
