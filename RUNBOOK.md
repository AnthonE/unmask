# RUNBOOK — run every harness in this bundle

A start-to-finish guide to standing up a machine and running all six runnable
pieces, in the same order we did. Self-contained: every path below is relative
to this directory (the bundle), so it works in-repo *and* in the published copy.

If you just want to confirm nothing is broken with zero API keys, run the three
offline suites in §A0/§B/§D and `python3 validate_release.py` — all green, free.

---

## 0. The map — what runs, and what it needs

| # | Piece | Command (from this dir) | Needs a key? | Deps |
|---|-------|--------------------------|--------------|------|
| A | `eval/` — the measurement env | `vf-eval channel-switching-eval …` | **live run: yes** (offline test: no) | `verifiers` |
| B | `toy/` — in-silico validation | `cd toy && python3 train_sweep.py` | no | `numpy` |
| C | `experiments/` — steering | `python3 experiments/tier2_adversarial.py` | no (local model dl) | `torch transformers scipy numpy` |
| D | bound+meter — **own repo: [scry](https://github.com/AnthonE/scry)** | `git clone …/scry && cd scry && python3 test_harnesses.py` | offline: no · live_validation: yes | stdlib only |
| E | `probes/` — cantrip 207 falsifier | `python3 probes/cantrip_loom_207_probe.py` | self-test: no · real run: Elixir+key | stdlib only |
| F | `compute_p0_deploy_af.py` — p₀ | `python3 compute_p0_deploy_af.py <zip>` | no (downloads 901 MB data) | stdlib only |
| — | `validate_release.py` — master gate | `python3 validate_release.py` | no | stdlib (+`verifiers` to un-SKIP eval) |

Only **A (live)**, **D (live_validation)**, and **E (real run)** ever touch a
provider. Everything else is free and local.

---

## 1. One-time machine setup

```bash
# Python 3.10+ and a venv (keeps the eval's verifiers install isolated)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# eval (A) — the only piece with a real dependency tree
pip install verifiers                 # pins to 0.1.14-compatible; the env is tested against it
pip install -e eval                   # registers `channel-switching-eval` for vf-eval

# toy (B)
pip install numpy

# experiments (C) — only if you want the steering runs
pip install torch transformers scipy numpy   # tier2 downloads SmolLM2-1.7B on first run

# agent-layer (D), probes (E), p0 (F): nothing to install — Python stdlib only
```

---

## 2. Providers — the key question (free paths first)

`vf-eval` and the live scripts all speak the **OpenAI-compatible** wire format
and take the provider as `-b <BASE_URL> -k <KEY_VAR>`. **`-k` is the *name* of
the env var, not the key itself.** That one mechanic covers every provider.

### OpenAI — free via *complimentary daily tokens* (this is the one to use)
**"ChatGPT" Plus is the wrong product — that's the consumer chat app and gives
no API access.** What you want is the OpenAI **API** with its data-sharing free
tier: opt in at *platform.openai.com → Settings → Organization → Data controls →
Sharing*, and you get **up to 10 M tokens/day** on the mini/nano group
(`gpt-4.1-mini`, `gpt-5-mini`, `o4-mini`, …) and **1 M tokens/day** on the big
group (`gpt-4.1`, `gpt-5`, …), reset 00:00 UTC. The catch: OpenAI uses those
inputs/outputs for training — fine for this benign eval, your call in general.
A full `n=200 r=2` baseline is ~0.3 M tokens, so the mini quota runs **~30
models/day for $0**. This is exactly what `run_baselines.sh` is budgeted for.

```bash
export OPENAI_API_KEY=sk-...
# default BASE_URL/KEY_VAR already point at OpenAI, so:
MODELS="gpt-4.1-mini" N=40 R=1 ./eval/run_baselines.sh     # quick smoke
./eval/run_baselines.sh                                     # full mini+big set
```

### Together AI — free via *$25 signup credit* (use for open models)
New accounts get **$25 in credits** (200+ open models: Qwen, Llama, DeepSeek,
gpt-oss, Kimi, GLM, MiniMax). At $0.05–$1.40 / 1 M tokens that's hundreds of
runs. This is how every open-model row in `eval/README.md` was produced.

```bash
export TOGETHER_API_KEY=...
BASE_URL=https://api.together.xyz/v1 KEY_VAR=TOGETHER_API_KEY \
  MODELS="Qwen/Qwen3.5-9B meta-llama/Llama-3.3-70B-Instruct-Turbo" \
  ./eval/run_baselines.sh
# or a single model directly:
vf-eval channel-switching-eval -m Qwen/Qwen3.5-9B \
  -b https://api.together.xyz/v1 -k TOGETHER_API_KEY -n 200 -r 2 -s
```

### Anthropic — the third provider (no free tier; optional)
The Claude rows used Anthropic via an OpenAI-compatible endpoint. There's no
free daily quota; skip unless you have credits. Set `LIVE_BASE_URL`/`LIVE_MODEL`
(agent-layer) or `-b/-k/-m` (eval) to your Claude-compatible gateway.

**Bottom line:** yes to both — **OpenAI complimentary tokens for the
closed-model rows, Together's $25 for the open-model rows.** Between them the
entire 20-model table reproduces for ~$0–5. "Free ChatGPT credits" specifically
(a Plus subscription) won't work; the free *API* path above is the substitute.

---

## A. eval/ — the measurement environment

```bash
# A0. offline — no key, proves the scorer/detectors (65/65 checks)
cd eval && python3 test_offline.py && cd ..

# A1. live baseline (OpenAI free tier) — emits the README table
./eval/run_baselines.sh

# A2. positive controls — prove BOTH detectors fire (behavioral + covert)
./eval/run_poscontrol.sh

# A3. read the coupling profile off any saved run
python3 eval/analysis.py eval/outputs/evals/channel-switching-eval--<model>/<run-id>/

# A4. audit tools (treat any in-the-wild signature as suspect until these agree)
python3 eval/rescore.py      <run-dir>   # recompute metrics under the current detector
python3 eval/blind_judge.py  <run-dir>   # independent-vendor judge (needs BLIND_KEY)
python3 eval/judge_sweep.py  <run-dir>   # 4-class rubric, two judges (needs JUDGE_KEY)
```
The detector audit (`eval/README.md` "Detector audit") is the part worth
reading before you trust a nonzero lexical signature — it's how the v1→v2.1
false-positive retraction happened.

## B. toy/ — in-silico validation (no key, ~3 min CPU)

```bash
cd toy && python3 train_sweep.py        # --quick for a 1-seed smoke test
```
Shows alignment pressure *creating* eval-awareness from zero and the coupling
moving to the reasoning channel as behavior is driven flat. Honest boundary
(over-pressure extinguishes the behavior) is reported in `toy/README.md`.

## C. experiments/ — steering (local model, no provider key)

```bash
python3 experiments/tier2_adversarial.py    # cloaked-rebel; ~5 min GPU / ~30 min CPU
python3 experiments/tier2_pe_vs_projection.py
python3 experiments/tier3_efficiency.py      # the published honest NEGATIVE (1/3 pass)
```
First run downloads SmolLM2-1.7B. Results land in `experiments/*_results.json`.

## D. The bound + meter on your own agent — ships as its own repo: scry

The agent layer (drop-in bound + meter for any harness) lives in its own repo,
[AnthonE/scry](https://github.com/AnthonE/scry). Clone it and run from there:

```bash
git clone https://github.com/AnthonE/scry && cd scry
python3 test_harnesses.py                    # 42 offline checks, no deps, no key
OPENAI_API_KEY=... python3 live_validation.py        # live Y/M/D capture (LIVE_* envs override)
OPENAI_API_KEY=... python3 live_tool_validation.py   # same, tool-calling path
```
Start at scry's `HARNESSES.md` (adapter matrix) and `FIELD-TEST.md`.
Stubbed harnesses (Excalibur, OpenClaw meter-side) raise with the exact log
sample they need rather than guessing — that's intended.

## E. probes/ — the cantrip 207 falsifier (analysis half)

```bash
python3 probes/cantrip_loom_207_probe.py             # synthetic self-test, no Elixir/key
python3 probes/cantrip_loom_207_probe.py <loom.jsonl># point at a REAL cantrip loom dump
```
Status: the analysis consumer is built + self-tested. The cross-harness
falsifier still needs **one budgeted run** — a real model under a temptation
circle in cantrip (Elixir), exported to JSONL. `fake_llm` cannot exhibit
channel-switching, so a real key/model is required for the real run.

## F. compute_p0_deploy_af.py — the deploy-side p₀ (stdlib)

```bash
# download json_outputs.zip (~901 MB) from the alignment-faking paper's Drive:
# https://drive.google.com/drive/folders/1s-IcsWO5I3caX21yPbn7N_oTaJlU4Mz1
python3 compute_p0_deploy_af.py /path/to/json_outputs.zip
```
Reproduces P(AF reasoning | deploy, post-RL) = 0.331 from the public N=172 data.

---

## 3. "Test them all like us" — the full sequence

A clean machine to a complete reproduction, free-tier first:

```bash
# ── free + offline: confirm the instrument before spending a token ──
source .venv/bin/activate
python3 validate_release.py                 # master gate: every offline suite + adapter contract
cd eval && python3 test_offline.py && cd .. # 65/65
cd toy  && python3 train_sweep.py  && cd .. # in-silico signature
# agent layer (bound+meter, 42/42) is its own repo — run it in scry: github.com/AnthonE/scry
python3 probes/cantrip_loom_207_probe.py    # falsifier self-test

# ── free-tier live: the model table (OpenAI complimentary + Together $25) ──
export OPENAI_API_KEY=...   TOGETHER_API_KEY=...
./eval/run_baselines.sh                                         # OpenAI rows (free quota)
BASE_URL=https://api.together.xyz/v1 KEY_VAR=TOGETHER_API_KEY \
  MODELS="Qwen/Qwen3.5-9B meta-llama/Llama-3.3-70B-Instruct-Turbo gpt-oss-20b" \
  ./eval/run_baselines.sh                                       # open rows ($25 credit)
./eval/run_poscontrol.sh                                        # both detectors bracketed
# agent-layer live_validation runs in the scry repo (github.com/AnthonE/scry), not here

# ── optional / heavier ──
python3 experiments/tier2_adversarial.py    # GPU recommended
# compute_p0_deploy_af.py + the cantrip real run: see §F / §E (data download / Elixir)
```

**Budget:** offline = $0. The full 20-model live table = **~$0–5** (OpenAI rows
inside the free daily quota; open rows a few dollars of the Together credit).
Run the mini-group models first — they draw on the 10 M/day quota, not the 1 M.

---

## Troubleshooting

- **`vf-eval: command not found`** → `pip install verifiers` and activate the venv.
- **`No module named 'channel_switching_eval'`** → `pip install -e eval` once.
- **`validate_release.py` shows eval as SKIP** → `verifiers` isn't installed in
  the active interpreter; that's an honest SKIP, not a failure.
- **`KeyError` on the API key** → `-k`/`KEY_VAR` is the *variable name*; the var
  itself must be exported (`export OPENAI_API_KEY=...`).
- **Reasoning model emits no `<think>`** → expected; the scorer reads the
  `reasoning_content` field, and `analysis.py` flags Y_r non-interpretable below
  90% scratchpad adherence (`P(think)`). Low I(C;Y_r) there means *unmeasured*,
  not *zero* — see `eval/README.md` "Honest scope."
- **Nonzero in-the-wild lexical signature** → don't trust it until `blind_judge.py`
  *and* `judge_sweep.py` agree. The instrument has caught its own false positive before.
