# fixtures/ — real session logs, conformance-checked by Layer B

`validate_release.py` Layer B proves each adapter logs **Y/M/D** on a *canonical*
sample. That proves the adapter's shape — not that a **real runtime** (the live
game server, Hermes, a cantrip loom, an OpenClaw deploy) actually emits that
shape. This directory is the bridge: drop a real log in and Layer B conformance-
checks the real thing.

## Layout

```
fixtures/
  turns/<name>.jsonl        already-converted Turn records — one asdict(Turn) per
                            line (e.g. live_session.py output). Loaded directly.
  <adapter>/<name>.json     a RAW harness export for a registered adapter
                            (cantrip | verifiers | openai_messages | …). One
                            session per file; converted via convert(<adapter>).
```

## The contract a fixture must pass (same as the canonical samples)

- at least one turn carries a **named Y** (the bound),
- every **meterable** turn (has reasoning M) carries **both** M and D — the
  Paper 207 split; a turn that collapses them can hold the bound but not the meter,
- `channel_profile()` computes the coupling profile without error.

A fixture that fails is a **FAIL** — the real harness isn't logging what the meter
needs, which is exactly what we want surfaced before shipping. A `[STUB]` adapter
(Excalibur, OpenClaw) gains a real fixture the moment someone drops one session
log here: that log both promotes the stub to a working adapter *and* becomes its
regression test, in one move.

## Privacy

Real logs can carry model outputs and prompt content. `.gitignore` here excludes
the log data by default — the *slot* ships, the *data* stays local unless you
deliberately add a sanitized fixture.
