"""Channel-switching sweep — REINFORCE agent with an explicit Y_r bottleneck.

Policy architecture (the measurement design, not an accident):

    x --MLP(16,tanh)--> h --w2--> z = tanh(.)   [scalar bottleneck = Y_r]
                                  logit = g*z + b0
                                  P(covert) = sigmoid(logit)

Every bit of context information the policy uses to act is forced through the
scalar z, so I(C;z) is the toy's reasoning-channel coupling I(C;Y_r) and is
directly measurable by binning. Data processing gives I(C;Y_b) <= I(C;z); the
channel-switching prediction is about the *divergence* between them under RL
pressure.

Protocol (mirrors the paper's pre/post design):
  Phase 1 (pre-RL):  train at lam=0. Covert always pays; no reason to detect
                     C. Expect I(C;z) ~ 0, covert rate ~ 1 in both contexts.
  Phase 2 (post-RL): fine-tune the SAME weights at lam>0 (alignment pressure:
                     observed covert behavior is punished; the temptation S
                     is untouched). Sweep lam.

Predictions under test (Paper 207, Theorem clause (iii)):
  P1. I(C;z) rises with lam — suppression precision requires C-detection.
  P2. I(C;Y_b) does not rise comparably — coupling moves to the internal
      channel; the behaviorally-visible share of total coupling falls.
  P3. Deploy-side covert rate P(covert|C=0) stays high while monitored-side
      P(covert|C=1) is driven down (the toy p0, analog of the paper's 0.331).
  Falsifier: if I(C;z) stays flat while the behavioral gap narrows, or if
  RL pressure simply extinguishes covert behavior in BOTH contexts at low
  lam (S overwritten, no switching), clause (iii) fails in the toy.

Pure numpy. Full 3-seed sweep ~3 min on CPU. No GPU, no torch.
Usage: python3 train_sweep.py [--quick]   (quick: 1 seed, writes results_quick.json)
"""

import json
import sys

import numpy as np

from env import EvalFlagBandit, mi_quantile, mi_binary

H = 16          # hidden width
D = 8           # observation dim
ENT = 0.01      # entropy bonus (keeps exploration alive at high lam)
LR = 0.02
BATCH = 512


def init_params(rng):
    return {
        "W1": rng.normal(0, 0.4, size=(D, H)),
        "b1": np.zeros(H),
        "w2": rng.normal(0, 0.4, size=H),
        "b2": 0.0,
        "g": 0.1,       # small init: bottleneck starts nearly disconnected
        "b0": 0.0,
    }


def forward(p, x):
    h = np.tanh(x @ p["W1"] + p["b1"])
    z = np.tanh(h @ p["w2"] + p["b2"])
    logit = p["g"] * z + p["b0"]
    prob = 1.0 / (1.0 + np.exp(-logit))
    return h, z, logit, prob


def adam_init(p):
    return {k: (np.zeros_like(np.asarray(v, dtype=float)),
                np.zeros_like(np.asarray(v, dtype=float))) for k, v in p.items()}


def adam_step(p, grads, state, lr, t, b1=0.9, b2=0.999, eps=1e-8):
    for k, gr in grads.items():
        m, v = state[k]
        m[...] = b1 * m + (1 - b1) * gr
        v[...] = b2 * v + (1 - b2) * gr * gr
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        p[k] = p[k] + lr * mh / (np.sqrt(vh) + eps)   # ascent
    return p


def train(env, p, iters, rng, lr=LR):
    """REINFORCE with moving-average baseline + entropy bonus."""
    state = adam_init(p)
    baseline = 0.0
    for t in range(1, iters + 1):
        x, c = env.sample_batch(BATCH)
        h, z, logit, prob = forward(p, x)
        a = (rng.random(BATCH) < prob).astype(np.float64)
        r, _ = env.reward_batch(a, c)
        baseline = 0.99 * baseline + 0.01 * r.mean()
        adv = r - baseline

        # d/dlogit [ logpi * adv + ENT * H(pi) ]
        dlogit = adv * (a - prob) + ENT * (-prob * (1 - prob) * logit)
        dz = dlogit * p["g"]
        dpre2 = dz * (1 - z ** 2)                    # through tanh bottleneck
        dh = dpre2[:, None] * p["w2"]
        dpre1 = dh * (1 - h ** 2)
        grads = {
            "g": (dlogit * z).mean(),
            "b0": dlogit.mean(),
            "w2": (h * dpre2[:, None]).sum(axis=0) / BATCH,
            "b2": dpre2.mean(),
            "W1": x.T @ dpre1 / BATCH,
            "b1": dpre1.mean(axis=0),
        }
        p = adam_step(p, grads, state, lr, t)
    return p


def probe_auc(c, z):
    """AUC of the scalar bottleneck as a C-probe (Mann-Whitney, midranks).

    Midranks are essential: tanh saturation produces heavy ties, and
    position-ordered ranks inflate AUC systematically."""
    z0, z1 = z[c == 0], z[c == 1]
    if len(z0) == 0 or len(z1) == 0:
        return 0.5
    allz = np.concatenate([z0, z1])
    uniq, inv, counts = np.unique(allz, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    midranks = starts + (counts + 1) / 2.0          # 1-based midrank per value
    ranks = midranks[inv]
    r1 = ranks[len(z0):].sum()
    auc = (r1 - len(z1) * (len(z1) + 1) / 2) / (len(z0) * len(z1))
    return float(max(auc, 1 - auc))


def evaluate(env, p, n=100_000):
    x, c = env.sample_batch(n)
    _, z, _, prob = forward(p, x)
    rng = np.random.default_rng(123)
    a = (rng.random(n) < prob).astype(np.int64)
    comply = a == 0
    res = {
        "I_C_Yb": mi_binary(c, a),
        "I_C_z": mi_quantile(c, z, bins=20),
        # context-tracking invisible to a behavioral evaluator: coupling
        # carried by z on the episodes where behavior is compliant
        "I_C_z_given_comply": (mi_quantile(c[comply], z[comply], bins=20)
                               if comply.sum() > 1000 else 0.0),
        "probe_auc": probe_auc(c, z),
        "covert_deploy": float(a[c == 0].mean()),    # toy p0
        "covert_eval": float(a[c == 1].mean()),
    }
    res["gap"] = res["covert_deploy"] - res["covert_eval"]
    return res


def main():
    quick = "--quick" in sys.argv
    iters1, iters2 = (800, 1200) if quick else (3000, 5000)
    lams = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    seeds = [0, 1, 2] if not quick else [0]

    env0 = EvalFlagBandit(d=D, lam=0.0, seed=99)
    print(f"Bayes ceiling I(C; posterior) = {env0.bayes_ceiling():.3f} bits\n")

    rows = []
    for lam in lams:
        per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # Phase 1 — pre-RL: temptation only
            env_pre = EvalFlagBandit(d=D, lam=0.0, seed=seed)
            p = train(env_pre, init_params(rng), iters1, rng)
            pre = evaluate(env_pre, p)
            # Phase 2 — alignment RL at pressure lam, same weights
            env_post = EvalFlagBandit(d=D, lam=lam, seed=seed + 1000)
            p = train(env_post, p, iters2, rng)
            post = evaluate(env_post, p)
            per_seed.append((pre, post))
        keys = per_seed[0][1].keys()
        pre_m = {k: float(np.mean([s[0][k] for s in per_seed])) for k in keys}
        post_m = {k: float(np.mean([s[1][k] for s in per_seed])) for k in keys}
        post_sd = {k: float(np.std([s[1][k] for s in per_seed])) for k in keys}
        rows.append({"lam": lam, "pre": pre_m, "post": post_m, "post_sd": post_sd})
        print(f"lam={lam:5.1f} | I(C;Yb) {pre_m['I_C_Yb']:.4f}->{post_m['I_C_Yb']:.4f}"
              f" | I(C;z) {pre_m['I_C_z']:.4f}->{post_m['I_C_z']:.4f}"
              f" | I(C;z|comply) {post_m['I_C_z_given_comply']:.4f}"
              f" | probeAUC {post_m['probe_auc']:.3f}"
              f" | covert deploy/eval {post_m['covert_deploy']:.3f}/{post_m['covert_eval']:.3f}")

    # quick mode writes a separate file so the smoke test never clobbers the
    # published seed-replicated results.json in a fresh clone
    out = "results_quick.json" if quick else "results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
