"""Sample efficiency with and without generalised advantage estimation.

PPO as used everywhere today uses GAE; the original paper's experiments did too, but the
algorithm works with plain discounted returns. The question here is whether GAE buys
anything measurable on this task: how many environment steps each variant needs to reach
the evaluation threshold.

Run: ~/.venvs/gpu/bin/python compare_gae.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import ActorCritic, CartPole, evaluate, train  # noqa: E402


def steps_to_threshold(use_gae, hard=False, threshold=475.0, max_iters=200, seed=0):
    """Train in blocks, checking the evaluation score between blocks."""
    model = ActorCritic(seed=seed)
    rng = np.random.default_rng(seed)
    env = CartPole(seed=1234, hard=hard)
    from implementation import Adam, collect, generalised_advantage, clipped_surrogate
    opt = Adam(model.params(), lr=3e-3)
    steps = 0
    for _ in range(max_iters):
        obs, acts, logp_old, rets, vals, dones = collect(model, env, 8, rng)
        steps += len(rets)
        adv = generalised_advantage(rets, vals, dones) if use_gae else rets - vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        for _ in range(4):
            order = rng.permutation(n)
            for start in range(0, n, 64):
                idx = order[start:start + 64]
                probs, values = model.forward(obs[idx])
                b_acts = acts[idx]
                logp = np.log(probs[np.arange(len(b_acts)), b_acts] + 1e-12)
                _, dlogp = clipped_surrogate(logp, logp_old[idx], adv[idx], 0.2)
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(b_acts)), b_acts] = 1.0
                dlogits = dlogp[:, None] * (onehot - probs)
                h, hv = None, None
                obs_, h_, logits_, probs_, hv_, vals_ = model.cache
                grads = {}
                grads["Wa2"] = h_.T @ dlogits
                grads["ba2"] = dlogits.sum(axis=0)
                dh = dlogits @ model.Wa2.T
                da = dh * (1 - h_ * h_)
                grads["Wa1"] = obs_.T @ da
                grads["ba1"] = da.sum(axis=0)
                dvals = 2.0 * (vals_ - rets[idx]) / len(idx)
                grads["Wv2"] = hv_.T @ dvals[:, None]
                grads["bv2"] = np.array([dvals.sum()])
                dvh = dvals[:, None] @ model.Wv2.T
                dva = dvh * (1 - hv_ * hv_)
                grads["Wv1"] = obs_.T @ dva
                grads["bv1"] = dva.sum(axis=0)
                opt.step(model.params(), grads)
        score = evaluate(model, episodes=10, seed=99)
        if score >= threshold:
            return steps, score
    return None, evaluate(model, episodes=10, seed=99)


def main():
    out = []
    for hard in (False, True):
        for use_gae in (False, True):
            runs = [steps_to_threshold(use_gae, hard=hard, max_iters=120, seed=s)[0]
                    for s in range(5)]
            ok = [r for r in runs if r is not None]
            row = {"hard": hard, "gae": use_gae,
                   "reached": f"{len(ok)}/5",
                   "median_steps": int(np.median(ok)) if ok else None,
                   "runs": runs}
            print(json.dumps(row), flush=True)
            out.append(row)
    with open(os.path.join(HERE, "results_gae.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
