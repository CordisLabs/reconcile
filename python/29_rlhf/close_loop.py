"""Close the loop: preferences -> reward model -> PPO policy.

The RLHF paper does three things in sequence: collect pairwise preferences, fit a reward
model with a Bradley-Terry loss, then optimise a policy against that learned reward. The
other slices stop at the second step. This one runs all three on a task whose true reward
is known, so the question "did the policy learn the thing the preferences were about" has
a numeric answer rather than a judgement.

Task: emit a 5-token sequence over 4 tokens. The hidden reward is a fixed linear function
of the token counts, which the "human" never states; only pairwise comparisons are given.
"""

import numpy as np

N_POS, N_TOK = 5, 4
W_TRUE = np.array([1.0, -0.5, 0.0, 2.0])
OPTIMAL = N_POS * W_TRUE.max()          # always emit the best token
RANDOM = N_POS * W_TRUE.mean()          # uniform sampling


def score(counts):
    return float(counts @ W_TRUE)


def sample_sequences(rng, n):
    toks = rng.integers(0, N_TOK, size=(n, N_POS))
    counts = np.zeros((n, N_TOK))
    for p in range(N_POS):
        counts[np.arange(n), toks[:, p]] += 1
    return toks, counts


def preference_pairs(rng, n_pairs, temperature=0.2):
    """Bradley-Terry labels from the hidden reward."""
    a_tok, a_c = sample_sequences(rng, n_pairs)
    b_tok, b_c = sample_sequences(rng, n_pairs)
    sa, sb = a_c @ W_TRUE, b_c @ W_TRUE
    p = 1.0 / (1.0 + np.exp(-(sa - sb) / temperature))
    label = (rng.random(n_pairs) < p).astype(int)      # 1 = a preferred
    return a_c, b_c, label, (a_tok, b_tok)


# --------------------------------------------------------------------------
# reward model: small MLP on counts, trained with the Bradley-Terry loss
# --------------------------------------------------------------------------

class RewardModel:
    def __init__(self, hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (N_TOK, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, counts):
        h = np.tanh(counts @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel(), h

    def loss_and_grad(self, a_c, b_c, label):
        ra, ha = self.forward(a_c)
        rb, hb = self.forward(b_c)
        diff = ra - rb
        sig = 1.0 / (1.0 + np.exp(-diff))
        n = a_c.shape[0]
        # -log sigmoid(diff) if label==1 else -log sigmoid(-diff)
        ddiff = np.where(label == 1, sig - 1.0, sig)
        loss = -np.mean(np.where(label == 1, np.log(sig + 1e-12), np.log(1 - sig + 1e-12)))
        dpre_a = ddiff[:, None] * self.W2[:, 0][None, :] * (1 - ha ** 2) / n
        dpre_b = -ddiff[:, None] * self.W2[:, 0][None, :] * (1 - hb ** 2) / n
        grads = {
            "W2": ((ha * ddiff[:, None]).sum(0) - (hb * ddiff[:, None]).sum(0))[:, None],
            "b2": np.array([0.0]),                       # the bias shifts both rewards equally
            "W1": a_c.T @ dpre_a + b_c.T @ dpre_b,
            "b1": dpre_a.sum(0) + dpre_b.sum(0),
        }
        return float(loss), grads


def train_reward(model, a_c, b_c, label, epochs=300, lr=1e-2):
    m = {k: np.zeros_like(v) for k, v in model.__dict__.items() if isinstance(v, np.ndarray) and v.ndim}
    v2 = {k: np.zeros_like(v) for k, v in m.items()}
    t = 0
    for _ in range(epochs):
        loss, g = model.loss_and_grad(a_c, b_c, label)
        t += 1
        for k, arr in (("W1", model.W1), ("b1", model.b1), ("W2", model.W2), ("b2", model.b2)):
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
            arr -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
    return loss


# --------------------------------------------------------------------------
# policy: one categorical per position, trained with PPO against the learned reward
# --------------------------------------------------------------------------

class Policy:
    def __init__(self, seed=0):
        self.theta = np.random.default_rng(seed).normal(0, 0.1, (N_POS, N_TOK))

    def probs(self):
        z = self.theta - self.theta.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def sample(self, rng, n):
        p = self.probs()
        toks = np.stack([rng.choice(N_TOK, size=n, p=p[pos]) for pos in range(N_POS)], axis=1)
        logp = sum(np.log(p[pos][toks[:, pos]] + 1e-12) for pos in range(N_POS))
        return toks, logp

    def logp_of(self, toks):
        p = self.probs()
        return sum(np.log(p[pos][toks[:, pos]] + 1e-12) for pos in range(N_POS))


def counts_of(toks):
    c = np.zeros((toks.shape[0], N_TOK))
    for pos in range(N_POS):
        c[np.arange(toks.shape[0]), toks[:, pos]] += 1
    return c


def ppo(reward_model, iters=300, batch=64, epochs=4, clip=0.2, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    policy = Policy(seed)
    baseline = 0.0
    history = []
    for it in range(iters):
        toks, logp_old = policy.sample(rng, batch)
        c = counts_of(toks)
        r, _ = reward_model.forward(c)
        adv = r - baseline
        baseline = 0.95 * baseline + 0.05 * r.mean()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(epochs):
            logp = policy.logp_of(toks)
            ratio = np.exp(logp - logp_old)
            obj = np.minimum(ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv)
            use_unclipped = (ratio * adv) <= (np.clip(ratio, 1 - clip, 1 + clip) * adv)
            # d obj / d logp, averaged over the batch
            scale = np.where(use_unclipped, ratio * adv, 0.0) / batch
            # chain to the policy parameters: d logp / d theta
            p = policy.probs()
            grad = np.zeros_like(policy.theta)
            for pos in range(N_POS):
                onehot = np.zeros((batch, N_TOK))
                onehot[np.arange(batch), toks[:, pos]] = 1.0
                dlogp = onehot - p[pos][None, :]
                grad[pos] = -(scale[:, None] * dlogp).sum(0)
            policy.theta -= lr * grad
        history.append(float(r.mean()))
    return policy, history


def metrics(policy, reward_model, rng, n=4000):
    toks, _ = policy.sample(rng, n)
    c = counts_of(toks)
    learned, _ = reward_model.forward(c)
    return {
        "true_mean": float(np.mean(c @ W_TRUE)),
        "learned_mean": float(np.mean(learned)),
        "optimal": float(OPTIMAL),
        "random": float(RANDOM),
        "best_token_freq": float(np.mean(toks == int(np.argmax(W_TRUE)))),
    }


def run(seed=0):
    rng = np.random.default_rng(seed)
    a_c, b_c, label, _ = preference_pairs(rng, 6000)
    model = RewardModel(seed=seed)
    loss = train_reward(model, a_c, b_c, label)
    # held-out preference accuracy of the learned reward
    a2, b2, l2, _ = preference_pairs(rng, 2000)
    ra, _ = model.forward(a2)
    rb, _ = model.forward(b2)
    acc = float(np.mean((ra > rb).astype(int) == l2))
    policy, hist = ppo(model, seed=seed)
    out = metrics(policy, model, rng)
    out.update({"reward_loss": round(float(loss), 4), "preference_accuracy": round(acc, 4),
                "early_reward": round(float(np.mean(hist[:10])), 3),
                "late_reward": round(float(np.mean(hist[-10:])), 3)})
    return out, hist


if __name__ == "__main__":
    out, _ = run()
    for k, v in out.items():
        print(f"{k:>22}: {v}")
