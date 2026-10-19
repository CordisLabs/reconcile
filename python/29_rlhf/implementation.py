"""Deep Reinforcement Learning from Human Feedback (Christiano et al., 2017).

The pipeline in the paper is: collect pairwise preferences, fit a reward model with a
Bradley-Terry loss, then optimise a policy against that reward. This slice implements the
middle step — the part that turns comparisons into a scalar reward — and measures it
directly, because that is the step with a clean, exact judge.

The preference data comes from a known hidden reward, so "how well did we recover it" is
a number rather than a judgement. Labels are sampled through the same Bradley-Terry model
with a temperature, which puts a ceiling on achievable accuracy and makes the comparison
honest: a model at 75% when the Bayes rate is 80% is doing well, and a model at 75% when
the labels are noiseless is not.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bradley_terry_prob(diff):
    """P(a preferred over b) = sigmoid(r_a - r_b)."""
    return sigmoid(diff)


def sample_preference(r_a, r_b, temperature, rng):
    """Sample a label from Bradley-Terry; returns 1 if a is preferred."""
    p = sigmoid((r_a - r_b) / temperature)
    return 1 if rng.random() < p else 0


class RewardModel:
    """Small MLP from a feature vector to a scalar reward."""

    def __init__(self, d, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(d), (d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
        self.b2 = np.zeros(1)

    def params(self):
        return [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return float((h @ self.W2 + self.b2)[0]), h

    def loss_and_grad(self, a, b, label):
        """Bradley-Terry negative log-likelihood for one pair.

        label == 1 means a is preferred. The loss is -log sigmoid(r_a - r_b), which is
        convex in the difference and gives the simple gradient below.
        """
        ra, ha = self.forward(a)
        rb, hb = self.forward(b)
        diff = ra - rb
        # d(-log sigmoid(diff))/d(diff) = sigmoid(diff) - 1 = -sigmoid(-diff)
        if label == 1:
            ddiff = sigmoid(diff) - 1.0
        else:
            ddiff = sigmoid(diff)          # -log(1 - sigmoid(diff))
        loss = -np.log(sigmoid(diff) if label == 1 else sigmoid(-diff) + 1e-300)

        grads = {}
        # r = tanh(x W1 + b1) @ W2 + b2
        dh_a = self.W2[:, 0] * ddiff
        dh_b = self.W2[:, 0] * (-ddiff)
        grads["W2"] = (ha * ddiff)[:, None] + (hb * (-ddiff))[:, None]
        grads["b2"] = np.array([ddiff - ddiff])
        grads["W1"] = np.outer(a, dh_a * (1 - ha * ha)) + np.outer(b, dh_b * (1 - hb * hb))
        grads["b1"] = dh_a * (1 - ha * ha) + dh_b * (1 - hb * hb)
        return float(loss), grads


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params}
        self.v = {k: np.zeros_like(v) for k, v in params}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, p in params:
            gr = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gr
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gr * gr
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# --------------------------------------------------------------------------
# synthetic preference data
# --------------------------------------------------------------------------

def hidden_reward(x, w):
    return float(x @ w)


def make_dataset(n_pairs, d, w, temperature, rng):
    """Random feature vectors, labelled by a noisy Bradley-Terry comparison."""
    data = []
    for _ in range(n_pairs):
        a = rng.normal(size=d)
        b = rng.normal(size=d)
        ra = hidden_reward(a, w)
        rb = hidden_reward(b, w)
        label = sample_preference(ra, rb, temperature, rng)
        data.append((a, b, label))
    return data


def empirical_bayes_rate(d, w, temperature, rng, n=20000):
    """Accuracy of the true reward on this noise level — the ceiling for any model."""
    correct = 0
    for _ in range(n):
        a = rng.normal(size=d)
        b = rng.normal(size=d)
        ra = hidden_reward(a, w)
        rb = hidden_reward(b, w)
        label = sample_preference(ra, rb, temperature, rng)
        if (ra > rb) == (label == 1):
            correct += 1
    return correct / n


def train(model, data, steps=400, lr=3e-3, seed=0, log_every=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    losses = []
    for step in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        batch = [data[int(rng.integers(0, len(data)))] for _ in range(16)]
        total = 0.0
        for a, b, label in batch:
            loss, g = model.loss_and_grad(a, b, label)
            total += loss
            for k in acc:
                acc[k] += g[k]
        for k in acc:
            acc[k] /= len(batch)
        opt.step(model.params(), acc)
        losses.append(total / len(batch))
    return losses


def preference_accuracy(model, data):
    correct = 0
    for a, b, label in data:
        ra, _ = model.forward(a)
        rb, _ = model.forward(b)
        pred = 1 if ra > rb else 0
        if pred == label:
            correct += 1
    return correct / len(data)


def candidate_set_accuracy(model, w, d, k=5, n=200, rng=None):
    """How often the learned reward picks the oracle-best candidate out of k."""
    rng = rng or np.random.default_rng(0)
    hits = 0
    for _ in range(n):
        cands = [rng.normal(size=d) for _ in range(k)]
        true_scores = [hidden_reward(c, w) for c in cands]
        learned = [model.forward(c)[0] for c in cands]
        if int(np.argmax(learned)) == int(np.argmax(true_scores)):
            hits += 1
    return hits / n
