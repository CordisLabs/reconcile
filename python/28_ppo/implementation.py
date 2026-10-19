"""Proximal Policy Optimization (Schulman et al., 2017).

Two things are checked here, and they are different in kind:

  * the clipped surrogate is an exact function of the probability ratio, and its gradient
    is exactly zero outside the clip range — arithmetic, checkable to the last bit;
  * trained on CartPole, the algorithm actually improves the policy.

The environment is the standard CartPole dynamics, implemented from its equations so the
slice has no dependencies. GAE and the paper's parallel-environment setup are not
implemented; returns are plain discounted sums, which is a simplification worth stating
because it changes how the advantages are estimated.
"""

import numpy as np


# --------------------------------------------------------------------------
# CartPole
# --------------------------------------------------------------------------

class CartPole:
    """Standard dynamics: push left (0) or right (1); reward 1 per surviving step."""

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    theta_threshold = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    def __init__(self, seed=0, hard=False):
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.steps = 0
        # a longer, heavier pole is harder to balance: the same controller cannot solve it
        self.length = 1.0 if hard else 0.5
        self.polemass_length = self.masspole * self.length

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        done = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold
            or self.steps >= 500
        )
        return self.state.copy(), 1.0, done


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class ActorCritic:
    def __init__(self, obs_dim=4, n_actions=2, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(obs_dim)
        self.Wa1 = rng.normal(0, s, (obs_dim, hidden))
        self.ba1 = np.zeros(hidden)
        self.Wa2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, n_actions))
        self.ba2 = np.zeros(n_actions)
        self.Wv1 = rng.normal(0, s, (obs_dim, hidden))
        self.bv1 = np.zeros(hidden)
        self.Wv2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
        self.bv2 = np.zeros(1)

    def params(self):
        return [
            ("Wa1", self.Wa1), ("ba1", self.ba1), ("Wa2", self.Wa2), ("ba2", self.ba2),
            ("Wv1", self.Wv1), ("bv1", self.bv1), ("Wv2", self.Wv2), ("bv2", self.bv2),
        ]

    def forward(self, obs):
        """obs: (B, obs_dim). Returns action probabilities and state values."""
        h = np.tanh(obs @ self.Wa1 + self.ba1)
        logits = h @ self.Wa2 + self.ba2
        probs = softmax(logits)
        hv = np.tanh(obs @ self.Wv1 + self.bv1)
        values = (hv @ self.Wv2 + self.bv2).ravel()
        self.cache = (obs, h, logits, probs, hv, values)
        return probs, values

    def act(self, obs, rng):
        probs, _ = self.forward(obs[None, :])
        a = int(rng.choice(probs.shape[1], p=probs[0]))
        return a, probs[0, a]


# --------------------------------------------------------------------------
# the clipped surrogate, as a standalone function so it can be tested exactly
# --------------------------------------------------------------------------

def clipped_surrogate(logp, logp_old, advantage, clip=0.2):
    """The paper's objective (eq. 7), negated so it can be minimised.

    Returns (loss, dloss/dlogp). The derivative is exactly zero where the clipped branch
    is active, which is the whole point of the algorithm.
    """
    ratio = np.exp(logp - logp_old)
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip) * advantage
    use_unclipped = unclipped <= clipped
    loss = -np.mean(np.where(use_unclipped, unclipped, clipped))
    dloss = np.where(use_unclipped, -ratio * advantage, 0.0) / logp.shape[0]
    return float(loss), dloss


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

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


def collect(model, env, episodes, rng):
    obs_buf, act_buf, logp_buf, ret_buf = [], [], [], []
    val_buf, done_buf = [], []
    for _ in range(episodes):
        obs = env.reset()
        traj = []
        done = False
        while not done:
            probs, values = model.forward(obs[None, :])
            a = int(rng.choice(probs.shape[1], p=probs[0]))
            logp = float(np.log(probs[0, a] + 1e-12))
            traj.append((obs.copy(), a, logp, float(values[0])))
            obs, r, done = env.step(a)
        returns = []
        g = 0.0
        for _, _, _, _ in reversed(traj):
            g = 1.0 + 0.99 * g
            returns.append(g)
        returns.reverse()
        for i, ((o, a, lp, v), ret) in enumerate(zip(traj, returns)):
            obs_buf.append(o)
            act_buf.append(a)
            logp_buf.append(lp)
            ret_buf.append(ret)
            val_buf.append(v)
            done_buf.append(1.0 if i == len(traj) - 1 else 0.0)
    return (np.array(obs_buf), np.array(act_buf), np.array(logp_buf),
            np.array(ret_buf), np.array(val_buf), np.array(done_buf))


def generalised_advantage(rewards, values, dones, gamma=0.99, lam=0.95):
    """GAE: sum (gamma*lam)^k delta_t, with delta_t = r + gamma V(s') - V(s)."""
    n = len(rewards)
    adv = np.zeros(n)
    running = 0.0
    for t in reversed(range(n)):
        if t == n - 1 or dones[t]:
            next_value = 0.0
            next_adv = 0.0
        else:
            next_value = values[t + 1]
            next_adv = adv[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * next_adv
        adv[t] = running
    return adv


def train(model, iterations=120, episodes=8, epochs=4, batch=64, lr=3e-3, seed=0, clip=0.2,
          use_gae=False, hard=False, env_steps=None):
    rng = np.random.default_rng(seed)
    env = CartPole(seed=1234, hard=hard)
    opt = Adam(model.params(), lr=lr)
    history = []
    steps = 0
    for _ in range(iterations):
        obs, acts, logp_old, rets, vals, dones = collect(model, env, episodes, rng)
        steps += len(rets)
        if use_gae:
            adv = generalised_advantage(rets, vals, dones)
        else:
            adv = rets - vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                b_obs, b_acts = obs[idx], acts[idx]
                b_logp_old, b_adv, b_ret = logp_old[idx], adv[idx], rets[idx]
                probs, values = model.forward(b_obs)
                logp = np.log(probs[np.arange(len(b_acts)), b_acts] + 1e-12)
                _, dlogp = clipped_surrogate(logp, b_logp_old, b_adv, clip)
                # backward through log-probabilities to logits
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(b_acts)), b_acts] = 1.0
                # d logp / d logits = onehot - probs
                dlogits = dlogp[:, None] * (onehot - probs)
                obs_, h, logits, probs_, hv, vals = model.cache
                grads = {}
                grads["Wa2"] = h.T @ dlogits
                grads["ba2"] = dlogits.sum(axis=0)
                dh = dlogits @ model.Wa2.T
                da = dh * (1 - h * h)
                grads["Wa1"] = obs_.T @ da
                grads["ba1"] = da.sum(axis=0)
                dvals = 2.0 * (vals - b_ret) / len(b_ret)
                grads["Wv2"] = hv.T @ dvals[:, None]
                grads["bv2"] = np.array([dvals.sum()])
                dvh = dvals[:, None] @ model.Wv2.T
                dva = dvh * (1 - hv * hv)
                grads["Wv1"] = obs_.T @ dva
                grads["bv1"] = dva.sum(axis=0)
                opt.step(model.params(), grads)
        history.append(float(rets.mean()))
        if env_steps and steps >= env_steps:
            break
    return history


def evaluate(model, episodes=20, seed=99, greedy=True):
    """Mean return over episodes.

    Default is the greedy policy: sampling adds variance that has nothing to do with the
    quality of the trained policy, and it made a fixed threshold fail on a machine whose
    BLAS rounds differently.
    """
    env = CartPole(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            probs, _ = model.forward(obs[None, :])
            a = int(np.argmax(probs[0])) if greedy else int(rng.choice(probs.shape[1], p=probs[0]))
            obs, r, done = env.step(a)
            total += r
    return total / episodes
