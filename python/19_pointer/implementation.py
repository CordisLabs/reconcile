"""Pointer Networks (Vinyals, Fortunato, Jaitly, 2015).

The model has no output vocabulary: every step produces a distribution over the input
positions. The pointer mechanism is additive attention,

    u_j = v_a . tanh(W_a h_j + U_a s_t)
    a_j = softmax_j(u_j)          (masked to positions not yet chosen)

Task: sorting. Feed n numbers, output their indices in ascending order. This is the
paper's own toy problem, and it has an exact answer, so accuracy is a real number.

The paper's cells are LSTMs; this implementation uses vanilla tanh cells. That changes
what can be compared (convergence, final accuracy) and not the mechanism under test.
"""

import numpy as np


def as_vec(x):
    """Scalar or vector input to a (dim,) row."""
    a = np.asarray(x, dtype=float).reshape(-1)
    return a


def softmax_masked(e, allowed):
    e = np.where(allowed, e, -1e9)
    e = e - e.max()
    ex = np.exp(e) * allowed
    return ex / (ex.sum() + 1e-12)


class PointerNet:
    def __init__(self, hid=32, att=32, seed=0, dim=1):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hid)
        self.H, self.A, self.dim = hid, att, dim
        self.We = rng.uniform(-s, s, (dim, hid))   # `dim` features per element
        self.Ue = rng.uniform(-s, s, (hid, hid))
        self.Wa = rng.uniform(-s, s, (hid, att))
        self.Ua = rng.uniform(-s, s, (hid, att))
        self.va = rng.uniform(-s, s, att)
        self.Ws = rng.uniform(-s, s, (dim + hid, hid))
        self.Us = rng.uniform(-s, s, (hid, hid))

    def params(self):
        return [("We", self.We), ("Ue", self.Ue), ("Wa", self.Wa), ("Ua", self.Ua),
                ("va", self.va), ("Ws", self.Ws), ("Us", self.Us)]

    def encode(self, xs):
        n = len(xs)
        hs = np.zeros((n, self.H))
        h = np.zeros(self.H)
        for j in range(n):
            h = np.tanh(as_vec(xs[j]) @ self.We + h @ self.Ue)
            hs[j] = h
        return hs, h

    def forward(self, xs, order, allowed_list=None):
        """`order` is the forced input sequence; alphas are returned per step.

        Step t attends over the input with the positions chosen at steps < t masked out.
        """
        n = len(xs)
        hs, h_last = self.encode(xs)
        s = h_last.copy()
        T = len(order)
        alphas = np.zeros((T, n))
        caches = []
        for t in range(T):
            allowed = np.ones(n, dtype=bool)
            if t > 0:
                allowed[order[:t]] = False
            xin = np.array([xs[order[t - 1]]], dtype=float).reshape(-1) if t > 0 else np.zeros(self.dim)
            pre = hs @ self.Wa + s @ self.Ua
            th = np.tanh(pre)
            e = th @ self.va
            alpha = softmax_masked(e, allowed)
            c = alpha @ hs
            x = np.concatenate([xin, c])
            s_new = np.tanh(x @ self.Ws + s @ self.Us)
            caches.append((s, x, s_new, pre, th, alpha, allowed, hs, c))
            alphas[t] = alpha
            s = s_new
        self.cache = (xs, hs, h_last, caches, order)
        return alphas

    def loss_and_grad(self, xs, order, target):
        alphas = self.forward(xs, order)
        T = len(target)
        picked = alphas[np.arange(T), target]
        loss = float(-np.mean(np.log(picked + 1e-12)))
        g = {k: np.zeros_like(v) for k, v in self.params()}

        dalpha_seed = np.zeros_like(alphas)
        dalpha_seed[np.arange(T), target] = -1.0 / (T * picked)

        xs_, hs, h_last, caches, _ = self.cache
        n = len(xs)
        dhs = np.zeros_like(hs)
        dS = np.zeros(self.H)             # gradient w.r.t. s_{t+1}; starts at 0 past the end

        for t in reversed(range(T)):
            s_prev, x, s_new, pre, th, alpha, allowed, hsc, c = caches[t]
            # 1. through the cell
            da = dS * (1.0 - s_new * s_new)
            g["Ws"] += np.outer(x, da)
            g["Us"] += np.outer(s_prev, da)
            dx = da @ self.Ws.T
            dS_prev = da @ self.Us.T
            dc = dx[self.dim:]

            # 2. attention: two paths meet here (the CE on alpha, and c = alpha @ hs)
            dalpha = dalpha_seed[t] + hsc @ dc
            de = alpha * (dalpha - (alpha * dalpha).sum())
            g["va"] += th.T @ de
            dth = np.outer(de, self.va)
            dpre = dth * (1.0 - th * th)
            g["Wa"] += hsc.T @ dpre
            g["Ua"] += np.outer(s_prev, dpre.sum(axis=0))
            dhs += dpre @ self.Wa.T + np.outer(alpha, dc)
            dS_prev += (dpre @ self.Ua.T).sum(axis=0)
            dS = dS_prev

        # 3. the decoder starts from the last encoder state
        dhs[n - 1] += dS

        # 4. encoder: reverse-time over positions
        carry = np.zeros(self.H)
        h_prev = np.zeros((n + 1, self.H))     # h_prev[j] = h_{j-1}, zeros at j = 0
        h = np.zeros(self.H)
        for j in range(n):
            h_prev[j] = h
            h = np.tanh(as_vec(xs[j]) @ self.We + h @ self.Ue)
        for j in reversed(range(n)):
            a = dhs[j] + carry
            dpre = a * (1.0 - hs[j] * hs[j])
            g["We"] += np.outer(as_vec(xs[j]), dpre)
            g["Ue"] += np.outer(h_prev[j], dpre)
            carry = dpre @ self.Ue.T
        return loss, g


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
# sorting task
# --------------------------------------------------------------------------

def make_example(rng, n):
    xs = rng.random(n)
    order = np.argsort(xs)
    return xs, list(order)


def train(model, steps, n=6, batch=16, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    losses = []
    for _ in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        total = 0.0
        for _ in range(batch):
            xs, order = make_example(rng, n)
            # teacher forcing: the decoder is fed the true order, and the same
            # sequence also defines which positions are masked at each step
            loss, g = model.loss_and_grad(xs, order, order)
            total += loss
            for k in acc:
                acc[k] += g[k]
        for k in acc:
            acc[k] /= batch
        opt.step(model.params(), acc)
        losses.append(total / batch)
    return losses


def predict(model, xs):
    n = len(xs)
    chosen = []
    for t in range(n):
        # one extra dummy entry at the end: step t only reads positions < t
        alphas = model.forward(xs, chosen + [0])
        j = int(np.argmax(alphas[-1]))
        chosen.append(j)
    return chosen


def accuracy(model, n=6, samples=100, seed=1):
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(samples):
        xs, order = make_example(rng, n)
        if predict(model, xs) == list(order):
            correct += 1
    return correct / samples
