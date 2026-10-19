"""Order Matters: Sequence to Sequence for Sets (Vinyals, Bengio, Kudlur, 2015).

The paper's point is architectural: when the input is a *set*, the encoder must not
depend on the order in which the elements arrive. Two encoders are implemented here so
that the difference can be measured rather than asserted:

  * "set"  — a shared per-element map, pooled by an attention-weighted sum. Permuting the
             input permutes the per-element representations and leaves the pooled
             context unchanged: exact invariance.
  * "lstm" — a recurrent encoder over the arrival order. Permuting the input changes the
             final state, so the pooled context changes. This is the failure mode the
             paper describes.

The decoder is the pointer mechanism from the previous slice, so the task can be sorting:
the model emits positions in ascending order of value.
"""

import numpy as np


def softmax_masked(e, allowed):
    e = np.where(allowed, e, -1e9)
    e = e - e.max()
    ex = np.exp(e) * allowed
    return ex / (ex.sum() + 1e-12)


class SetPointer:
    def __init__(self, hid=32, att=32, encoder="set", seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hid)
        self.H, self.A, self.encoder = hid, att, encoder
        self.We = rng.uniform(-s, s, (1, hid))      # shared per-element map
        self.be = np.zeros(hid)
        self.Ue = rng.uniform(-s, s, (hid, hid))    # only used by the lstm encoder
        self.Upool = rng.uniform(-s, s, (hid, 1))   # pooling score
        self.Wa = rng.uniform(-s, s, (hid, att))
        self.Ua = rng.uniform(-s, s, (hid, att))
        self.va = rng.uniform(-s, s, att)
        self.Ws = rng.uniform(-s, s, (1 + hid, hid))
        self.Us = rng.uniform(-s, s, (hid, hid))

    def params(self):
        out = [("We", self.We), ("be", self.be), ("Ue", self.Ue), ("Upool", self.Upool),
               ("Wa", self.Wa), ("Ua", self.Ua), ("va", self.va),
               ("Ws", self.Ws), ("Us", self.Us)]
        return out

    def encode(self, xs):
        n = len(xs)
        hs = np.zeros((n, self.H))
        if self.encoder == "set":
            for j in range(n):
                hs[j] = np.tanh(xs[j] * self.We[0] + self.be)
            scores = hs @ self.Upool[:, 0]
            w = np.exp(scores - scores.max())
            w /= w.sum()
            return hs, w @ hs, w
        # order-dependent baseline
        h = np.zeros(self.H)
        for j in range(n):
            h = np.tanh(xs[j] * self.We[0] + h @ self.Ue + self.be)
            hs[j] = h
        return hs, h.copy(), None

    def forward(self, xs, order):
        hs, s0, pool_w = self.encode(xs)
        s = s0.copy()
        T = len(order)
        n = len(xs)
        alphas = np.zeros((T, n))
        caches = []
        for t in range(T):
            allowed = np.ones(n, dtype=bool)
            if t > 0:
                allowed[order[:t]] = False
            xin = np.array([xs[order[t - 1]]]) if t > 0 else np.array([0.0])
            pre = hs @ self.Wa + s @ self.Ua
            th = np.tanh(pre)
            e = th @ self.va
            alpha = softmax_masked(e, allowed)
            c = alpha @ hs
            x = np.concatenate([xin, c])
            s_new = np.tanh(x @ self.Ws + s @ self.Us)
            caches.append((s, x, s_new, pre, th, alpha, hs, c))
            alphas[t] = alpha
            s = s_new
        self.cache = (xs, hs, s0, pool_w, caches, order)
        return alphas

    def loss_and_grad(self, xs, order, target):
        alphas = self.forward(xs, order)
        T, n = alphas.shape
        picked = alphas[np.arange(T), target]
        loss = float(-np.mean(np.log(picked + 1e-12)))
        g = {k: np.zeros_like(v) for k, v in self.params()}

        dalpha_seed = np.zeros_like(alphas)
        dalpha_seed[np.arange(T), target] = -1.0 / (T * picked)

        xs_, hs, s0, pool_w, caches, _ = self.cache
        dhs = np.zeros_like(hs)
        dS = np.zeros(self.H)

        for t in reversed(range(T)):
            s_prev, x, s_new, pre, th, alpha, hsc, c = caches[t]
            da = dS * (1.0 - s_new * s_new)
            g["Ws"] += np.outer(x, da)
            g["Us"] += np.outer(s_prev, da)
            dx = da @ self.Ws.T
            dS_prev = da @ self.Us.T
            dc = dx[1:]

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

        # gradient into the pooled initial state s0
        ds0 = dS.copy()

        if self.encoder == "set":
            # s0 = w @ hs with w = softmax(hs Upool)
            score = hs @ self.Upool[:, 0]
            dw = hs @ ds0
            g["Upool"] += hs.T @ (pool_w * (dw - (pool_w * dw).sum()))[:, None]
            dhs += np.outer(pool_w, ds0)
            # the pooling Jacobian is w_k I + w_k * u (hs_k - s0)^T, so the correction
            # is w_k (ds0 . (hs_k - s0)) u: the scalar is a dot product, the direction is u
            u = self.Upool[:, 0]
            for j in range(len(xs_)):
                dhs[j] += pool_w[j] * float(ds0 @ (hs[j] - s0)) * u
            for j in range(len(xs_)):
                dpre = dhs[j] * (1.0 - hs[j] * hs[j])
                g["We"] += xs_[j] * dpre[None, :]
                g["be"] += dpre
        else:
            carry = ds0
            h_prev = np.zeros((len(xs_), self.H))
            h = np.zeros(self.H)
            for j in range(len(xs_)):
                h_prev[j] = h
                h = np.tanh(xs_[j] * self.We[0] + h @ self.Ue + self.be)
            for j in reversed(range(len(xs_))):
                a = dhs[j] + carry
                dpre = a * (1.0 - hs[j] * hs[j])
                g["We"] += xs_[j] * dpre[None, :]
                g["be"] += dpre
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


def make_example(rng, n):
    xs = rng.random(n)
    return xs, list(np.argsort(xs))


def train(model, steps, n=6, batch=16, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    losses = []
    for _ in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        total = 0.0
        for _ in range(batch):
            xs, order = make_example(rng, n)
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
    for _ in range(n):
        alphas = model.forward(xs, chosen + [0])
        j = int(np.argmax(alphas[-1]))
        chosen.append(j)
    return chosen


def accuracy(model, n=6, samples=100, seed=1):
    """A decode counts as correct if the pointed values come out in ascending order."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(samples):
        xs, order = make_example(rng, n)
        pred = predict(model, xs)
        if sorted(pred) != list(range(n)):
            continue
        vals = [xs[j] for j in pred]
        if all(vals[i] < vals[i + 1] for i in range(n - 1)):
            ok += 1
    return ok / samples


def encode_permutation_gap(model, xs, rng, trials=20):
    """How much the pooled encoder context changes under a random permutation."""
    hs, s0, _ = model.encode(xs)
    gaps = []
    for _ in range(trials):
        perm = rng.permutation(len(xs))
        hs_p, s0_p, _ = model.encode([xs[i] for i in perm])
        gaps.append(float(np.abs(s0 - s0_p).max()))
    return max(gaps)
