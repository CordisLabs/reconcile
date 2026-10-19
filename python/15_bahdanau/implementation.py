"""Neural Machine Translation by Jointly Learning to Align and Translate
(Bahdanau, Cho, Bengio, 2014).

The paper's idea: instead of squeezing the source into one fixed-length vector, let the
decoder soft-search the source at every step. The alignment model is an additive score,

    e_ij = v_a . tanh(W_a s_{i-1} + U_a h_j)
    a_ij = softmax_j(e_ij)
    c_i  = sum_j a_ij h_j

Two simplifications, stated plainly because they change what "faithful" means here:
the paper's encoder and decoder are GRUs, and this implementation uses vanilla tanh
recurrent cells. The object under test is the attention mechanism, not the cell.

Task: reverse a sequence. The correct alignment is known exactly (source position j maps
to target position L-1-j), so the attention can be scored against ground truth rather
than eyeballed as a heat map.
"""

import numpy as np

PAD, BOS, EOS = 0, 1, 2


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class BahdanauTranslator:
    def __init__(self, vocab, emb=8, hid=24, att=24, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hid)
        self.V, self.E, self.H, self.A = vocab, emb, hid, att
        self.Emb = rng.normal(0, 0.5, (vocab, emb))
        # bidirectional encoder
        self.Wf = rng.uniform(-s, s, (emb, hid))
        self.Uf = rng.uniform(-s, s, (hid, hid))
        self.Wb = rng.uniform(-s, s, (emb, hid))
        self.Ub = rng.uniform(-s, s, (hid, hid))
        # attention
        self.Wa = rng.uniform(-s, s, (2 * hid, att))
        self.Ua = rng.uniform(-s, s, (hid, att))
        self.va = rng.uniform(-s, s, att)
        # decoder cell: input is [embedding of previous token; context]
        self.Ws = rng.uniform(-s, s, (emb + 2 * hid, hid))
        self.Us = rng.uniform(-s, s, (hid, hid))
        self.Wo = rng.uniform(-s, s, (hid, vocab))

    def params(self):
        return [
            ("Emb", self.Emb), ("Wf", self.Wf), ("Uf", self.Uf), ("Wb", self.Wb),
            ("Ub", self.Ub), ("Wa", self.Wa), ("Ua", self.Ua), ("va", self.va),
            ("Ws", self.Ws), ("Us", self.Us), ("Wo", self.Wo),
        ]

    # -- encoder ---------------------------------------------------------
    def encode(self, src):
        """src: list of token ids. Returns annotations H (L, 2H) and caches."""
        L = len(src)
        X = self.Emb[src]                       # (L, E)
        hf = np.zeros((L, self.H))
        hb = np.zeros((L, self.H))
        h = np.zeros(self.H)
        for t in range(L):
            h = np.tanh(X[t] @ self.Wf + h @ self.Uf)
            hf[t] = h
        h = np.zeros(self.H)
        for t in reversed(range(L)):
            h = np.tanh(X[t] @ self.Wb + h @ self.Ub)
            hb[t] = h
        H = np.concatenate([hf, hb], axis=1)
        return H, (X, hf, hb)

    # -- decoder step ----------------------------------------------------
    def attention(self, s_prev, H, cache):
        """Additive attention over the L annotations. Returns c, alpha, cache."""
        pre = H @ self.Wa + s_prev @ self.Ua           # (L, A)
        th = np.tanh(pre)
        e = th @ self.va                                # (L,)
        alpha = softmax(e)
        c = alpha @ H                                   # (2H,)
        cache["att"] = (pre, th, e, alpha, H, s_prev)
        return c, alpha

    def forward(self, src, tgt_in):
        """Teacher forcing. Returns logits (T, V), alphas (T, L), and caches."""
        H, enc_cache = self.encode(src)
        L = len(src)
        T = len(tgt_in)
        s = np.zeros(self.H)
        logits = np.zeros((T, self.V))
        alphas = np.zeros((T, L))
        caches = []
        for t in range(T):
            acache = {}
            c, alpha = self.attention(s, H, acache)
            x = np.concatenate([self.Emb[tgt_in[t]], c])
            s_new = np.tanh(x @ self.Ws + s @ self.Us)
            logits[t] = s_new @ self.Wo
            caches.append((s, x, s_new, acache))
            alphas[t] = alpha
            s = s_new
        self.cache = (src, tgt_in, H, enc_cache, caches, alphas)
        return logits, alphas

    # -- loss and gradients ----------------------------------------------
    def loss_and_grad(self, src, tgt_in, tgt_out):
        logits, _ = self.forward(src, tgt_in)
        T = len(tgt_out)
        p = softmax(logits)
        loss = float(-np.mean(np.log(p[np.arange(T), tgt_out] + 1e-12)))
        dlogits = p.copy()
        dlogits[np.arange(T), tgt_out] -= 1.0
        dlogits /= T

        src_, tgt_in_, H, (X, hf, hb), caches, _ = self.cache
        g = {k: np.zeros_like(v) for k, v in self.params()}
        dH = np.zeros_like(H)
        ds_next = np.zeros(self.H)

        for t in reversed(range(T)):
            s_prev, x, s_new, acache = caches[t]
            pre, th, e, alpha, Hc, s_prev_ = acache["att"]
            dz = dlogits[t]
            g["Wo"] += np.outer(s_new, dz)
            ds = dz @ self.Wo.T + ds_next

            da = ds * (1.0 - s_new * s_new)              # through tanh
            g["Ws"] += np.outer(x, da)
            g["Us"] += np.outer(s_prev, da)
            dx = da @ self.Ws.T
            ds_prev = da @ self.Us.T

            dc = dx[self.E:]                              # context part
            demb = dx[:self.E]
            g["Emb"][tgt_in_[t]] += demb

            # c = alpha @ H
            dalpha = Hc @ dc
            dH += np.outer(alpha, dc)
            # alpha = softmax(e)
            de = alpha * (dalpha - (alpha * dalpha).sum())
            # e = th @ va
            g["va"] += th.T @ de
            dth = np.outer(de, self.va)
            dpre = dth * (1.0 - th * th)
            g["Wa"] += Hc.T @ dpre
            g["Ua"] += np.outer(s_prev_, dpre.sum(axis=0))
            dH += dpre @ self.Wa.T
            ds_prev += (dpre @ self.Ua.T).sum(axis=0)
            ds_next = ds_prev

        # encoder backward: H = [hf; hb]
        dhf = dH[:, :self.H]
        dhb = dH[:, self.H:]
        # backward-direction RNN runs right-to-left
        carry = np.zeros(self.H)
        for t in reversed(range(len(src_))):
            a = dhf[t] + carry
            dpre = a * (1.0 - hf[t] * hf[t])
            g["Wf"] += np.outer(X[t], dpre)
            g["Uf"] += np.outer(hf[t - 1] if t > 0 else np.zeros(self.H), dpre)
            g["Emb"][src_[t]] += dpre @ self.Wf.T
            carry = dpre @ self.Uf.T
        carry = np.zeros(self.H)
        for t in range(len(src_)):
            a = dhb[t] + carry
            dpre = a * (1.0 - hb[t] * hb[t])
            g["Wb"] += np.outer(X[t], dpre)
            g["Ub"] += np.outer(hb[t + 1] if t + 1 < len(src_) else np.zeros(self.H), dpre)
            g["Emb"][src_[t]] += dpre @ self.Wb.T
            carry = dpre @ self.Ub.T
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
# task: reverse the source, so the alignment is known exactly
# --------------------------------------------------------------------------

def make_pair(rng, length, n_symbols):
    src = [int(x) for x in rng.integers(3, 3 + n_symbols, size=length)]
    tgt = src[::-1]
    return src, [BOS] + tgt, tgt + [EOS]


def train(model, steps, length=5, n_symbols=6, batch=32, lr=3e-3, seed=0):
    """`length` may be an int or a (lo, hi) range sampled per example."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    losses = []
    for _ in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        total = 0.0
        for _ in range(batch):
            L = length if isinstance(length, int) else int(rng.integers(length[0], length[1] + 1))
            src, tin, tout = make_pair(rng, L, n_symbols)
            loss, g = model.loss_and_grad(src, tin, tout)
            total += loss
            for k in acc:
                acc[k] += g[k]
        for k in acc:
            acc[k] /= batch
        opt.step(model.params(), acc)
        losses.append(total / batch)
    return losses


def evaluate(model, n=100, length=5, n_symbols=6, seed=1):
    """Exact-match rate (greedy decoding) and alignment hit rate."""
    rng = np.random.default_rng(seed)
    exact = 0
    hits = 0
    total = 0
    for _ in range(n):
        src, tin, tout = make_pair(rng, length, n_symbols)
        _, alphas = model.forward(src, tin)
        pred = []
        s = np.zeros(model.H)
        H, _ = model.encode(src)
        tok = BOS
        for t in range(length):
            acache = {}
            c, _ = model.attention(s, H, acache)
            x = np.concatenate([model.Emb[tok], c])
            s = np.tanh(x @ model.Ws + s @ model.Us)
            tok = int(np.argmax(s @ model.Wo))
            pred.append(tok)
        if pred == src[::-1]:
            exact += 1
        # alignment: target position t should attend to source position L-1-t
        peak = alphas.argmax(axis=1)
        for t in range(len(src)):
            total += 1
            if peak[t] == len(src) - 1 - t:
                hits += 1
    return exact / n, hits / max(total, 1)
