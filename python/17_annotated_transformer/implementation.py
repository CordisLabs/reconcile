"""The Annotated Transformer (Rush, 2018) — the full encoder-decoder, written out.

Components: sinusoidal positional encoding, multi-head scaled dot-product attention, a
masked self-attention decoder, cross-attention, position-wise feed-forward blocks,
residual connections, layer norm, and label smoothing.

Task: copy. The decoder reads a source sequence and reproduces it, which is the smallest
task that exercises every part of the architecture (encoder, masked decoder, cross
attention) and whose answer is exact.

Simplifications, since "the paper" here is a tutorial: one encoder layer and one decoder
layer, post-layer-norm as in the original, and no learning-rate warmup schedule.
"""

import numpy as np


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def positional_encoding(max_len, d_model):
    """Sinusoidal table from the paper: PE(pos, 2i) = sin(pos/10000^(2i/d))."""
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def layer_norm_forward(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    return xh * g + b, (xh, inv, g)


def layer_norm_backward(dy, cache):
    xh, inv, g = cache
    n = dy.shape[-1]
    dg = (dy * xh).sum(axis=0)
    db = dy.sum(axis=0)
    dxh = dy * g
    dx = inv * (dxh - dxh.mean(axis=-1, keepdims=True)
                - xh * (dxh * xh).mean(axis=-1, keepdims=True))
    return dx, dg, db


def split_heads(x, n_heads):
    T, D = x.shape
    dk = D // n_heads
    return x.reshape(T, n_heads, dk).transpose(1, 0, 2)


def merge_heads(x):
    h, T, dk = x.shape
    return x.transpose(1, 0, 2).reshape(T, h * dk)


def attention_forward(Q, K, V, mask=None):
    dk = Q.shape[-1]
    scale = 1.0 / np.sqrt(dk)
    S = (Q @ K.transpose(0, 2, 1)) * scale        # (h, Tq, Tk)
    if mask is not None:
        S = np.where(mask, S, -1e9)
    P = softmax(S, axis=-1)
    O = P @ V
    return O, (Q, K, V, P, scale)


def attention_backward(dO, cache, mask=None):
    Q, K, V, P, scale = cache
    dV = P.transpose(0, 2, 1) @ dO
    dP = dO @ V.transpose(0, 2, 1)
    dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
    if mask is not None:
        dS = np.where(mask, dS, 0.0)
    dQ = (dS @ K) * scale
    dK = (dS.transpose(0, 2, 1) @ Q) * scale
    return dQ, dK, dV


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class Transformer:
    def __init__(self, vocab, d_model=32, n_heads=4, d_ff=64, max_len=32, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.V, self.D, self.HH, self.F, self.L = vocab, d_model, n_heads, d_ff, max_len
        self.pe = positional_encoding(max_len, d_model)
        self.EncEmb = rng.normal(0, 0.5, (vocab, d_model))
        self.DecEmb = rng.normal(0, 0.5, (vocab, d_model))
        # encoder block
        self.e = {
            "Wq": rng.uniform(-s, s, (d_model, d_model)), "Wk": rng.uniform(-s, s, (d_model, d_model)),
            "Wv": rng.uniform(-s, s, (d_model, d_model)), "Wo": rng.uniform(-s, s, (d_model, d_model)),
            "g1": np.ones(d_model), "b1": np.zeros(d_model),
            "W1": rng.uniform(-s, s, (d_model, d_ff)), "bff1": np.zeros(d_ff),
            "W2": rng.uniform(-s, s, (d_ff, d_model)), "bff2": np.zeros(d_model),
            "g2": np.ones(d_model), "b2": np.zeros(d_model),
        }
        # decoder block
        self.d = {
            "Wq": rng.uniform(-s, s, (d_model, d_model)), "Wk": rng.uniform(-s, s, (d_model, d_model)),
            "Wv": rng.uniform(-s, s, (d_model, d_model)), "Wo": rng.uniform(-s, s, (d_model, d_model)),
            "g1": np.ones(d_model), "b1": np.zeros(d_model),
            "Wcq": rng.uniform(-s, s, (d_model, d_model)),
            "Wck": rng.uniform(-s, s, (d_model, d_model)),
            "Wcv": rng.uniform(-s, s, (d_model, d_model)),
            "Wco": rng.uniform(-s, s, (d_model, d_model)),
            "g2": np.ones(d_model), "b2": np.zeros(d_model),
            "W1": rng.uniform(-s, s, (d_model, d_ff)), "bff1": np.zeros(d_ff),
            "W2": rng.uniform(-s, s, (d_ff, d_model)), "bff2": np.zeros(d_model),
            "g3": np.ones(d_model), "b3": np.zeros(d_model),
        }
        self.Wout = rng.uniform(-s, s, (d_model, vocab))
        self.bout = np.zeros(vocab)

    def params(self):
        out = [("EncEmb", self.EncEmb), ("DecEmb", self.DecEmb),
               ("Wout", self.Wout), ("bout", self.bout)]
        for k, v in self.e.items():
            out.append(("e_" + k, v))
        for k, v in self.d.items():
            out.append(("d_" + k, v))
        return out

    # -- blocks ----------------------------------------------------------
    def _mha(self, x, p, n_heads, mask=None):
        Q = split_heads(x @ p["Wq"], n_heads)
        K = split_heads(x @ p["Wk"], n_heads)
        V = split_heads(x @ p["Wv"], n_heads)
        O, acache = attention_forward(Q, K, V, mask)
        merged = merge_heads(O)
        return merged @ p["Wo"], (x, Q, K, V, acache, merged)

    def _mha_backward(self, dout, cache, p, n_heads, mask=None):
        x, Q, K, V, acache, merged = cache
        dO = dout @ p["Wo"].T
        g = {"Wo": merged.T @ dout}
        dQ, dK, dV = attention_backward(split_heads(dO, n_heads), acache, mask)
        dx = merge_heads(dQ) @ p["Wq"].T + merge_heads(dK) @ p["Wk"].T + merge_heads(dV) @ p["Wv"].T
        g["Wq"] = x.T @ merge_heads(dQ)
        g["Wk"] = x.T @ merge_heads(dK)
        g["Wv"] = x.T @ merge_heads(dV)
        return dx, g

    def _cross(self, y, enc, p, n_heads):
        Q = split_heads(y @ p["Wcq"], n_heads)
        K = split_heads(enc @ p["Wck"], n_heads)
        V = split_heads(enc @ p["Wcv"], n_heads)
        O, acache = attention_forward(Q, K, V, None)
        merged = merge_heads(O)
        return merged @ p["Wco"], (y, enc, Q, K, V, acache, merged)

    def _cross_backward(self, dout, cache, p, n_heads):
        y, enc, Q, K, V, acache, merged = cache
        dO = dout @ p["Wco"].T
        g = {"Wco": merged.T @ dout}
        dQ, dK, dV = attention_backward(split_heads(dO, n_heads), acache, None)
        dy = merge_heads(dQ) @ p["Wcq"].T
        denc = merge_heads(dK) @ p["Wck"].T + merge_heads(dV) @ p["Wcv"].T
        g["Wcq"] = y.T @ merge_heads(dQ)
        g["Wck"] = enc.T @ merge_heads(dK)
        g["Wcv"] = enc.T @ merge_heads(dV)
        return dy, denc, g

    def _ffn(self, x, p):
        h = x @ p["W1"] + p["bff1"]
        a = np.maximum(0.0, h)
        return a @ p["W2"] + p["bff2"], (x, h, a)

    def _ffn_backward(self, dout, cache, p):
        x, h, a = cache
        g = {"W2": a.T @ dout, "bff2": dout.sum(axis=0)}
        da = dout @ p["W2"].T
        dh = da * (h > 0)
        g["W1"] = x.T @ dh
        g["bff1"] = dh.sum(axis=0)
        return dh @ p["W1"].T, g

    # -- forward ---------------------------------------------------------
    def forward(self, src, tgt_in):
        T = len(src)
        U = len(tgt_in)
        xe = self.EncEmb[src] + self.pe[:T]
        a_e, mha_cache = self._mha(xe, self.e, self.HH)
        x1 = xe + a_e
        n1, ln1_cache = layer_norm_forward(x1, self.e["g1"], self.e["b1"])
        f_e, ffn_cache_e = self._ffn(n1, self.e)
        x2 = n1 + f_e
        enc, ln2_cache = layer_norm_forward(x2, self.e["g2"], self.e["b2"])

        yd = self.DecEmb[tgt_in] + self.pe[:U]
        mask = np.tril(np.ones((U, U), dtype=bool))[None, :, :]
        a_d, mha_d_cache = self._mha(yd, self.d, self.HH, mask)
        y1 = yd + a_d
        n2, ln3_cache = layer_norm_forward(y1, self.d["g1"], self.d["b1"])
        c_d, cross_cache = self._cross(n2, enc, self.d, self.HH)
        y2 = n2 + c_d
        n3, ln4_cache = layer_norm_forward(y2, self.d["g2"], self.d["b2"])
        f_d, ffn_cache_d = self._ffn(n3, self.d)
        y3 = n3 + f_d
        dec, ln5_cache = layer_norm_forward(y3, self.d["g3"], self.d["b3"])
        logits = dec @ self.Wout + self.bout
        self.cache = dict(
            src=src, tgt_in=tgt_in, xe=xe, mha_cache=mha_cache, ln1=ln1_cache,
            ffn_e=ffn_cache_e, ln2=ln2_cache, yd=yd, mha_d=mha_d_cache, ln3=ln3_cache,
            cross=cross_cache, ln4=ln4_cache, ffn_d=ffn_cache_d, ln5=ln5_cache,
            enc=enc, dec=dec, mask=mask,
        )
        return logits

    # -- loss ------------------------------------------------------------
    def loss_and_grad(self, src, tgt_in, tgt_out, label_smoothing=0.0):
        logits = self.forward(src, tgt_in)
        T = len(tgt_out)
        p = softmax(logits)
        smooth = label_smoothing
        target = np.full_like(p, smooth / self.V)
        target[np.arange(T), tgt_out] = 1.0 - smooth + smooth / self.V
        loss = float(-(target * np.log(p + 1e-12)).sum(axis=1).mean())
        dlogits = (p - target) / T

        c = self.cache
        grads = {k: np.zeros_like(v) for k, v in self.params()}
        grads["Wout"] = c["dec"].T @ dlogits
        grads["bout"] = dlogits.sum(axis=0)
        ddec = dlogits @ self.Wout.T

        def add(prefix, gd):
            for k, v in gd.items():
                grads[prefix + k] += v

        # --- decoder ---
        # dec = LN(y3)
        dy3, grads["d_g3"], grads["d_b3"] = layer_norm_backward(ddec, c["ln5"])
        # y3 = n3 + ffn(n3)
        dn3_ffn, g = self._ffn_backward(dy3, c["ffn_d"], self.d)
        add("d_", g)
        dn3 = dy3 + dn3_ffn
        # n3 = LN(y2)
        dy2, grads["d_g2"], grads["d_b2"] = layer_norm_backward(dn3, c["ln4"])
        # y2 = n2 + cross(n2, enc)
        dn2_cross, denc, g = self._cross_backward(dy2, c["cross"], self.d, self.HH)
        add("d_", g)
        dn2 = dy2 + dn2_cross
        # n2 = LN(y1)
        dy1, grads["d_g1"], grads["d_b1"] = layer_norm_backward(dn2, c["ln3"])
        # y1 = yd + masked_mha(yd)
        dyd_attn, g = self._mha_backward(dy1, c["mha_d"], self.d, self.HH, c["mask"])
        add("d_", g)
        dyd = dy1 + dyd_attn
        np.add.at(grads["DecEmb"], c["tgt_in"], dyd)

        # --- encoder ---
        # enc = LN(x2)
        dx2, grads["e_g2"], grads["e_b2"] = layer_norm_backward(denc, c["ln2"])
        # x2 = n1 + ffn(n1)
        dn1_ffn, g = self._ffn_backward(dx2, c["ffn_e"], self.e)
        add("e_", g)
        dn1 = dx2 + dn1_ffn
        # n1 = LN(x1)
        dx1, grads["e_g1"], grads["e_b1"] = layer_norm_backward(dn1, c["ln1"])
        # x1 = xe + mha(xe)
        dxe_attn, g = self._mha_backward(dx1, c["mha_cache"], self.e, self.HH)
        add("e_", g)
        dxe = dx1 + dxe_attn
        np.add.at(grads["EncEmb"], c["src"], dxe)
        return loss, grads


class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params}
        self.v = {k: np.zeros_like(v) for k, v in params}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, p in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


BOS, EOS = 1, 2


def make_batch(rng, batch, length, n_symbols=6):
    pairs = []
    for _ in range(batch):
        src = [int(x) for x in rng.integers(3, 3 + n_symbols, size=length)]
        tin = [BOS] + src
        tout = src + [EOS]
        pairs.append((src, tin, tout))
    return pairs


def train(model, steps, length=5, batch=8, lr=2e-3, label_smoothing=0.0, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    losses = []
    for _ in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        total = 0.0
        pairs = make_batch(rng, batch, length)
        for src, tin, tout in pairs:
            loss, g = model.loss_and_grad(src, tin, tout, label_smoothing)
            total += loss
            for k in acc:
                acc[k] += g[k]
        for k in acc:
            acc[k] /= batch
        opt.step(model.params(), acc)
        losses.append(total / batch)
    return losses


def greedy_decode(model, src, max_len=12):
    H = len(src)
    out = []
    tin = [BOS]
    for _ in range(max_len):
        logits = model.forward(src, tin)
        tok = int(np.argmax(logits[-1]))
        if tok == EOS:
            break
        out.append(tok)
        tin.append(tok)
    return out


def evaluate(model, n=100, length=5, n_symbols=6, seed=1):
    """Exact-match rate plus perplexity on hard targets (smoothed or not)."""
    rng = np.random.default_rng(seed)
    exact = 0
    ce = 0.0
    count = 0
    for _ in range(n):
        src = [int(x) for x in rng.integers(3, 3 + n_symbols, size=length)]
        tin = [BOS] + src
        tout = src + [EOS]
        if greedy_decode(model, src, max_len=length + 2) == src:
            exact += 1
        logits = model.forward(src, tin)
        p = softmax(logits)
        for t, tok in enumerate(tout):
            ce += -np.log(p[t, tok] + 1e-12)
            count += 1
    return exact / n, float(np.exp(ce / count))
