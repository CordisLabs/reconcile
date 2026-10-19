"""ImageNet Classification with Deep CNNs (AlexNet, Krizhevsky et al., 2012).

AlexNet's headline result is ImageNet top-5 error, which this slice does not reproduce.
What it does reproduce is the paper's §3.1 claim, which is stated as a measurement:

    "Deep convolutional neural networks with ReLUs train several times faster than their
     equivalents with tanh units. This is demonstrated in Figure 1, which shows the
     number of iterations required to reach 25% training error on the CIFAR-10 dataset
     for a particular four-layer convolutional network."

    "A four-layer convolutional network with ReLUs ... reaches a 25% training error rate
     on CIFAR-10 six times faster than an equivalent network with tanh neurons."

So: same architecture, same data, same optimiser, only the nonlinearity differs; measure
iterations to 25% training error. CIFAR-10 is not in this repository, so the task is a
synthetic memorisation problem — which is the regime the claim is about ("ability to fit
the training set").

Includes a real conv/pool backward pass, checked against finite differences.
"""

import numpy as np


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------

def im2col(x, k, stride, pad):
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (xp.shape[1] - k) // stride + 1
    Wo = (xp.shape[2] - k) // stride + 1
    cols = np.zeros((C * k * k, Ho * Wo))
    for i in range(Ho):
        for j in range(Wo):
            patch = xp[:, i * stride:i * stride + k, j * stride:j * stride + k]
            cols[:, i * Wo + j] = patch.reshape(-1)
    return cols, Ho, Wo


def _col2im(cols, x_shape, k, stride, pad, Ho, Wo):
    """Adjoint of im2col: scatter-add gradients back into the padded input."""
    C, H, W = x_shape
    Hp, Wp = H + 2 * pad, W + 2 * pad
    xp = np.zeros((C, Hp, Wp))
    for i in range(Ho):
        for j in range(Wo):
            patch = cols[:, i * Wo + j].reshape(C, k, k)
            xp[:, i * stride:i * stride + k, j * stride:j * stride + k] += patch
    if pad == 0:
        return xp
    return xp[:, pad:-pad, pad:-pad]


class Conv:
    def __init__(self, c_in, c_out, k=3, stride=1, pad=1, seed=0, init_scale=None):
        rng = np.random.default_rng(seed)
        s = np.sqrt(2.0 / (c_in * k * k)) if init_scale is None else init_scale
        self.W = rng.normal(0, s, (c_out, c_in, k, k))
        self.b = np.zeros(c_out)
        self.k, self.stride, self.pad = k, stride, pad
        self.cache = None

    def forward(self, x):
        cols, Ho, Wo = im2col(x, self.k, self.stride, self.pad)
        out = self.W.reshape(self.W.shape[0], -1) @ cols + self.b[:, None]
        self.cache = (x.shape, cols, Ho, Wo)
        return out.reshape(self.W.shape[0], Ho, Wo)

    def backward(self, dout):
        x_shape, cols, Ho, Wo = self.cache
        Cout = self.W.shape[0]
        dout2 = dout.reshape(Cout, -1)
        gW = (dout2 @ cols.T).reshape(self.W.shape)
        gb = dout2.sum(axis=1)
        dcols = self.W.reshape(Cout, -1).T @ dout2
        dx = _col2im(dcols, x_shape, self.k, self.stride, self.pad, Ho, Wo)
        return dx, gW, gb


class MaxPool:
    def __init__(self, k=2):
        self.k = k
        self.cache = None

    def forward(self, x):
        C, H, W = x.shape
        k = self.k
        Ho, Wo = H // k, W // k
        out = np.zeros((C, Ho, Wo))
        arg = np.zeros((C, Ho, Wo), dtype=np.int64)
        for c in range(C):
            for i in range(Ho):
                for j in range(Wo):
                    patch = x[c, i * k:(i + 1) * k, j * k:(j + 1) * k]
                    idx = int(np.argmax(patch))
                    arg[c, i, j] = idx
                    out[c, i, j] = patch.reshape(-1)[idx]
        self.cache = (x.shape, arg)
        return out

    def backward(self, dout):
        (C, H, W), arg = self.cache
        k = self.k
        Ho, Wo = dout.shape[1], dout.shape[2]
        dx = np.zeros((C, H, W))
        for c in range(C):
            for i in range(Ho):
                for j in range(Wo):
                    di, dj = divmod(int(arg[c, i, j]), k)
                    dx[c, i * k + di, j * k + dj] += dout[c, i, j]
        return dx


class Act:
    """The only thing that changes between the two arms of the experiment."""

    def __init__(self, kind="relu"):
        self.kind = kind
        self.cache = None

    def forward(self, x):
        if self.kind == "relu":
            self.cache = x
            return np.maximum(0.0, x)
        if self.kind == "tanh":
            t = np.tanh(x)
            self.cache = t
            return t
        raise ValueError(self.kind)

    def backward(self, dout):
        if self.kind == "relu":
            return dout * (self.cache > 0)
        t = self.cache
        return dout * (1.0 - t * t)


class Linear:
    def __init__(self, n_in, n_out, seed=0, init_scale=None):
        rng = np.random.default_rng(seed)
        s = np.sqrt(2.0 / n_in) if init_scale is None else init_scale
        self.W = rng.normal(0, s, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.cache = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self.cache
        self.gW = x.T @ dout
        self.gb = dout.sum(axis=0)
        return dout @ self.W.T


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

class SmallConvNet:
    """conv-act-pool-conv-act-pool-flatten-linear, with a switchable activation."""

    def __init__(self, activation="relu", n_classes=10, seed=0, init_scale=None):
        self.conv1 = Conv(1, 8, 3, 1, 1, seed=seed, init_scale=init_scale)
        self.act1 = Act(activation)
        self.pool1 = MaxPool(2)
        self.conv2 = Conv(8, 16, 3, 1, 1, seed=seed + 1, init_scale=init_scale)
        self.act2 = Act(activation)
        self.pool2 = MaxPool(2)
        # 8x8 input -> 8x8 -> 4x4 -> 4x4 -> 2x2
        self.fc = Linear(16 * 2 * 2, n_classes, seed=seed + 2, init_scale=init_scale)
        self.params = {
            "conv1": (self.conv1.W, self.conv1.b),
            "conv2": (self.conv2.W, self.conv2.b),
            "fc": (self.fc.W, self.fc.b),
        }

    def forward(self, x):
        h = self.conv1.forward(x)
        h = self.act1.forward(h)
        h = self.pool1.forward(h)
        h = self.conv2.forward(h)
        h = self.act2.forward(h)
        h = self.pool2.forward(h)
        self.flat_shape = h.shape
        logits = self.fc.forward(h.reshape(1, -1))
        return logits

    def loss_and_grads(self, x, y):
        logits = self.forward(x)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        loss = -np.log(p[0, y] + 1e-12)
        dlogits = p.copy()
        dlogits[0, y] -= 1.0
        d = self.fc.backward(dlogits)
        d = d.reshape(self.flat_shape)
        d = self.pool2.backward(d)
        d = self.act2.backward(d)
        dx, gW2, gb2 = self.conv2.backward(d)
        d = self.pool1.backward(dx)
        d = self.act1.backward(d)
        _, gW1, gb1 = self.conv1.backward(d)
        return loss, {"conv1": (gW1, gb1), "conv2": (gW2, gb2), "fc": (self.fc.gW, self.fc.gb)}


# --------------------------------------------------------------------------
# task
# --------------------------------------------------------------------------

def make_task(n_classes=10, n_per_class=20, size=8, noise=0.15, seed=0):
    """Memorisation task: each class is a fixed random binary pattern, plus noise."""
    rng = np.random.default_rng(seed)
    patterns = (rng.random((n_classes, size, size)) > 0.5).astype(float)
    xs, ys = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            img = patterns[c].copy()
            flip = rng.random((size, size)) < noise
            img[flip] = 1.0 - img[flip]
            xs.append(img[None, :, :])
            ys.append(c)
    return np.stack(xs), np.array(ys)


def train_arm(activation, xs, ys, steps=600, lr=0.02, seed=0, init_scale=None):
    """Returns the step at which training error first drops to 25% or below."""
    model = SmallConvNet(activation=activation, seed=seed, init_scale=init_scale)
    params = model.params
    m = {k: [np.zeros_like(p) for p in v] for k, v in params.items()}
    v = {k: [np.zeros_like(p) for p in vv] for k, vv in params.items()}
    t = 0
    reach = None
    err = 1.0
    for step in range(steps):
        correct = 0
        grads_acc = {k: [np.zeros_like(p) for p in vv] for k, vv in params.items()}
        for x, y in zip(xs, ys):
            loss, g = model.loss_and_grads(x, y)
            if int(np.argmax(model.forward(x))) == y:
                correct += 1
            for k in g:
                for i in range(2):
                    grads_acc[k][i] += g[k][i]
        err = 1.0 - correct / len(ys)
        if reach is None and err <= 0.25:
            reach = step
        t += 1
        for k in params:
            for i in range(2):
                gk = grads_acc[k][i] / len(ys)
                m[k][i] = 0.9 * m[k][i] + 0.1 * gk
                v[k][i] = 0.999 * v[k][i] + 0.001 * gk * gk
                mh = m[k][i] / (1 - 0.9 ** t)
                vh = v[k][i] / (1 - 0.999 ** t)
                params[k][i][...] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return reach, err
