"""Neural Turing Machines (Graves, Wayne, Danihelka, 2014).

The paper's mechanism is a differentiable memory: a controller reads and writes an N x W
matrix through addressing weights that are a probability distribution over rows.

Only the addressing and memory machinery is implemented here, and that is the part with
exact properties: the weights are a softmax so they sum to one, content addressing with a
large sharpening becomes one-hot at the nearest row, a zero interpolation gate leaves the
previous weights untouched, and shifting is a circular convolution. No copy task is
trained; that needs a controller and BPTT through the whole addressing chain and is not
claimed.
"""

import numpy as np


def cosine_similarity(memory, key):
    """Cosine similarity between the key and every row, in [-1, 1]."""
    num = memory @ key
    den = np.linalg.norm(memory, axis=1) * (np.linalg.norm(key) + 1e-12)
    return num / (den + 1e-12)


def content_addressing(memory, key, beta):
    """w = softmax(beta * cos(K, M_i)) — the paper's eq. 2-3."""
    sims = cosine_similarity(memory, key) * beta
    sims = sims - sims.max()
    e = np.exp(sims)
    return e / e.sum()


def location_addressing(prev_w, gate, shift, gamma):
    """w = sharpen(g * prev + (1-g) * circular_shift(prev, shift)) — eq. 5-6."""
    n = prev_w.shape[0]
    # circular convolution with the shift distribution
    conj = np.array([
        sum(shift[(i - j) % n] * prev_w[j] for j in range(n))
        for i in range(n)
    ])
    w = gate * prev_w + (1.0 - gate) * conj
    w = np.power(np.maximum(w, 1e-12), gamma)
    return w / w.sum()


def read(memory, w):
    """r = sum_i w_i M_i."""
    return w @ memory


def write(memory, w, erase, add):
    """M_i <- M_i * (1 - w_i * e) + w_i * a — eq. 8-9."""
    e = np.outer(w, erase)
    a = np.outer(w, add)
    return memory * (1.0 - e) + a


class NTM:
    """A minimal cell: one read head and one write head, driven by explicit addresses."""

    def __init__(self, slots=8, width=4, seed=0):
        rng = np.random.default_rng(seed)
        self.slots, self.width = slots, width
        self.memory = rng.normal(0, 0.1, (slots, width))
        self.prev_read = np.ones(slots) / slots
        self.prev_write = np.ones(slots) / slots

    def read_step(self, key, beta):
        w = content_addressing(self.memory, key, beta)
        self.prev_read = w
        return read(self.memory, w), w

    def write_step(self, key, beta, erase, add):
        w = content_addressing(self.memory, key, beta)
        self.prev_write = w
        self.memory = write(self.memory, w, erase, add)
        return w
