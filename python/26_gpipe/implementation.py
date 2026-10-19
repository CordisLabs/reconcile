"""GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism (Huang et al., 2019).

The paper's central claim is an equivalence: splitting a batch into micro-batches and
running them through a pipeline produces the *same* result as one forward pass, while
holding far fewer activations at once. That is arithmetic, so it is checked exactly.

A model is represented as a chain of stages. `forward_full` runs the whole batch through
every stage, keeping every activation for the backward pass. `forward_micro` splits the
batch and lets each stage work on a different micro-batch at the same time, which is what
GPipe does; the peak number of live activation elements is counted in both cases.
"""

import numpy as np


class Pipeline:
    def __init__(self, dims, seed=0):
        """dims = [d0, d1, ..., dK] gives K linear stages (plus a tanh between them)."""
        rng = np.random.default_rng(seed)
        self.dims = dims
        self.stages = []
        for i in range(len(dims) - 1):
            s = 1.0 / np.sqrt(dims[i])
            self.stages.append((rng.normal(0, s, (dims[i], dims[i + 1])), np.zeros(dims[i + 1])))

    def _run_stage(self, k, h):
        W, b = self.stages[k]
        out = h @ W + b
        if k < len(self.stages) - 1:
            out = np.tanh(out)
        return out

    def forward_full(self, x):
        """One pass over the whole batch; every intermediate is kept."""
        alive = 0
        h = x
        for k in range(len(self.stages)):
            h = self._run_stage(k, h)
            alive += h.size            # kept until the backward pass
        return h, alive

    def forward_micro(self, x, micro):
        """Fill-and-drain schedule over `micro` chunks; returns the output and peak live."""
        n = x.shape[0]
        if n % micro != 0:
            raise ValueError("batch must divide evenly into micro-batches")
        chunks = [x[i * n // micro:(i + 1) * n // micro] for i in range(micro)]
        K = len(self.stages)
        # buffers[k] holds activation tensors waiting to enter stage k
        buffers = [[] for _ in range(K + 1)]
        for i in range(micro):
            buffers[0].append(chunks[i])
        peak = 0
        for t in range(K + micro - 1):
            for k in range(K):
                mb = t - k
                if 0 <= mb < micro and buffers[k]:
                    inp = buffers[k].pop(0)
                    out = self._run_stage(k, inp)
                    buffers[k + 1].append(out)
            live = sum(t.size for buf in buffers for t in buf)
            peak = max(peak, live)
        return np.concatenate(buffers[K], axis=0), peak

    def parameters(self):
        return sum(W.size + b.size for W, b in self.stages)

# NOTE: a closed-form estimate of the peak would be easy to get wrong, and an earlier
# version of this file did get it wrong — it predicted a linear gain in `micro`, while the
# measured peak saturates near the pipeline depth. The numbers below are measured by
# running the schedule, not estimated.
def peak_curve(pipeline, x, micros):
    """(micro, peak elements, output) for each split size; used by the tests and report."""
    full_out, full_peak = pipeline.forward_full(x)
    rows = []
    for m in micros:
        out, peak = pipeline.forward_micro(x, m)
        rows.append((m, peak, float(np.abs(out - full_out).max()), full_peak))
    return rows


def backward_full(pipeline, x, grad_out=None):
    """Gradients through the whole batch, keeping every intermediate."""
    out, _ = pipeline.forward_full(x)
    if grad_out is None:
        grad_out = np.ones_like(out)
    grads = [(np.zeros_like(W), np.zeros_like(b)) for W, b in pipeline.stages]
    h = x
    acts = [x]
    for k in range(len(pipeline.stages)):
        h = pipeline._run_stage(k, h)
        acts.append(h)
    dh = grad_out
    for k in reversed(range(len(pipeline.stages))):
        W, b = pipeline.stages[k]
        if k < len(pipeline.stages) - 1:
            dh = dh * (1.0 - acts[k + 1] ** 2)
        dW = acts[k].T @ dh
        db = dh.sum(axis=0)
        grads[k] = (dW, db)
        dh = dh @ W.T
    return grads


def backward_micro(pipeline, x, micro, grad_out=None):
    """Gradients accumulated over micro-batches — the pipeline version of the same thing."""
    n = x.shape[0]
    if n % micro != 0:
        raise ValueError("batch must divide evenly into micro-batches")
    chunks = [x[i * n // micro:(i + 1) * n // micro] for i in range(micro)]
    grads = [(np.zeros_like(W), np.zeros_like(b)) for W, b in pipeline.stages]
    for chunk in chunks:
        g = backward_full(pipeline, chunk, None if grad_out is None else
                          grad_out[:chunk.shape[0]])
        for k, (dW, db) in enumerate(g):
            grads[k] = (grads[k][0] + dW, grads[k][1] + db)
    return grads


def backward_recompute(pipeline, x, grad_out=None):
    """Gradients with activations recomputed instead of stored.

    Nothing between the input and the output is kept: the backward walk replays each
    stage's forward from the input. That trades compute for memory, which is what
    checkpointing does. Returns (grads, peak_stored_elements).
    """
    if grad_out is None:
        out, _ = pipeline.forward_full(x)
        grad_out = np.ones_like(out)
    K = len(pipeline.stages)
    grads = [(np.zeros_like(W), np.zeros_like(b)) for W, b in pipeline.stages]
    peak = 0
    dh = grad_out
    for k in reversed(range(K)):
        h = x
        for j in range(k):
            h = pipeline._run_stage(j, h)
        prev = h
        out_k = pipeline._run_stage(k, prev)
        # tanh' at this stage's output; the last stage is linear, as in _run_stage
        if k < K - 1:
            dh = dh * (1.0 - out_k ** 2)
        W, _b = pipeline.stages[k]
        grads[k] = (prev.T @ dh, dh.sum(axis=0))
        dh = dh @ W.T
        peak = max(peak, prev.size + out_k.size)
    return grads, peak
