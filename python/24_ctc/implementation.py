"""Deep Speech 2 (Amodei et al., 2015) — CTC loss, checked against brute force.

CTC maps a length-T frame sequence to a shorter label sequence by summing over every
alignment that collapses to it. The forward-backward recursion is the standard way to
compute that sum; brute-force enumeration is the definition. For small T and small
alphabets the two can be compared exactly, which is what the tests do.

Only the loss is implemented. The paper's models are deep recurrent networks trained on
thousands of hours of speech; that is not in scope and is not claimed.
"""

import numpy as np

BLANK = 0


def log_softmax(logits, axis=-1):
    m = logits.max(axis=axis, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=axis, keepdims=True))


def _extend(target):
    """Interleave blanks: a, blank, b, blank, c ... so repeats can be separated."""
    out = [BLANK]
    for t in target:
        out.extend([t, BLANK])
    return out


def ctc_loss(log_probs, target):
    """Negative log probability of `target` under all alignments, via forward recursion.

    log_probs: (T, V) log-probabilities. target: list of label ids (no blanks).
    Returns -log P(target).
    """
    T, V = log_probs.shape
    L = len(target)
    if L == 0:
        return float(-log_probs[:, BLANK].sum())
    ext = _extend(target)
    S = len(ext)
    NEG = -1e30
    alpha = np.full((T, S), NEG)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            if ext[s] == BLANK and s < 2:
                prev = alpha[t - 1, s]
            else:
                prev = alpha[t - 1, s]
                if s >= 1:
                    prev = np.logaddexp(prev, alpha[t - 1, s - 1])
                if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                    prev = np.logaddexp(prev, alpha[t - 1, s - 2])
            alpha[t, s] = prev + log_probs[t, ext[s]]
    end = alpha[T - 1, S - 1]
    if S > 1:
        end = np.logaddexp(end, alpha[T - 1, S - 2])
    return float(-end)


def collapse(path):
    """CTC collapse: merge consecutive repeats, then drop blanks."""
    out = []
    prev = None
    for p in path:
        if p != prev:
            if p != BLANK:
                out.append(p)
        prev = p
    return out


def brute_force_loss(log_probs, target, alphabet):
    """Enumerate every path of length T and sum those that collapse to `target`."""
    T = log_probs.shape[0]
    target = list(target)
    total = 0.0
    from itertools import product

    for path in product(range(alphabet), repeat=T):
        if collapse(path) == target:
            total += float(np.exp(sum(log_probs[t, path[t]] for t in range(T))))
    return float(-np.log(total)) if total > 0 else float("inf")


# --------------------------------------------------------------------------
# decoding and gradient
# --------------------------------------------------------------------------

def greedy_decode(log_probs):
    """Best path, collapsed."""
    path = log_probs.argmax(axis=1)
    return collapse(list(path))


def beam_search(log_probs, beam=8):
    """Prefix beam search: keep the `beam` most probable prefixes under CTC.

    Merges paths that differ only by blanks or repeats, which greedy decoding cannot do.
    """
    T = log_probs.shape[0]
    beams = {(): 0.0}                     # prefix -> log probability
    for t in range(T):
        nxt = {}
        for prefix, lp in beams.items():
            for k in range(log_probs.shape[1]):
                # a blank adds nothing; a repeat of the last symbol also collapses away
                if k == BLANK or (prefix and k == prefix[-1]):
                    cand = prefix
                else:
                    cand = prefix + (k,)
                score = lp + log_probs[t, k]
                cur = nxt.get(cand)
                if cur is None:
                    nxt[cand] = score
                else:
                    m = max(cur, score)
                    nxt[cand] = m + np.log(np.exp(cur - m) + np.exp(score - m))
        top = sorted(nxt.items(), key=lambda kv: -kv[1])[:beam]
        beams = dict(top)
    return list(max(beams.items(), key=lambda kv: kv[1])[0])


def ctc_loss_and_grad(logits, target):
    """Negative log likelihood and its gradient with respect to the logits."""
    log_probs = log_softmax(logits)
    T, V = log_probs.shape
    loss = ctc_loss(log_probs, target)

    ext = _extend(target)
    S = len(ext)
    NEG = -1e30

    alpha = np.full((T, S), NEG)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            prev = alpha[t - 1, s]
            if s >= 1:
                prev = np.logaddexp(prev, alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                prev = np.logaddexp(prev, alpha[t - 1, s - 2])
            alpha[t, s] = prev + log_probs[t, ext[s]]

    beta = np.full((T, S), NEG)
    beta[T - 1, S - 1] = 0.0
    if S > 1:
        beta[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        for s in range(S):
            b = beta[t + 1, s] + log_probs[t + 1, ext[s]]
            if s + 1 < S:
                b = np.logaddexp(b, beta[t + 1, s + 1] + log_probs[t + 1, ext[s + 1]])
            if s + 2 < S and ext[s + 2] != BLANK and ext[s + 2] != ext[s]:
                b = np.logaddexp(b, beta[t + 1, s + 2] + log_probs[t + 1, ext[s + 2]])
            beta[t, s] = b

    end = alpha[T - 1, S - 1]
    if S > 1:
        end = np.logaddexp(end, alpha[T - 1, S - 2])

    # dL/dlogp, then through the log_softmax Jacobian
    dlogp = np.zeros((T, V))
    for t in range(T):
        for s in range(S):
            if alpha[t, s] <= NEG / 2 or beta[t, s] <= NEG / 2:
                continue
            dlogp[t, ext[s]] -= np.exp(alpha[t, s] + beta[t, s] - end)
    probs = np.exp(log_probs)
    # dL/dlogits = dL/dlogp - softmax * sum_j(dL/dlogp_j)
    grad = dlogp - probs * dlogp.sum(axis=1, keepdims=True)
    return loss, grad
