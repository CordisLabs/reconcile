"""CS231n: Convolutional Neural Networks for Visual Recognition — convolution, pooling
and parameter counting from scratch in NumPy.

The checkable claim here is structural rather than empirical: the im2col trick used to
make convolution fast is exactly the naive convolution, only rearranged. "Fast" is not
an excuse for a different answer.

The second checkable number is VGG16's parameter count, 138,357,544, which the course
notes use as a worked example. It is pure arithmetic over the architecture table.
"""

import numpy as np


def conv2d_naive(x, W, b=None, stride=1, pad=0):
    """x: (C, H, Wd), W: (Cout, C, k, k). Straight loops, the reference answer."""
    C, H, Wd = x.shape
    Cout, Cw, k, _ = W.shape
    if C != Cw:
        raise ValueError("channel mismatch")
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Hp, Wp = xp.shape[1], xp.shape[2]
    Ho = (Hp - k) // stride + 1
    Wo = (Wp - k) // stride + 1
    out = np.zeros((Cout, Ho, Wo))
    for co in range(Cout):
        for i in range(Ho):
            for j in range(Wo):
                patch = xp[:, i * stride:i * stride + k, j * stride:j * stride + k]
                out[co, i, j] = float((patch * W[co]).sum())
        if b is not None:
            out[co] += b[co]
    return out


def im2col(x, k, stride=1, pad=0):
    """(C,H,W) -> (C*k*k, Ho*Wo), each column one receptive field."""
    C, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (xp.shape[1] - k) // stride + 1
    Wo = (xp.shape[2] - k) // stride + 1
    cols = np.zeros((C * k * k, Ho * Wo))
    for i in range(Ho):
        for j in range(Wo):
            patch = xp[:, i * stride:i * stride + k, j * stride:j * stride + k]
            cols[:, i * Wo + j] = patch.reshape(-1)
    return cols, Ho, Wo


def conv2d_im2col(x, W, b=None, stride=1, pad=0):
    Cout, C, k, _ = W.shape
    cols, Ho, Wo = im2col(x, k, stride, pad)
    out = W.reshape(Cout, -1) @ cols
    if b is not None:
        out += b[:, None]
    return out.reshape(Cout, Ho, Wo)


def maxpool_forward(x, k=2, stride=2):
    """Returns the pooled output and the argmax indices needed for the backward pass."""
    C, H, Wd = x.shape
    Ho = (H - k) // stride + 1
    Wo = (Wd - k) // stride + 1
    out = np.zeros((C, Ho, Wo))
    argmax = np.zeros((C, Ho, Wo), dtype=np.int64)
    for c in range(C):
        for i in range(Ho):
            for j in range(Wo):
                patch = x[c, i * stride:i * stride + k, j * stride:j * stride + k]
                idx = int(np.argmax(patch))
                argmax[c, i, j] = idx
                out[c, i, j] = patch.reshape(-1)[idx]
    return out, argmax


def maxpool_backward(dout, argmax, x_shape, k=2, stride=2):
    """Routes each gradient to the position that won the max."""
    C, Ho, Wo = dout.shape
    dx = np.zeros(x_shape)
    for c in range(C):
        for i in range(Ho):
            for j in range(Wo):
                idx = int(argmax[c, i, j])
                di, dj = divmod(idx, k)
                dx[c, i * stride + di, j * stride + dj] += dout[c, i, j]
    return dx


def conv_output_size(size, k, stride=1, pad=0):
    return (size + 2 * pad - k) // stride + 1


# VGG16 configuration D, as used in the CS231n notes.
VGG16_CONV = [
    (64, 2), (128, 2), (256, 3), (512, 3), (512, 3),
]


def vgg16_params(fc_units=4096, classes=1000, in_ch=3, spatial=7):
    """Count parameters directly from the architecture, no framework involved."""
    total = 0
    prev = in_ch
    for planes, count in VGG16_CONV:
        for _ in range(count):
            total += planes * prev * 3 * 3 + planes
            prev = planes
    total += fc_units * (prev * spatial * spatial) + fc_units
    total += fc_units * fc_units + fc_units
    total += classes * fc_units + classes
    return total
