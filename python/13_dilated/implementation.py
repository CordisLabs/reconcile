"""Multi-Scale Context Aggregation by Dilated Convolutions (Yu & Koltun, 2015).

Dilated convolution, implemented as a normal convolution with the kernel "spread out",
and the receptive-field claims of the paper measured directly.

The paper makes two claims that are exact arithmetic, not empirical tendencies:
  * stacking 3x3 convs with dilations 1,2,4 gives a 15x15 receptive field in 3 layers,
  * the basic context module (7 layers, dilations 1,1,2,4,8,16,1) has a 67x67
    receptive field — the paper uses 33 pixels of padding because of it.

Both are measured here by feeding a delta and counting the nonzero outputs. Because the
stack is linear, the support of that response *is* the receptive field exactly.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def spread_kernel(kernel, dilation):
    """Insert dilation-1 zeros between taps: the dilation identity.

    A k x k kernel dilated by d is exactly a (k-1)*d+1 kernel with zeros interleaved,
    so a dilated convolution is a normal convolution with a sparse kernel.
    """
    if dilation == 1:
        return kernel.copy()
    k = kernel.shape[0]
    out = np.zeros(((k - 1) * dilation + 1,) * 2)
    out[::dilation, ::dilation] = kernel
    return out


def conv2d_same(x, kernel):
    """Single-channel 2D convolution, output the same size as the input.

    Padding is chosen so the output is aligned with the input: p = (k-1)/2 for odd k.
    """
    k = kernel.shape[0]
    if k % 2 == 0:
        raise ValueError("kernel size must be odd")
    p = k // 2
    xp = np.pad(x, p)
    win = sliding_window_view(xp, (k, k))
    return np.einsum("ijkl,kl->ij", win, kernel)


def dilated_conv2d(x, kernel, dilation):
    return conv2d_same(x, spread_kernel(kernel, dilation))


def context_module(x, dilations):
    """Linear version of the paper's context module: 3x3 convs at increasing dilation.

    The paper's module also has ReLU and 1x1 projections between layers; those do not
    affect the receptive field, which is what is being measured, so they are omitted
    and noted in the README.
    """
    h = x
    kernel = np.ones((3, 3))
    for d in dilations:
        h = dilated_conv2d(h, kernel, d)
    return h


def receptive_field(dilations, canvas=321):
    """Measure the receptive field by feeding a single 1 at the centre.

    The operator is linear and shift-invariant, so the support of the impulse response
    has the same size as the receptive field of any interior output.
    """
    x = np.zeros((canvas, canvas))
    x[canvas // 2, canvas // 2] = 1.0
    y = context_module(x, dilations)
    rows = np.nonzero(np.any(y != 0, axis=1))[0]
    cols = np.nonzero(np.any(y != 0, axis=0))[0]
    return int(rows.size), int(cols.size)


def parameter_count(dilations, kernel_size=3):
    """Parameters in the stack; a single conv of the same receptive field would be worse."""
    return len(dilations) * kernel_size * kernel_size
