from importlib import reload

import numpy as np

import skimage as ski

# from skimage.transform._hough_transform import _probabilistic_hough_line
import py_hough
reload(py_hough)


def _sort_lines(lines):
    # Sort each line by x, y, sort lines by contents.
    sorted_lines = []
    for ln in lines:
        sorted_lines.append(sorted(ln, key=lambda x: (x[0], x[1])))
    return sorted(sorted_lines)


_ph = py_hough._probabilistic_hough_line
sl = _sort_lines
# Single line in LR (x)
L = 20

# Two lines (x, y)
more = np.zeros((30, 30))
offset = 5
n = L
back_off = offset + n - 1
for i in range(n):
    oi = offset + i
    more[oi, oi] = 1
diags = [
    [(offset, offset), (back_off, back_off)],
]

threshold = 10
line_length = L - 1
line_gap = 2

# image = ski.data.camera()
# more = ski.feature.canny(image, 2, 1, 25)
# line_length = 5

theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

seed = 88
# for seed in range(100):
if True:
    lines, mro, mx_rho, mx_theta = _ph(more,
                                       threshold=threshold,
                                       line_length=line_length,
                                       line_gap=line_gap,
                                       theta=theta,
                                       rng=seed,
                                       verbose=True)
    print('rho', mx_rho, 'theta', mx_theta, mro)
    if not sl(lines) == diags:
        print('Error at', seed)
