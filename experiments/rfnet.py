"""Random-feature ReLU networks used in the simulation study.

Hidden layers are random (He initialisation) and the output layer is fitted by
ridge regression, so the randomness of "training" is the hidden-layer seed xi.
Architectures with the same seed are nested: a width-w network uses the first
w units of the seed's maximal-width weights, which pairs the comparisons.
"""
import numpy as np

D_X = 5
W_MAX = 256
L_MAX = 3
SIGMA = 0.5
RIDGE = 1e-3


def f_true(x):
    return (np.sin(2 * x[:, 0]) + x[:, 1] * x[:, 2]
            + 0.5 * np.cos(3 * x[:, 3]) + 0.3 * x[:, 4] ** 2)


def draw_data(n, rng):
    x = rng.standard_normal((n, D_X))
    y = f_true(x) + SIGMA * rng.standard_normal(n)
    return x, y


def seed_weights(seed):
    """Maximal-width random weights for every layer, generated from one seed."""
    rng = np.random.default_rng(seed)
    ws, bs = [], []
    fan_in = D_X
    for _ in range(L_MAX):
        ws.append(rng.standard_normal((fan_in if fan_in == D_X else W_MAX, W_MAX))
                  * np.sqrt(2.0 / (fan_in if fan_in == D_X else W_MAX)))
        bs.append(0.1 * rng.standard_normal(W_MAX))
        fan_in = W_MAX
    return ws, bs


def features(x, w, depth, weights):
    ws, bs = weights
    h = x
    prev = D_X
    for l in range(depth):
        W = ws[l][:prev, :w]
        if l > 0:
            W = W * np.sqrt(W_MAX / prev)  # keep He scaling for the sliced fan-in
        h = np.maximum(h @ W + bs[l][:w], 0.0)
        prev = w
    return h


def ridge_fit(h, y, lam=RIDGE):
    n, p = h.shape
    H = np.hstack([h, np.ones((n, 1))])
    A = H.T @ H + lam * n * np.eye(p + 1)
    A[-1, -1] -= lam * n  # do not penalise the intercept
    return np.linalg.solve(A, H.T @ y)


def predict(h, beta):
    return h @ beta[:-1] + beta[-1]


class Fitted:
    """A trained network (w, depth, seed) with an optional pruning mask."""

    def __init__(self, xtr, ytr, w, depth, weights, keep=None):
        self.w, self.depth, self.weights = w, depth, weights
        h = features(xtr, w, depth, weights)
        self.keep = np.arange(w) if keep is None else keep
        self.beta = ridge_fit(h[:, self.keep], ytr)
        self.h_train = h

    def loss(self, x, y):
        h = features(x, self.w, self.depth, self.weights)[:, self.keep]
        return (predict(h, self.beta) - y) ** 2

    def importance_order(self):
        """Units ordered from least to most important, computed on D_train."""
        sd = self.h_train[:, self.keep].std(axis=0)
        imp = np.abs(self.beta[:-1]) * sd
        return self.keep[np.argsort(imp, kind="stable")]


def pruned(xtr, ytr, full, frac):
    """Remove the least important fraction of last-layer units and refit."""
    order = full.importance_order()
    k = int(round(frac * full.w))
    keep = np.sort(order[k:])
    return Fitted(xtr, ytr, full.w, full.depth, full.weights, keep=keep)
