"""GPU (PyTorch) implementation of the random-feature networks in rfnet.py.

All R seeds of one architecture are handled in one batch: features are
(R, n, width) tensors and the R ridge systems are solved with one batched call.
Random numbers (data and hidden-layer weights) are still drawn with numpy from
the same seeds as rfnet.py, so with float64 the results agree with the CPU
reference up to rounding (check_gpu.py verifies this).

Device: --device, or the environment variable FST_DEVICE, or cuda if available.
Precision: float64 by default; FST_DTYPE=float32 is faster on consumer GPUs.
"""
import math
import os

import numpy as np
import torch

from rfnet import D_X, L_MAX, RIDGE, W_MAX, seed_weights


def default_device():
    return os.environ.get("FST_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def default_dtype():
    return torch.float32 if os.environ.get("FST_DTYPE") == "float32" else torch.float64


class Model:
    """R fitted networks with a common architecture (one per seed)."""

    def __init__(self, w, depth, W, beta, h_train, keep=None):
        self.w, self.depth, self.W = w, depth, W
        self.beta, self.h_train, self.keep = beta, h_train, keep


class RF:
    """Batched random-feature ReLU networks for one fixed training sample."""

    def __init__(self, xtr, ytr, device=None, dtype=None):
        self.dev = torch.device(device or default_device())
        self.dt = dtype or default_dtype()
        self.xtr, self.ytr = self.t(xtr), self.t(ytr)

    def t(self, a):
        return torch.as_tensor(np.asarray(a), device=self.dev, dtype=self.dt)

    # ------------------------------------------------------------ weights
    def weights(self, seeds):
        """Hidden-layer weights for a list of seeds, identical to rfnet.seed_weights."""
        per = [seed_weights(int(s)) for s in seeds]
        ws = [self.t(np.stack([p[0][l] for p in per])) for l in range(L_MAX)]
        bs = [self.t(np.stack([p[1][l] for p in per])) for l in range(L_MAX)]
        return ws, bs

    # ----------------------------------------------------------- features
    def features(self, x, w, depth, W):
        """x: (n, d) tensor -> (R, n, w) hidden features of the last layer."""
        ws, bs = W
        h, prev = x, D_X
        for l in range(depth):
            Wl = ws[l][:, :prev, :w]
            if l > 0:
                Wl = Wl * math.sqrt(W_MAX / prev)  # keep He scaling for the sliced fan-in
            h = torch.relu(torch.matmul(h, Wl) + bs[l][:, None, :w])
            prev = w
        return h

    @staticmethod
    def _gather(h, keep):
        if keep is None:
            return h
        return torch.gather(h, 2, keep[:, None, :].expand(-1, h.shape[1], -1))

    def _ridge(self, h):
        R, n, p = h.shape
        H = torch.cat([h, torch.ones(R, n, 1, device=self.dev, dtype=self.dt)], 2)
        Ht = H.transpose(1, 2)
        reg = torch.full((p + 1,), RIDGE * n, device=self.dev, dtype=self.dt)
        reg[-1] = 0.0                                   # intercept not penalised
        A = Ht @ H + torch.diag(reg)
        rhs = Ht @ self.ytr.view(1, n, 1)
        return torch.linalg.solve(A, rhs).squeeze(-1)  # (R, p + 1)

    # -------------------------------------------------------- fit / predict
    def fit(self, w, depth, W, keep=None):
        h = self._gather(self.features(self.xtr, w, depth, W), keep)
        return Model(w, depth, W, self._ridge(h), h, keep)

    @torch.no_grad()
    def loss(self, m, x, y, chunk=8192):
        """Squared errors, (R, n) tensor. x, y are tensors on the device."""
        out = []
        for k in range(0, x.shape[0], chunk):
            h = self._gather(self.features(x[k:k + chunk], m.w, m.depth, m.W), m.keep)
            pred = (h @ m.beta[:, :-1, None]).squeeze(-1) + m.beta[:, -1:]
            out.append((pred - y[k:k + chunk]) ** 2)
        return torch.cat(out, 1)

    def loss_np(self, m, x, y):
        return self.loss(m, x, y).cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------- pruning
    def pruned(self, full, frac):
        """Remove the least important fraction of last-layer units of an unpruned
        model (importance |beta_j| * sd_S(h_j), computed on the training data) and
        refit. Same rule as rfnet.pruned, applied to every seed at once."""
        assert full.keep is None
        sd = full.h_train.std(dim=1, unbiased=False)
        imp = full.beta[:, :-1].abs() * sd
        order = torch.argsort(imp, dim=1, stable=True)
        k = int(round(frac * full.w))
        keep = torch.sort(order[:, k:], dim=1).values
        return self.fit(full.w, full.depth, full.W, keep=keep)
