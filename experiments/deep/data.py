"""Data for the deep experiments.

Each task has one fixed training sample S and two disjoint held-out sets:
  dec   -- the pool from which decision/validation samples are resampled,
  truth -- used only to report the generalisation error of selected networks.
"""
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------ synthetic MLP
D_MLP = 10
SIGMA = 0.5


def f_mlp(x):
    return (np.sin(2 * x[:, 0]) + x[:, 1] * x[:, 2] + 0.5 * np.cos(3 * x[:, 3])
            + 0.3 * x[:, 4] ** 2 + 0.5 * np.tanh(x[:, 5] + x[:, 6])
            + 0.2 * x[:, 7] * x[:, 8] * x[:, 9])


def _draw(n, rng):
    x = rng.standard_normal((n, D_MLP))
    return x, f_mlp(x) + SIGMA * rng.standard_normal(n)


def load_synthetic(n_train, n_dec, n_truth, device, split_seed=0):
    rng = np.random.default_rng(split_seed)
    out = []
    for n in (n_train, n_dec, n_truth):
        x, y = _draw(n, rng)
        out.append((torch.tensor(x, dtype=torch.float32, device=device),
                    torch.tensor(y, dtype=torch.float32, device=device)))
    return out  # train, dec, truth


# ----------------------------------------------------------------- CIFAR-10
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar10(root, n_train, device, split_seed=0):
    """Fixed training subset of size n_train; the 10k test images are split
    into a decision pool (5k) and a truth set (5k). Images stay uint8 on device."""
    import torchvision
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(tr.data))[:n_train]

    def to_t(x, y):
        return (torch.tensor(x).permute(0, 3, 1, 2).contiguous().to(device),
                torch.tensor(np.asarray(y), dtype=torch.long).to(device))

    train = to_t(tr.data[idx], np.asarray(tr.targets)[idx])
    p = rng.permutation(len(te.data))
    h = len(p) // 2
    yte = np.asarray(te.targets)
    dec = to_t(te.data[p[:h]], yte[p[:h]])
    truth = to_t(te.data[p[h:]], yte[p[h:]])
    return train, dec, truth


def normalize(x_uint8):
    x = x_uint8.float().div_(255.0)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def augment(x, gen):
    """Random 32x32 crop from 4-pixel zero padding and horizontal flip, on device."""
    N = x.shape[0]
    dev = x.device
    pad = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (N,), generator=gen, device=dev)
    j = torch.randint(0, 9, (N,), generator=gen, device=dev)
    ar = torch.arange(32, device=dev)
    rows = (i.view(N, 1) + ar.view(1, 32))[:, None, :, None]
    cols = (j.view(N, 1) + ar.view(1, 32))[:, None, None, :]
    n_idx = torch.arange(N, device=dev)[:, None, None, None]
    c_idx = torch.arange(3, device=dev)[None, :, None, None]
    out = pad[n_idx, c_idx, rows, cols]
    flip = torch.rand(N, generator=gen, device=dev) < 0.5
    return torch.where(flip.view(N, 1, 1, 1), out.flip(3), out)
