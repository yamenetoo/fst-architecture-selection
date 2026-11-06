"""Check that the GPU backend (rfgpu.py) reproduces the CPU reference (rfnet.py).

Fits a few seeds for several architectures and for pruned networks with both
implementations and compares the test losses. Run it once before the
experiments:
    python check_gpu.py [--device cuda]
"""
import argparse

import numpy as np
import torch

from rfgpu import RF
from rfnet import Fitted, draw_data, pruned, seed_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    xtr, ytr = draw_data(600, np.random.default_rng(2026))
    xte, yte = draw_data(2000, np.random.default_rng(7))
    rf = RF(xtr, ytr, a.device)
    print(f"device: {rf.dev}, dtype: {rf.dt}"
          + (f", GPU: {torch.cuda.get_device_name(rf.dev)}" if rf.dev.type == "cuda" else ""))
    seeds = [11, 222, 3333]
    W = rf.weights(seeds)
    xt, yt = rf.t(xte), rf.t(yte)
    worst = 0.0
    for w, d in [(8, 1), (64, 2), (256, 3)]:
        gpu = rf.loss_np(rf.fit(w, d, W), xt, yt).mean(1)
        cpu = np.array([Fitted(xtr, ytr, w, d, seed_weights(s)).loss(xte, yte).mean() for s in seeds])
        diff = np.abs(gpu - cpu).max() / np.abs(cpu).max()
        worst = max(worst, diff)
        print(f"  width {w:>3}, depth {d}: relative difference {diff:.1e}")
    full = rf.fit(256, 1, W)
    for fr in [0.3, 0.7]:
        gpu = rf.loss_np(rf.pruned(full, fr), xt, yt).mean(1)
        cpu = np.array([pruned(xtr, ytr, Fitted(xtr, ytr, 256, 1, seed_weights(s)), fr).loss(xte, yte).mean()
                        for s in seeds])
        diff = np.abs(gpu - cpu).max() / np.abs(cpu).max()
        worst = max(worst, diff)
        print(f"  pruned {fr:.0%} of width 256: relative difference {diff:.1e}")
    tol = 1e-6 if rf.dt == torch.float64 else 1e-3
    print("OK: GPU and CPU agree" if worst < tol else
          f"WARNING: largest relative difference {worst:.1e} exceeds {tol:.0e}")


if __name__ == "__main__":
    main()
