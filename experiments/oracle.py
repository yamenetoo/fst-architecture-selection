"""Monte Carlo 'oracle' risks mu_S(a) for the fixed training sample S (GPU).

mu_S(a) = E_xi E_(x,y) loss, estimated with N_SEEDS hidden-layer seeds and a
large independent test sample. Seeds are processed in batches on the GPU.
Saved to oracle.npz.

    python oracle.py [--out oracle.npz] [--device cuda] [--seed-batch 25]
"""
import argparse
import time

import numpy as np
import torch

from rfgpu import RF
from rfnet import draw_data

S_SEED = 2026
N_TRAIN = 600
N_TEST = 40000
N_SEEDS = 300
WIDTHS = [4, 8, 16, 32, 64, 128, 256]
DEPTHS = [1, 2, 3]
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Pruning scenario B: start from a network that does not overfit, so pruning
# costs risk; small 5% steps let per-step losses stay below delta while the
# total loss grows (the situation Corollary 2 is about).
PRUNE2_W = 64
FRACS2 = [round(0.05 * k, 2) for k in range(19)]   # 0.00, 0.05, ..., 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="oracle.npz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed-batch", type=int, default=25,
                    help="seeds per GPU batch (lower it if the GPU runs out of memory)")
    a = ap.parse_args()

    xtr, ytr = draw_data(N_TRAIN, np.random.default_rng(S_SEED))
    xte, yte = draw_data(N_TEST, np.random.default_rng(10 ** 6))
    rf = RF(xtr, ytr, a.device)
    xt, yt = rf.t(xte), rf.t(yte)
    print(f"device: {rf.dev}, dtype: {rf.dt}")

    risk = np.zeros((N_SEEDS, len(WIDTHS), len(DEPTHS)))
    prune = np.zeros((N_SEEDS, len(FRACS)))
    prune2 = np.zeros((N_SEEDS, len(FRACS2)))
    seeds = [10 ** 7 + s for s in range(N_SEEDS)]
    t0 = time.time()
    with torch.no_grad():
        for c0 in range(0, N_SEEDS, a.seed_batch):
            sl = slice(c0, min(c0 + a.seed_batch, N_SEEDS))
            W = rf.weights(seeds[sl])
            for i, w in enumerate(WIDTHS):
                for j, d in enumerate(DEPTHS):
                    risk[sl, i, j] = rf.loss(rf.fit(w, d, W), xt, yt).mean(1).cpu().numpy()
            full = rf.fit(256, 1, W)
            for k, fr in enumerate(FRACS):
                m = full if fr == 0 else rf.pruned(full, fr)
                prune[sl, k] = rf.loss(m, xt, yt).mean(1).cpu().numpy()
            full2 = rf.fit(PRUNE2_W, 1, W)
            for k, fr in enumerate(FRACS2):
                m = full2 if fr == 0 else rf.pruned(full2, fr)
                prune2[sl, k] = rf.loss(m, xt, yt).mean(1).cpu().numpy()
            print(f"seeds {sl.stop}/{N_SEEDS}  {time.time() - t0:.0f}s", flush=True)

    np.savez(a.out, risk=risk, prune=prune, prune2=prune2, fracs2=FRACS2,
             widths=WIDTHS, depths=DEPTHS, fracs=FRACS)
    mu = risk.mean(0)
    se = risk.std(0, ddof=1) / np.sqrt(N_SEEDS)
    print("mu_S(w, d)  (rows: width, cols: depth 1..3), MC s.e. in brackets")
    for i, w in enumerate(WIDTHS):
        print(w, " ".join(f"{mu[i, j]:.4f}[{se[i, j]:.4f}]" for j in range(3)))
    pm = prune.mean(0)
    print("pruning (256,1): mu(a_k) - mu(a_0) by fraction")
    for k, fr in enumerate(FRACS):
        dk = prune[:, k] - prune[:, 0]
        print(fr, f"{pm[k]:.4f}", f"diff={dk.mean():.4f}[{dk.std(ddof=1)/np.sqrt(N_SEEDS):.4f}]")
    print(f"pruning scenario B ({PRUNE2_W},1): total loss mu(a_k) - mu(a_0) and step loss mu(a_k) - mu(a_k-1)")
    p2 = prune2.mean(0)
    for k, fr in enumerate(FRACS2):
        step = p2[k] - p2[k - 1] if k else 0.0
        print(f"{fr:.2f} total={p2[k] - p2[0]:+.4f} step={step:+.4f}")
    print(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
