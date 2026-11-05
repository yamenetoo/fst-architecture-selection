"""Monte Carlo 'oracle' risks mu_S(a) for the fixed training sample S.

mu_S(a) = E_xi E_(x,y) loss, estimated with N_SEEDS hidden-layer seeds and a
large independent test sample. Saved to oracle.npz.
"""
import sys
import numpy as np
from rfnet import draw_data, seed_weights, Fitted, pruned

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


def main(out="oracle.npz"):
    xtr, ytr = draw_data(N_TRAIN, np.random.default_rng(S_SEED))
    xte, yte = draw_data(N_TEST, np.random.default_rng(10 ** 6))
    risk = np.zeros((N_SEEDS, len(WIDTHS), len(DEPTHS)))
    prune = np.zeros((N_SEEDS, len(FRACS)))
    prune2 = np.zeros((N_SEEDS, len(FRACS2)))
    for s in range(N_SEEDS):
        wts = seed_weights(10 ** 7 + s)
        for i, w in enumerate(WIDTHS):
            for j, d in enumerate(DEPTHS):
                risk[s, i, j] = Fitted(xtr, ytr, w, d, wts).loss(xte, yte).mean()
        full = Fitted(xtr, ytr, 256, 1, wts)
        for k, fr in enumerate(FRACS):
            m = full if fr == 0 else pruned(xtr, ytr, full, fr)
            prune[s, k] = m.loss(xte, yte).mean()
        full2 = Fitted(xtr, ytr, PRUNE2_W, 1, wts)
        for k, fr in enumerate(FRACS2):
            m = full2 if fr == 0 else pruned(xtr, ytr, full2, fr)
            prune2[s, k] = m.loss(xte, yte).mean()
    np.savez(out, risk=risk, prune=prune, prune2=prune2, fracs2=FRACS2, widths=WIDTHS, depths=DEPTHS, fracs=FRACS)
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


if __name__ == "__main__":
    main(*sys.argv[1:])
