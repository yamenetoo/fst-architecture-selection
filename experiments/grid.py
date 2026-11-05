"""Exact size of the four standard errors across validation sizes n and seeds R.

Every consecutive width pair (depth 1) is tested with H0: theta >= theta_true,
so each null sits exactly on its boundary and a calibrated one-sided test
rejects with probability ALPHA. Theory (Proposition 1): the seed-only error
omits sigma_2^2 / n, so it should over-reject when n is small or R is large;
the single-seed error omits sigma_1^2 at every n.

Run AFTER oracle.py:
    python grid.py [B]          # default B = 500
Writes grid.json and table_grid.tex.
"""
import json
import sys
import numpy as np
from rfnet import draw_data, seed_weights, Fitted
from oracle import S_SEED, N_TRAIN, WIDTHS
from sim import reject

NS = [50, 100, 250, 1000]
RS = [5, 10, 20]
ALPHA = 0.05
METHODS = ["crossed", "seed_only", "single", "welch"]


def main(B=500):
    B = int(B)
    mu = np.load("oracle.npz")["risk"].mean(0)[:, 0]      # depth 1
    theta = np.diff(mu)
    xtr, ytr = draw_data(N_TRAIN, np.random.default_rng(S_SEED))
    rej = {(n, R, m): [] for n in NS for R in RS for m in METHODS}
    for b in range(B):
        rng = np.random.default_rng(5 * 10 ** 8 + b)
        v = draw_data(max(NS), rng)
        wts = [seed_weights(s) for s in rng.integers(0, 2 ** 31, max(RS))]
        L = [np.stack([Fitted(xtr, ytr, w, 1, wt).loss(*v) for wt in wts]) for w in WIDTHS]
        for n in NS:
            for R in RS:
                for m in METHODS:
                    rej[(n, R, m)].append(np.mean([
                        reject(m, L[i][:R, :n], L[i + 1][:R, :n], theta[i], ALPHA)
                        for i in range(len(WIDTHS) - 1)]))
        if (b + 1) % 50 == 0:
            print(f"{b + 1}/{B}")
    out = {f"n={n},R={R}": {m: float(np.mean(rej[(n, R, m)])) for m in METHODS}
           for n in NS for R in RS}
    out["B"], out["alpha"] = B, ALPHA
    with open("grid.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Exact size at the boundary (nominal {ALPHA}), averaged over the {len(theta)} width pairs")
    print(f"{'n':>5} {'R':>3} " + " ".join(f"{m:>10}" for m in METHODS))
    with open("table_grid.tex", "w") as fh:
        for n in NS:
            for R in RS:
                row = out[f"n={n},R={R}"]
                print(f"{n:>5} {R:>3} " + " ".join(f"{row[m]:>10.3f}" for m in METHODS))
                fh.write(f"{n} & {R} & " + " & ".join(f"{row[m]:.3f}" for m in METHODS) + " \\\\\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
