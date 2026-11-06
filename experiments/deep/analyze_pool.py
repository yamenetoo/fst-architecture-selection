"""Monte Carlo study of the procedures on a pool of trained networks.

Design (pool bootstrap). The pool holds, for every architecture a and seed
s = 1..M, the losses of the trained network at every point of the decision pool.
One replication draws R seeds and n points *with replacement* from the pool.
They are then i.i.d. draws from the pool's empirical distribution, so
Assumption A1 of the article holds exactly. The true risk in this world is the
pool mean mu(a), which is known, and false rejections can be counted exactly.

Studies:
  1. exact size of each standard error at the boundary c = theta(a -> b),
     over every edge of the width x depth lattice and a grid of (n, R);
  2. two-stage growing (width chain, then depth chain): FWER, selected
     architecture, excess risk;
  3. pruning the largest network: cumulative vs stepwise margin.

Example
  python analyze_pool.py --task resnet --loss err --gamma 0.005 --delta 0.01
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import reject  # noqa: E402  (same tests as the simulation in the article)

METHODS = ["crossed", "seed_only", "single", "welch"]
NAMES = {"crossed": "Crossed (proposed)", "crossed+safety": "Crossed + safety check",
         "seed_only": "Seed-only SE", "single": "Single seed", "welch": "Unpaired (Welch)"}


def load_pool(pool, task, loss):
    rx = re.compile(r"_w(\d+)_d(\d+)_s(\d+)\.npz$")
    recs = {}
    for f in glob.glob(os.path.join(pool, f"{task}_w*_d*_s*.npz")):
        m = rx.search(f)
        if m:
            recs[tuple(map(int, m.groups()))] = dict(np.load(f))
    if not recs:
        sys.exit(f"no pool files for task '{task}' in {pool}")
    archs = sorted({(w, d) for (w, d, _) in recs})
    seeds = sorted(set.intersection(*[{s for (w, d, s) in recs if (w, d) == a} for a in archs]))
    widths = sorted({w for w, _ in archs})
    depths = sorted({d for _, d in archs})
    grid = [(w, d) for w in widths for d in depths if (w, d) in archs]
    # keep the widths that have every depth (the pruning architecture may be extra)
    widths = [w for w in widths if all((w, d) in archs for d in depths)]
    L = {a: np.stack([recs[(*a, s)][f"dec_{loss}"] for s in seeds]).astype(np.float64)
         for a in grid}
    T = {a: np.stack([recs[(*a, s)][f"truth_{loss}"] for s in seeds]).astype(np.float64)
         for a in grid}
    prune = None
    for (w, d, s), r in recs.items():
        if f"prune_dec_{loss}" in r:
            pa = (w, d)
            prune = dict(arch=pa, fracs=r["fracs"],
                         P=np.stack([recs[(*pa, s)][f"prune_dec_{loss}"] for s in seeds]).astype(np.float64),
                         PT=np.stack([recs[(*pa, s)][f"prune_truth_{loss}"] for s in seeds]).astype(np.float64))
            break
    params = {a: int(recs[(*a, seeds[0])]["n_params"]) for a in grid}
    return dict(L=L, T=T, widths=widths, depths=depths, seeds=seeds, prune=prune, params=params)


def sub(X, s, p):
    return X[np.ix_(s, p)]


def grow(idx, L, mu, widths, depths, gamma, alpha, method, safety):
    (s1, p1), (s2, p2), p3 = idx
    false, hits = False, 0
    i, d0 = 0, depths[0]
    while i + 1 < len(widths):
        a, b = (widths[i], d0), (widths[i + 1], d0)
        if not reject(method, sub(L[a], s1, p1), sub(L[b], s1, p1), -gamma, alpha / 2):
            break
        if mu[b] - mu[a] >= -gamma:
            false = True
        if safety and (sub(L[b], s1, p3) - sub(L[a], s1, p3)).mean() > 0:
            hits += 1
            break
        i += 1
    w, j = widths[i], 0
    while j + 1 < len(depths):
        a, b = (w, depths[j]), (w, depths[j + 1])
        if not reject(method, sub(L[a], s2, p2), sub(L[b], s2, p2), -gamma, alpha / 2):
            break
        if mu[b] - mu[a] >= -gamma:
            false = True
        if safety and (sub(L[b], s2, p3) - sub(L[a], s2, p3)).mean() > 0:
            hits += 1
            break
        j += 1
    return false, (w, depths[j]), hits


def prune_chain(idx, P, mu_p, delta, alpha, kind):
    s, p = idx
    k, false = 0, False
    while k + 1 < P.shape[1]:
        ref = 0 if kind == "cumulative" else k
        if not reject("crossed", sub(P[:, ref, :], s, p), sub(P[:, k + 1, :], s, p), delta, alpha):
            break
        if mu_p[k + 1] - mu_p[ref] >= delta:
            false = True
        k += 1
    return k, false


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["mlp", "resnet"], required=True)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--loss", default=None, help="se (mlp); err or ce (resnet)")
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--n", type=int, default=1000, help="decision sample size")
    ap.add_argument("--n2", type=int, default=1000, help="safety sample size")
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--B-grid", type=int, default=500)
    ap.add_argument("--grid-n", type=int, nargs="+", default=[100, 250, 1000, 2500])
    ap.add_argument("--grid-R", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    a.pool = a.pool or f"pool_{a.task}"
    a.loss = a.loss or ("se" if a.task == "mlp" else "err")
    rng = np.random.default_rng(a.seed)

    pool = load_pool(a.pool, a.task, a.loss)
    L, T, widths, depths = pool["L"], pool["T"], pool["widths"], pool["depths"]
    M, nd = next(iter(L.values())).shape
    mu = {k: v.mean() for k, v in L.items()}
    mu_truth = {k: v.mean() for k, v in T.items()}
    if max(a.grid_R + [a.R]) > M:
        print(f"warning: R larger than the pool's {M} seeds (seeds are resampled with replacement)")

    print(f"pool: {M} seeds, {nd} decision points, loss = {a.loss}")
    print("pool risk mu(w, d) [truth-set risk] and number of parameters")
    print("width " + " ".join(f"{'d=' + str(d):>26}" for d in depths))
    for w in widths:
        print(f"{w:>5} " + " ".join(f"{mu[(w, d)]:>9.4f} [{mu_truth[(w, d)]:.4f}] {pool['params'][(w, d)]:>8,}"
                                    for d in depths))
    print(f"\nGrowing nulls (H0 true if theta >= -gamma = {-a.gamma})")
    for d in depths[:1]:
        for i in range(len(widths) - 1):
            th = mu[(widths[i + 1], d)] - mu[(widths[i], d)]
            print(f"  width {widths[i]}->{widths[i + 1]} (d={d}): theta={th:+.4f}  "
                  f"H0 {'TRUE' if th >= -a.gamma else 'false'}")
    for w in widths:
        for j in range(len(depths) - 1):
            th = mu[(w, depths[j + 1])] - mu[(w, depths[j])]
            print(f"  depth {depths[j]}->{depths[j + 1]} (w={w}): theta={th:+.4f}  "
                  f"H0 {'TRUE' if th >= -a.gamma else 'false'}")

    out = {"task": a.task, "loss": a.loss, "M": M, "n_dec_pool": nd, "gamma": a.gamma,
           "delta": a.delta, "alpha": a.alpha, "B": a.B, "n": a.n, "R": a.R,
           "mu": {f"{w},{d}": mu[(w, d)] for (w, d) in mu},
           "mu_truth": {f"{w},{d}": mu_truth[(w, d)] for (w, d) in mu_truth}}

    # ---------------------------------------------------- 1. exact size grid
    edges = [((widths[i], d), (widths[i + 1], d)) for d in depths for i in range(len(widths) - 1)]
    edges += [((w, depths[j]), (w, depths[j + 1])) for w in widths for j in range(len(depths) - 1)]
    maxR, maxn = max(a.grid_R), max(a.grid_n)
    acc = {(n, R, m): np.zeros(len(edges)) for n in a.grid_n for R in a.grid_R for m in METHODS}
    for b in range(a.B_grid):
        s = rng.integers(0, M, maxR)
        p = rng.integers(0, nd, maxn)
        for n in a.grid_n:
            for R in a.grid_R:
                ss, pp = s[:R], p[:n]
                for m in METHODS:
                    acc[(n, R, m)] += [reject(m, sub(L[e0], ss, pp), sub(L[e1], ss, pp),
                                              mu[e1] - mu[e0], a.alpha) for e0, e1 in edges]
        if (b + 1) % 100 == 0:
            print(f"size grid {b + 1}/{a.B_grid}", flush=True)
    out["size"] = {f"n={n},R={R}": {m: dict(mean=float(acc[(n, R, m)].mean() / a.B_grid),
                                            max=float(acc[(n, R, m)].max() / a.B_grid))
                                    for m in METHODS}
                   for n in a.grid_n for R in a.grid_R}

    # --------------------------------------------------------- 2. growing
    res = {m: [] for m in METHODS + ["crossed+safety"]}
    for b in range(a.B):
        idx = ((rng.integers(0, M, a.R), rng.integers(0, nd, a.n)),
               (rng.integers(0, M, a.R), rng.integers(0, nd, a.n)),
               rng.integers(0, nd, a.n2))
        for m in METHODS:
            res[m].append(grow(idx, L, mu, widths, depths, a.gamma, a.alpha, m, False))
        res["crossed+safety"].append(grow(idx, L, mu, widths, depths, a.gamma, a.alpha,
                                          "crossed", True))
    best = min(mu.values())
    out["grow"] = {}
    for m, r in res.items():
        f = np.mean([x[0] for x in r])
        out["grow"][m] = dict(
            fwer=f, fwer_se=float(np.sqrt(f * (1 - f) / a.B)),
            width=float(np.mean([x[1][0] for x in r])),
            depth=float(np.mean([x[1][1] for x in r])),
            excess=float(np.mean([mu[x[1]] - best for x in r])),
            excess_sd=float(np.std([mu[x[1]] - best for x in r], ddof=1)),
            truth_risk=float(np.mean([mu_truth[x[1]] for x in r])),
            safety=float(np.mean([x[2] > 0 for x in r])))

    # ---------------------------------------------------------- 3. pruning
    pr = pool["prune"]
    if pr is not None:
        mu_p = pr["P"].mean(axis=(0, 2))
        mu_pt = pr["PT"].mean(axis=(0, 2))
        fr = pr["fracs"]
        print(f"\nPruning {pr['arch']}: total loss mu(a_k)-mu(a_0), step loss, truth-set total")
        for k in range(1, len(fr)):
            print(f"  {fr[k]:.2f}: total={mu_p[k] - mu_p[0]:+.4f}  step={mu_p[k] - mu_p[k - 1]:+.4f}  "
                  f"truth={mu_pt[k] - mu_pt[0]:+.4f}")
        pres = {"cumulative": [], "stepwise": []}
        for b in range(a.B):
            idx = (rng.integers(0, M, a.R), rng.integers(0, nd, a.n))
            for kind in pres:
                pres[kind].append(prune_chain(idx, pr["P"], mu_p, a.delta, a.alpha, kind))
        out["prune"] = {"arch": list(pr["arch"]), "fracs": fr.tolist(),
                        "mu": mu_p.tolist(), "mu_truth": mu_pt.tolist()}
        for kind, r in pres.items():
            exc = np.mean([mu_p[k] - mu_p[0] >= a.delta for k, _ in r])
            out["prune"][kind] = dict(
                harm_exceeds=float(exc), harm_exceeds_se=float(np.sqrt(exc * (1 - exc) / a.B)),
                frac=float(np.mean([fr[k] for k, _ in r])),
                harm=float(np.mean([mu_p[k] - mu_p[0] for k, _ in r])),
                harm_truth=float(np.mean([mu_pt[k] - mu_pt[0] for k, _ in r])),
                fwer=float(np.mean([f for _, f in r])))

    # ------------------------------------------------------------ outputs
    tag = f"{a.task}_{a.loss}"
    with open(f"results_deep_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(f"table_deep_size_{tag}.tex", "w") as fh:
        for n in a.grid_n:
            for R in a.grid_R:
                row = out["size"][f"n={n},R={R}"]
                fh.write(f"{n} & {R} & " + " & ".join(
                    f"{row[m]['mean']:.3f} ({row[m]['max']:.3f})" for m in METHODS) + " \\\\\n")
    with open(f"table_deep_grow_{tag}.tex", "w") as fh:
        for m in ["crossed", "crossed+safety", "seed_only", "single", "welch"]:
            g = out["grow"][m]
            fh.write(f"{NAMES[m]} & {g['fwer']:.3f} ({g['fwer_se']:.3f}) & {g['width']:.1f} & "
                     f"{g['depth']:.2f} & {g['excess']:.4f} ({g['excess_sd']:.4f}) & "
                     f"{g['truth_risk']:.4f} \\\\\n")
    if pr is not None:
        with open(f"table_deep_prune_{tag}.tex", "w") as fh:
            for kind, nm in [("cumulative", "Cumulative margin (proposed)"), ("stepwise", "Stepwise margin")]:
                g = out["prune"][kind]
                fh.write(f"{nm} & {g['harm_exceeds']:.3f} ({g['harm_exceeds_se']:.3f}) & "
                         f"{g['frac']:.3f} & {g['harm']:.4f} & {g['harm_truth']:.4f} \\\\\n")

    print(f"\nExact size at the boundary (nominal {a.alpha}); mean over {len(edges)} lattice edges (max)")
    print(f"{'n':>6} {'R':>3} " + " ".join(f"{m:>16}" for m in METHODS))
    for n in a.grid_n:
        for R in a.grid_R:
            row = out["size"][f"n={n},R={R}"]
            print(f"{n:>6} {R:>3} " + " ".join(f"{row[m]['mean']:>8.3f} ({row[m]['max']:.3f})" for m in METHODS))
    print("\nGrowing")
    for m, g in out["grow"].items():
        print(f"  {m:<15} FWER={g['fwer']:.3f}  width={g['width']:.1f}  depth={g['depth']:.2f}  "
              f"excess={g['excess']:.4f}  truth={g['truth_risk']:.4f}  safety={g['safety']:.3f}")
    if pr is not None:
        print("Pruning")
        for kind in ["cumulative", "stepwise"]:
            g = out["prune"][kind]
            print(f"  {kind:<11} P(loss>=delta)={g['harm_exceeds']:.3f}  frac={g['frac']:.3f}  "
                  f"loss={g['harm']:+.4f}  truth={g['harm_truth']:+.4f}")
    print(f"\nwrote results_deep_{tag}.json and table_deep_*_{tag}.tex")


if __name__ == "__main__":
    main()
