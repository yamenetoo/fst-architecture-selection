"""Simulation study for the letter (Section 5).

Run AFTER oracle.py:
    python oracle.py            # writes oracle.npz, prints mu_S(a)
    python sim.py [B] [gamma] [delta]
Defaults: B=1000 replications, gamma=0.005, delta=0.01 (MSE units).
Choose gamma and delta from the oracle printout so that the width chain
contains both false and true null hypotheses (see the printed truth table).
Writes results.json and table_building.tex / table_pruning.tex.
"""
import json
import sys
import numpy as np
from scipy import stats
from rfnet import draw_data, seed_weights, Fitted, pruned
from oracle import S_SEED, N_TRAIN, WIDTHS, DEPTHS, FRACS

R = 10            # seeds per comparison
N1 = 250          # size of V1a and of V1b
N2 = 500          # size of V2 (safety check)
ALPHA = 0.05


# ---------------------------------------------------------------- variance
def crossed_se(D):
    """Unbiased variance of the grand mean of a seeds x samples array
    (Proposition 1) and its Satterthwaite degrees of freedom."""
    Rr, n = D.shape
    m = D.mean()
    rm, cm = D.mean(1), D.mean(0)
    ms_b = n * ((rm - m) ** 2).sum() / (Rr - 1)
    ms_a = Rr * ((cm - m) ** 2).sum() / (n - 1)
    res = D - rm[:, None] - cm[None, :] + m
    ms_e = (res ** 2).sum() / ((Rr - 1) * (n - 1))
    num = ms_a + ms_b - ms_e
    if num <= 0:
        num = max(ms_a, ms_b)
    v = num / (n * Rr)
    df = num ** 2 / (ms_a ** 2 / (n - 1) + ms_b ** 2 / (Rr - 1)
                     + ms_e ** 2 / ((Rr - 1) * (n - 1)))
    return np.sqrt(v), max(df, 1.0)


def reject(method, La, Lb, c, alpha):
    """One-sided test of H0: mu(b) - mu(a) >= c. La, Lb: seeds x samples losses."""
    D = Lb - La
    if method == "crossed":
        se, df = crossed_se(D)
        est = D.mean()
    elif method == "seed_only":
        dr = D.mean(1)
        est, se, df = dr.mean(), dr.std(ddof=1) / np.sqrt(len(dr)), len(dr) - 1
    elif method == "single":
        d = D[0]
        est, se, df = d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d) - 1
    elif method == "welch":
        a, b = La.ravel(), Lb.ravel()
        est = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        df = a.size + b.size - 2
    else:
        raise ValueError(method)
    return (est - c) / se < stats.t.ppf(alpha, df)


# ---------------------------------------------------------------- one replication
def replicate(b, xtr, ytr, mu, mu_pr, gamma, delta):
    rng = np.random.default_rng(10 ** 8 + b)
    v1a, v1b, v2 = draw_data(N1, rng), draw_data(N1, rng), draw_data(N2, rng)
    seeds1 = rng.integers(0, 2 ** 31, R)
    seeds2 = rng.integers(0, 2 ** 31, R)
    w1 = [seed_weights(s) for s in seeds1]
    w2 = [seed_weights(s) for s in seeds2]
    cache = {}

    def losses(w, d, stage, data):
        key = (w, d, stage, id(data))
        if key not in cache:
            wts = w1 if stage == 1 else w2
            cache[key] = np.stack([Fitted(xtr, ytr, w, d, wt).loss(*data) for wt in wts])
        return cache[key]

    out = {}
    nW, nD = len(WIDTHS), len(DEPTHS)

    # calibration: every consecutive width pair at depth 1, tested unconditionally
    calib = {}
    for m in ["crossed", "seed_only", "single", "welch"]:
        calib[m] = [bool(reject(m, losses(WIDTHS[i], 1, 1, v1a),
                                losses(WIDTHS[i + 1], 1, 1, v1a), -gamma, ALPHA / 2))
                    for i in range(nW - 1)]
    out["calib"] = calib

    # building: width chain at depth 1 (alpha/2, V1a), then depth chain (alpha/2, V1b)
    def build(method, safety):
        false_rej, safety_hits = False, 0
        i = 0
        while i + 1 < nW:
            La, Lb = losses(WIDTHS[i], 1, 1, v1a), losses(WIDTHS[i + 1], 1, 1, v1a)
            if not reject(method, La, Lb, -gamma, ALPHA / 2):
                break
            if mu[i + 1, 0] - mu[i, 0] >= -gamma:
                false_rej = True
            if safety:
                d2 = (losses(WIDTHS[i + 1], 1, 1, v2) - losses(WIDTHS[i], 1, 1, v2)).mean()
                if d2 > 0:
                    safety_hits += 1
                    break
            i += 1
        j = 0
        while j + 1 < nD:
            La, Lb = losses(WIDTHS[i], DEPTHS[j], 2, v1b), losses(WIDTHS[i], DEPTHS[j + 1], 2, v1b)
            if not reject(method, La, Lb, -gamma, ALPHA / 2):
                break
            if mu[i, j + 1] - mu[i, j] >= -gamma:
                false_rej = True
            if safety:
                d2 = (losses(WIDTHS[i], DEPTHS[j + 1], 2, v2) - losses(WIDTHS[i], DEPTHS[j], 2, v2)).mean()
                if d2 > 0:
                    safety_hits += 1
                    break
            j += 1
        return dict(fwer=false_rej, w=WIDTHS[i], d=DEPTHS[j],
                    excess=float(mu[i, j] - mu.min()), safety=safety_hits)

    out["build"] = {m: build(m, False) for m in ["crossed", "seed_only", "single", "welch"]}
    out["build"]["crossed+safety"] = build("crossed", True)

    # pruning from a0 = (256, 1): cumulative (proposed) vs stepwise margins
    full = [Fitted(xtr, ytr, 256, 1, wt) for wt in w1]
    pr = {0.0: full}
    for fr in FRACS[1:]:
        pr[fr] = [pruned(xtr, ytr, f, fr) for f in full]
    v1 = (np.vstack([v1a[0], v1b[0]]), np.concatenate([v1a[1], v1b[1]]))
    L = {fr: np.stack([m.loss(*v1) for m in pr[fr]]) for fr in FRACS}

    def prune(kind):
        k = 0
        false_rej = False
        while k + 1 < len(FRACS):
            ref = 0 if kind == "cumulative" else k
            if not reject("crossed", L[FRACS[ref]], L[FRACS[k + 1]], delta, ALPHA):
                break
            if mu_pr[k + 1] - mu_pr[ref] >= delta:
                false_rej = True
            k += 1
        return dict(frac=FRACS[k], harm=float(mu_pr[k] - mu_pr[0]),
                    harm_exceeds=bool(mu_pr[k] - mu_pr[0] >= delta), fwer=false_rej)

    out["prune"] = {k: prune(k) for k in ["cumulative", "stepwise"]}
    return out


# ---------------------------------------------------------------- main
def main(B=1000, gamma=0.005, delta=0.01):
    B, gamma, delta = int(B), float(gamma), float(delta)
    orc = np.load("oracle.npz")
    mu = orc["risk"].mean(0)          # widths x depths
    mu_pr = orc["prune"].mean(0)      # fractions
    xtr, ytr = draw_data(N_TRAIN, np.random.default_rng(S_SEED))

    print("Truth of width-chain nulls (depth 1): H0 true if mu(w+) - mu(w) >= -gamma")
    for i in range(len(WIDTHS) - 1):
        th = mu[i + 1, 0] - mu[i, 0]
        print(f"  {WIDTHS[i]}->{WIDTHS[i+1]}: theta={th:+.4f}  H0 {'TRUE' if th >= -gamma else 'false'}")
    print("Truth of cumulative pruning nulls: H0 true if mu(a_k) - mu(a_0) >= delta")
    for k, fr in enumerate(FRACS[1:], 1):
        print(f"  {fr:.1f}: harm={mu_pr[k]-mu_pr[0]:+.4f}  H0 {'TRUE' if mu_pr[k]-mu_pr[0] >= delta else 'false'}")

    res = [replicate(b, xtr, ytr, mu, mu_pr, gamma, delta) for b in range(B)]
    print(f"done: {B} replications")

    def mc(p):  # Monte Carlo standard error of a proportion
        return np.sqrt(p * (1 - p) / B)

    summ = {"B": B, "gamma": gamma, "delta": delta, "build": {}, "calib": {}, "prune": {}}
    for m in res[0]["build"]:
        r = [x["build"][m] for x in res]
        f = np.mean([x["fwer"] for x in r])
        summ["build"][m] = dict(fwer=f, fwer_se=mc(f),
                                width=np.mean([x["w"] for x in r]),
                                depth=np.mean([x["d"] for x in r]),
                                excess=np.mean([x["excess"] for x in r]),
                                excess_sd=np.std([x["excess"] for x in r], ddof=1),
                                safety=np.mean([x["safety"] > 0 for x in r]))
    true_null = [i for i in range(len(WIDTHS) - 1) if mu[i + 1, 0] - mu[i, 0] >= -gamma]
    if true_null:  # the true null closest to the boundary theta = -gamma
        ib = min(true_null, key=lambda i: mu[i + 1, 0] - mu[i, 0] + gamma)
        summ["calib_pair"] = f"{WIDTHS[ib]}->{WIDTHS[ib+1]}"
        for m in res[0]["calib"]:
            summ["calib"][m] = float(np.mean([x["calib"][m][ib] for x in res]))
    for k in res[0]["prune"]:
        r = [x["prune"][k] for x in res]
        p = np.mean([x["harm_exceeds"] for x in r])
        summ["prune"][k] = dict(harm_exceeds=p, harm_exceeds_se=mc(p),
                                frac=np.mean([x["frac"] for x in r]),
                                harm=np.mean([x["harm"] for x in r]),
                                fwer=np.mean([x["fwer"] for x in r]))
    summ = json.loads(json.dumps(summ, default=float))
    with open("results.json", "w") as fh:
        json.dump(summ, fh, indent=2)
    print(json.dumps(summ, indent=2))

    names = {"crossed": "Crossed (proposed)", "crossed+safety": "Crossed + safety check",
             "seed_only": "Seed-only SE", "single": "Single seed ($R=1$)",
             "welch": "Unpaired (Welch)"}
    with open("table_building.tex", "w") as fh:
        for m in ["crossed", "crossed+safety", "seed_only", "single", "welch"]:
            s = summ["build"][m]
            cal = summ["calib"].get(m.replace("+safety", ""), float("nan"))
            cal_s = "--" if m == "crossed+safety" else f"{cal:.3f}"
            fh.write(f"{names[m]} & {s['fwer']:.3f} ({s['fwer_se']:.3f}) & {cal_s} & "
                     f"{s['width']:.1f} & {s['depth']:.2f} & {s['excess']:.4f} ({s['excess_sd']:.4f}) \\\\\n")
    with open("table_pruning.tex", "w") as fh:
        for k, nm in [("cumulative", "Cumulative margin (proposed)"), ("stepwise", "Stepwise margin")]:
            s = summ["prune"][k]
            fh.write(f"{nm} & {s['harm_exceeds']:.3f} ({s['harm_exceeds_se']:.3f}) & "
                     f"{s['frac']:.3f} & {s['harm']:.4f} \\\\\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
