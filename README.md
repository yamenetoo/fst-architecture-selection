<p align="center">
  <img src="assets/logo.svg" alt="FST-Arch logo" width="560">
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/yamenetoo/fst-architecture-selection/blob/main/notebooks/run_all_experiments.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
</p>

# FST-Arch: growing and pruning neural networks with family-wise error control

Simulation code for the article

> **Growing and pruning neural networks by fixed-sequence testing with family-wise error control**
> Mohamad Yamen AL-Mohamad — Department of Statistics, University of Tabriz (submitted).

The repository contains only the code needed to reproduce every number in the article's simulation section (Tables 1 and 2). The article itself is not included.

## The idea in one paragraph

Choosing the width and depth of a neural network is a sequence of decisions: *is the next larger network really better?* (growing) or *does the smaller network lose little?* (pruning). We treat every step as a one-sided hypothesis test on the **architecture risk** $\mu(a)$ — the expected loss over the data **and** over the training randomness (seed):

$$H_0:\ \mu(b)-\mu(a)\ \ge\ c \qquad\text{vs}\qquad H_1:\ \mu(b)-\mu(a) < c ,$$

and move from $a$ to its neighbour $b$ only when $H_0$ is rejected.

| Direction | constant $c$ | rejecting $H_0$ means |
|---|---|---|
| Growing | $-\gamma$ | the larger network improves the risk by more than $\gamma$ |
| Pruning | $\delta$ (vs. the **full** network $a_0$) | the pruned network loses less than $\delta$ |

Tests run along a chain of architectures fixed in advance, each at level $\alpha$, and stop at the first non-rejection (**fixed-sequence testing**). This gives:

- **FWER ≤ α** with no multiplicity correction, for any dependence between the tests;
- for pruning, $P\{\mu(\hat a)-\mu(a_0)\ge\delta\}\le\alpha$: the final network loses less than $\delta$ with probability at least $1-\alpha$;
- validity for the two-stage *width-then-depth* search (FWER ≤ α₁ + α₂ with independent validation splits) and for any safety check that only cancels moves.

Each test uses a **paired difference array** $D_{ri}$ (seed $r$ × validation point $i$). Its variance has three components (seed, point, interaction), and we estimate it without bias by

$$\hat V=\frac{\mathrm{MS}_Z+\mathrm{MS}_\xi-\mathrm{MS}_E}{nR},$$

with Satterthwaite degrees of freedom. The usual *seed-only* and *single-seed* standard errors each drop one component.

## Repository layout

```
experiments/
  rfnet.py    random-feature ReLU networks (nested across widths), ridge output layer, pruning
  oracle.py   Monte Carlo "true" risks mu(a) for every architecture and pruning level
  sim.py      replications of the growing and pruning procedures; writes the tables
notebooks/
  run_all_experiments.ipynb   runs everything on Google Colab
assets/logo.svg
```

## Run on Google Colab

Click the **Open in Colab** badge above and run all cells. The notebook clones this repository, computes the oracle risks, lets you set γ and δ, runs the replications, and shows the tables and plots.

## Run locally

```bash
git clone https://github.com/yamenetoo/fst-architecture-selection.git
cd fst-architecture-selection
pip install -r requirements.txt
cd experiments
python oracle.py                 # step 1: true risks mu(a); prints the risk table
python sim.py 1000 0.005 0.01    # step 2: B replications, gamma, delta
```

`oracle.py` writes `oracle.npz`. `sim.py` prints which null hypotheses are true for the chosen γ and δ, then runs the replications. Choose γ and δ so that the width chain contains both true and false nulls.

### Outputs of `sim.py`

| File | Content |
|---|---|
| `results.json` | all summaries (FWER, size, selected width/depth, excess risk, pruning loss) |
| `table_building.tex` | rows of Table 1 (growing), ready to paste into LaTeX |
| `table_pruning.tex` | rows of Table 2 (pruning) |

### What is compared

| Name | Standard error used in the test |
|---|---|
| `crossed` (proposed) | crossed seed × point estimator $\hat V$, Satterthwaite df |
| `crossed+safety` | same, plus a safety check on an independent split $V_2$ |
| `seed_only` | sd of the $R$ seed means $/\sqrt R$ |
| `single` | paired per-point SE from one seed ($R=1$) |
| `welch` | unpaired SE (ignores the pairing) |

For pruning, the **cumulative** margin (every pruned network vs. the full one) is compared with a **stepwise** margin (each network vs. the previous one).

### Simulation design

- Data: $Y=\sin(2X_1)+X_2X_3+0.5\cos(3X_4)+0.3X_5^2+\varepsilon$, $X\sim N_5(0,I)$, $\varepsilon\sim N(0,0.5^2)$; one fixed training sample of size 600.
- Networks: random-feature ReLU networks, widths 4–256, depths 1–3; only the output layer is fitted (ridge), so the seed is the only training randomness.
- Per replication: new validation splits $V_1^a, V_1^b$ (250 each) and $V_2$ (500), $R=10$ seeds per stage, $\alpha=0.05$.
- Pruning: from width 256, remove 10%, 20%, …, 90% of the units ranked on the training data, then refit.

## Citation

If you use this code, please cite the article (details will be added after publication).

## License

MIT — see [LICENSE](LICENSE).
