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

## 🚀 Quick start: run everything on Google Colab

<a href="https://colab.research.google.com/github/yamenetoo/fst-architecture-selection/blob/main/notebooks/run_all_experiments.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="28"></a>

One click opens [`notebooks/run_all_experiments.ipynb`](notebooks/run_all_experiments.ipynb) in Colab. Then choose **Runtime → Run all**. No GPU and no installation are needed.

Direct link: https://colab.research.google.com/github/yamenetoo/fst-architecture-selection/blob/main/notebooks/run_all_experiments.ipynb

**Trained MLPs and ResNets (GPU):**
<a href="https://colab.research.google.com/github/yamenetoo/fst-architecture-selection/blob/main/notebooks/deep_experiments.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="28"></a>
— [`notebooks/deep_experiments.ipynb`](notebooks/deep_experiments.ipynb). Choose a GPU runtime first.

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
  sim.py      replications of the growing and pruning procedures; writes Tables 1-2
  grid.py     exact size of each standard error over validation sizes n and seeds R; writes Table 3
  deep/
    models.py         trained MLP and CIFAR ResNet (ResNet-8/14/20/32, any width) with pruning masks
    data.py           synthetic regression data and CIFAR-10 with GPU data augmentation
    train_pool.py     trains every architecture with M shared seeds; stores per-point losses (resumable, multi-GPU)
    analyze_pool.py   pool-bootstrap study: exact size, growing FWER/power, pruning
notebooks/
  run_all_experiments.ipynb   random-feature simulation of the article (CPU)
  deep_experiments.ipynb      trained MLPs and ResNets (GPU)
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
python grid.py 500               # step 3: exact size over n in {50,100,250,1000}, R in {5,10,20}
```

`oracle.py` writes `oracle.npz`. `sim.py` prints which null hypotheses are true for the chosen γ and δ, then runs the replications. Choose γ and δ so that the width chain contains both true and false nulls.

### Outputs of `sim.py`

| File | Content |
|---|---|
| `results.json` | all summaries (FWER, size, selected width/depth, excess risk, pruning loss) |
| `table_building.tex` | rows of Table 1 (growing), ready to paste into LaTeX |
| `table_pruning.tex` | rows of Table 2 (pruning, scenarios A and B) |
| `grid.json`, `table_grid.tex` | output of `grid.py`: rows of Table 3 (exact size by n and R) |

### What is compared

| Name | Standard error used in the test |
|---|---|
| `crossed` (proposed) | crossed seed × point estimator $\hat V$, Satterthwaite df |
| `crossed+safety` | same, plus a safety check on an independent split $V_2$ |
| `seed_only` | sd of the $R$ seed means $/\sqrt R$ |
| `single` | paired per-point SE from one seed ($R=1$) |
| `welch` | unpaired SE (ignores the pairing) |

**Exact size.** Besides the FWER, every width pair is tested at the exact boundary $c=\theta_{\text{true}}$, where a calibrated one-sided test rejects with probability $\alpha$. `grid.py` repeats this over $n$ and $R$: theory predicts that the seed-only error over-rejects when $n$ is small or $R$ is large, and the single-seed error at every $n$.

For pruning, the **cumulative** margin (every pruned network vs. the full one) is compared with a **stepwise** margin (each network vs. the previous one).

### Simulation design

- Data: $Y=\sin(2X_1)+X_2X_3+0.5\cos(3X_4)+0.3X_5^2+\varepsilon$, $X\sim N_5(0,I)$, $\varepsilon\sim N(0,0.5^2)$; one fixed training sample of size 600.
- Networks: random-feature ReLU networks, widths 4–256, depths 1–3; only the output layer is fitted (ridge), so the seed is the only training randomness.
- Per replication: new validation splits $V_1^a, V_1^b$ (250 each) and $V_2$ (500), $R=10$ seeds per stage, $\alpha=0.05$.
- Pruning, scenario A: from width 256 (which overfits), remove 10%, 20%, …, 90% of the units ranked on the training data, then refit.
- Pruning, scenario B: from width 64, where pruning costs risk, remove 5%, 10%, …, 90% in small steps, so per-step losses stay below δ while the total loss grows.

## Experiments with trained networks (MLPs and ResNets)

The simulation above uses random-feature networks so that thousands of fits are cheap. The `experiments/deep` folder runs the same procedures on networks trained end to end.

| Task | Architectures | Training | Loss |
|---|---|---|---|
| `mlp` | widths 16–1024, depths 1–4 (up to about 3M parameters); regression with $d=10$ inputs | Adam, 5000 training points | squared error |
| `resnet` | CIFAR ResNet, base width $k\in\{8,16,32,64\}$, $n\in\{1,2,3\}$ blocks per stage (ResNet-8/14/20); `--widths 128 --depths 5` for larger models | SGD + one-cycle, crop/flip augmentation, 20k CIFAR-10 images | 0-1 error (or cross-entropy) |

**Pool bootstrap.** Each architecture is trained once per seed, with the **same** $M$ seeds for all architectures, so comparisons are paired. The per-point losses on a held-out decision pool are stored. Each Monte Carlo replication then resamples $R$ seeds and $n$ points *with replacement* from the pool. The draws are i.i.d. from the pool, so the true risks (pool means) are known exactly and no network is retrained per replication. A separate truth set reports the generalisation error of the selected networks.

**Pruning.** The largest network is pruned layer by layer (MLP: hidden units ranked by activation sd × outgoing-weight norm on the training data; ResNet: inner block channels ranked by BatchNorm |γ|, then BatchNorm statistics are re-estimated on training images).

```bash
cd experiments/deep
python train_pool.py --task mlp                                   # bigger MLPs
python train_pool.py --task resnet                                # ResNet-8/14/20 on CIFAR-10
python train_pool.py --task resnet --widths 16 32 64 128 --depths 1 2 3 5 --prune-width 128 --prune-depth 5   # larger
python analyze_pool.py --task mlp    --gamma 0.005 --delta 0.01
python analyze_pool.py --task resnet --loss err --gamma 0.005 --delta 0.01
```

Training is resumable (finished files are skipped). On a machine with two GPUs, run two processes with `--device cuda:0 --shard 0/2` and `--device cuda:1 --shard 1/2`.

## Citation

If you use this code, please cite the article (details will be added after publication).

## License

MIT — see [LICENSE](LICENSE).
