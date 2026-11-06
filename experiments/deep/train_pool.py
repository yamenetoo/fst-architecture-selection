"""Train a pool of networks: every architecture in a width x depth grid, M seeds each.

For every (width, depth, seed) the per-point losses on the decision pool and on
the truth set are saved to  <out>/<task>_w<W>_d<D>_s<S>.npz . The same seed is
used for every architecture, so it fixes the mini-batch order and the data
augmentation for all of them: comparisons between architectures are paired.
The pruning architecture is also pruned at every fraction (ranking from the
trained weights / training data only) and those losses are saved too.

Jobs whose file already exists are skipped, so an interrupted run resumes.
Split the jobs over several GPUs with --shard i/k (e.g. two processes on Kaggle).

Examples
  python train_pool.py --task mlp
  python train_pool.py --task resnet --widths 16 32 64 --depths 1 2 3 --seeds 10
  python train_pool.py --task resnet --device cuda:1 --shard 1/2
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import D_MLP, augment, load_cifar10, load_synthetic, normalize
from models import MLP, ResNetCIFAR, apply_pruning, n_params, recalibrate_bn

DEFAULTS = {
    "mlp": dict(widths=[16, 32, 64, 128, 256, 512, 1024], depths=[1, 2, 3, 4],
                seeds=10, epochs=60, n_train=5000, prune_width=512, prune_depth=3,
                fracs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "resnet": dict(widths=[8, 16, 32, 64], depths=[1, 2, 3],
                   seeds=10, epochs=15, n_train=20000, prune_width=64, prune_depth=3,
                   fracs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
}


# ------------------------------------------------------------------ training
def train_mlp(width, depth, seed, train, epochs, device, bs=256, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP(D_MLP, width, depth).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    x, y = train
    N = x.shape[0]
    steps = epochs * math.ceil(N / bs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N, generator=gen, device=device)
        for k in range(0, N, bs):
            idx = perm[k:k + bs]
            loss = F.mse_loss(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


def train_resnet(width, depth, seed, train, epochs, device, bs=512, lr=0.2, wd=5e-4):
    torch.manual_seed(seed)
    model = ResNetCIFAR(width, depth).to(device).to(memory_format=torch.channels_last)
    gen = torch.Generator(device=device).manual_seed(seed)
    x, y = train
    N = x.shape[0]
    steps = epochs * math.ceil(N / bs)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True,
                          weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.25)
    use_amp = str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N, generator=gen, device=device)
        for k in range(0, N, bs):
            idx = perm[k:k + bs]
            xb = augment(normalize(x[idx]), gen).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(xb), y[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_losses(task, model, data, bs=2000):
    """Per-point losses. mlp: squared error. resnet: 0-1 error and cross-entropy."""
    x, y = data
    model.eval()
    if task == "mlp":
        return {"se": (model(x) - y).pow(2).float().cpu().numpy()}
    err, ce = [], []
    use_amp = x.is_cuda
    for k in range(0, x.shape[0], bs):
        xb = normalize(x[k:k + bs]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(xb).float()
        yb = y[k:k + bs]
        err.append((logits.argmax(1) != yb).float().cpu())
        ce.append(F.cross_entropy(logits, yb, reduction="none").cpu())
    return {"err": torch.cat(err).numpy(), "ce": torch.cat(ce).numpy()}


def bn_batches(task, train, n_batches=20, bs=1000):
    if task != "resnet":
        return []
    x = train[0]
    return [normalize(x[k:k + bs]).contiguous(memory_format=torch.channels_last)
            for k in range(0, min(x.shape[0], n_batches * bs), bs)]


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["mlp", "resnet"], required=True)
    ap.add_argument("--widths", type=int, nargs="+")
    ap.add_argument("--depths", type=int, nargs="+",
                    help="hidden layers (mlp) or blocks per stage n (resnet, depth 6n+2)")
    ap.add_argument("--seeds", type=int, help="number of seeds M per architecture")
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--prune-width", type=int)
    ap.add_argument("--prune-depth", type=int)
    ap.add_argument("--fracs", type=float, nargs="+")
    ap.add_argument("--out", default=None, help="pool directory (default pool_<task>)")
    ap.add_argument("--data", default="./data_cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th job starting at i")
    a = ap.parse_args()
    for key, val in DEFAULTS[a.task].items():
        if getattr(a, key) is None:
            setattr(a, key, val)
    a.out = a.out or f"pool_{a.task}"
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    if a.task == "mlp":
        train, dec, truth = load_synthetic(a.n_train, 20000, 20000, a.device)
        trainer = train_mlp
    else:
        train, dec, truth = load_cifar10(a.data, a.n_train, a.device)
        trainer = train_resnet

    prune_arch = (a.prune_width, a.prune_depth)
    archs = sorted({(w, d) for w in a.widths for d in a.depths} | {prune_arch})
    jobs = [(w, d, s) for (w, d) in archs for s in range(a.seeds)]
    i, k = map(int, a.shard.split("/"))
    jobs = jobs[i::k]
    print(f"{len(jobs)} jobs on {a.device}; pool -> {a.out}")

    for (w, d, s) in jobs:
        path = os.path.join(a.out, f"{a.task}_w{w}_d{d}_s{s}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        model = trainer(w, d, a.seed_offset + s, train, a.epochs, a.device)
        rec = {f"dec_{m}": v for m, v in eval_losses(a.task, model, dec).items()}
        rec.update({f"truth_{m}": v for m, v in eval_losses(a.task, model, truth).items()})
        rec["n_params"] = n_params(model)
        if (w, d) == prune_arch:
            scores = model.unit_scores(train[0] if a.task == "mlp" else None)
            base = copy.deepcopy(model.state_dict())
            batches = bn_batches(a.task, train)
            pd, pt = {}, {}
            for fr in a.fracs:
                model.load_state_dict(base)
                apply_pruning(model.masks(), scores, fr)
                recalibrate_bn(model, batches)
                for m, v in eval_losses(a.task, model, dec).items():
                    pd.setdefault(m, []).append(v)
                for m, v in eval_losses(a.task, model, truth).items():
                    pt.setdefault(m, []).append(v)
            rec.update({f"prune_dec_{m}": np.stack(v) for m, v in pd.items()})
            rec.update({f"prune_truth_{m}": np.stack(v) for m, v in pt.items()})
            rec["fracs"] = np.asarray(a.fracs)
        rec["train_seconds"] = time.time() - t0
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **rec)
        os.replace(tmp, path)
        main_key = "dec_se" if a.task == "mlp" else "dec_err"
        print(f"{a.task} w={w:<5} d={d} seed={s:<3} params={rec['n_params']:>9,} "
              f"dec loss={rec[main_key].mean():.4f}  {rec['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
