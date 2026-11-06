"""Trained networks for the deep experiments: MLPs and CIFAR ResNets.

Both models carry pruning masks. A mask multiplies the output of one hidden
unit (MLP) or one inner channel of a residual block (ResNet), so pruning never
changes tensor shapes and the residual connections stay valid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- MLP
class MLP(nn.Module):
    """Fully connected ReLU network: `depth` hidden layers of `width` units."""

    def __init__(self, d_in, width, depth):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(d_in if l == 0 else width, width) for l in range(depth)])
        self.head = nn.Linear(width, 1)
        for l in range(depth):
            self.register_buffer(f"mask{l}", torch.ones(width))

    def masks(self):
        return [getattr(self, f"mask{l}") for l in range(len(self.layers))]

    def forward(self, x, return_hidden=False):
        h, hidden = x, []
        for lin, m in zip(self.layers, self.masks()):
            h = F.relu(lin(h)) * m
            hidden.append(h)
        out = self.head(h).squeeze(-1)
        return (out, hidden) if return_hidden else out

    @torch.no_grad()
    def unit_scores(self, x_train):
        """Importance of each hidden unit, from training data only:
        sd of its activation times the norm of its outgoing weights."""
        _, hidden = self.forward(x_train, return_hidden=True)
        scores = []
        for l, h in enumerate(hidden):
            nxt = self.layers[l + 1].weight if l + 1 < len(self.layers) else self.head.weight
            scores.append(h.std(0) * nxt.norm(dim=0))
        return scores


# ------------------------------------------------------------------ ResNet
class Block(nn.Module):
    """Basic residual block (He et al., 2016) with a mask on its inner channels."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))
        self.register_buffer("mask", torch.ones(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x))) * self.mask.view(1, -1, 1, 1)
        o = self.b2(self.c2(o))
        return F.relu(o + self.short(x))


class ResNetCIFAR(nn.Module):
    """CIFAR ResNet with base width k and n blocks per stage: depth 6n + 2
    (n = 1, 2, 3, 5 give ResNet-8, -14, -20, -32). Stage widths k, 2k, 4k."""

    def __init__(self, k, n, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, k, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(k)
        blocks, cin = [], k
        for s, mult in enumerate([1, 2, 4]):
            for b in range(n):
                stride = 2 if (s > 0 and b == 0) else 1
                blocks.append(Block(cin, k * mult, stride))
                cin = k * mult
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def masks(self):
        return [blk.mask for blk in self.blocks]

    @torch.no_grad()
    def unit_scores(self, x_train=None):
        """Channel importance |gamma| of the first BatchNorm of each block
        (network-slimming criterion); depends on the trained weights only."""
        return [blk.b1.weight.abs().detach().clone() for blk in self.blocks]


# ----------------------------------------------------------------- pruning
@torch.no_grad()
def apply_pruning(masks, scores, frac):
    """Layer-wise pruning: in every layer (or block) remove the floor(frac * size)
    units with the smallest score, keeping at least one. The ranking is fixed, so
    the networks obtained for increasing frac are nested."""
    for m, s in zip(masks, scores):
        m.fill_(1.0)
        k = min(int(frac * m.numel()), m.numel() - 1)
        if k > 0:
            m[torch.argsort(s)[:k]] = 0.0


@torch.no_grad()
def recalibrate_bn(model, batches):
    """Re-estimate BatchNorm running statistics after pruning (no gradient steps)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative average over the given batches
    model.train()
    for xb in batches:
        model(xb)
    for m in bns:
        m.momentum = 0.1
    model.eval()


def n_params(model):
    return sum(p.numel() for p in model.parameters())
