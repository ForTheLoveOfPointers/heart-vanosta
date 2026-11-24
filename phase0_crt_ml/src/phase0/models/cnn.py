"""A deliberately tiny 1-D CNN with a pairwise ranking head.

The point is not to win a benchmark against large models; it is to test whether
a raw-signal model can learn the patient-specific AV/VV ranking at all, and
whether it is more robust than the hand-crafted linear model under a shifted
sensor domain.

Architecture: three conv blocks (7 -> width -> 2*width -> 4*width) and a linear
score head.  About 2.7k trainable parameters at the default width.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..evaluate import metrics_from_scores
from .pairwise import build_pairs


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample, per-channel z-score over time."""
    mu = x.mean(dim=2, keepdim=True)
    sd = x.std(dim=2, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


class TinyCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        context_dim: int = 3,
        width: int = 8,
        input_length: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(width * 2, width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width * 4 * 2 + context_dim, 1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = self.features(_normalize(x))
        z = torch.cat([z.mean(dim=2), z.max(dim=2).values, context], dim=1)
        return self.head(z).squeeze(-1)


def _gaussian_kernel(sigma: float, size: int = 9) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


_SMOOTH_KERNELS = [_gaussian_kernel(s) for s in (0.6, 1.0, 1.6, 2.4, 3.2, 4.5)]


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Cheap waveform augmentation for domain robustness."""
    c, length = x.shape
    if random.random() < 0.7:
        k = random.choice(_SMOOTH_KERNELS).view(1, 1, -1).repeat(c, 1, 1)
        x = F.conv1d(x.unsqueeze(0), k, padding=k.shape[-1] // 2, groups=c).squeeze(0)
    if random.random() < 0.6:
        x = torch.roll(x, random.randint(-12, 12), dims=1)
    if random.random() < 0.6:
        x = x * torch.empty(c, 1).uniform_(0.7, 1.3)
    scale = x.std().clamp_min(1e-6)
    if random.random() < 0.6:
        x = x + torch.randn_like(x) * random.uniform(0.03, 0.10) * scale
    if random.random() < 0.5:
        t = torch.linspace(0.0, 1.0, length)
        freq = random.uniform(0.5, 2.0)
        phase = random.uniform(0.0, 6.283)
        amp = random.uniform(0.05, 0.30) * scale
        x = x + amp * torch.sin(2 * 3.14159265 * freq * t + phase).unsqueeze(0)
    if random.random() < 0.10:
        x[random.randrange(c)] = 0.0
    return x


def make_context(av: np.ndarray, vv: np.ndarray, hr: np.ndarray) -> np.ndarray:
    """Known programming/context channels for the CNN."""
    return np.stack(
        [
            np.asarray(av, dtype=np.float32) / 200.0,
            np.asarray(vv, dtype=np.float32) / 100.0,
            np.asarray(hr, dtype=np.float32) / 100.0,
        ],
        axis=1,
    ).astype(np.float32)


class PairDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        context: np.ndarray,
        pairs: np.ndarray,
        augment: bool = True,
    ):
        self.signals = signals
        self.context = context
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, k: int):
        i, j = self.pairs[k]
        xi = torch.from_numpy(self.signals[i].astype(np.float32))
        xj = torch.from_numpy(self.signals[j].astype(np.float32))
        if self.augment:
            xi = _augment(xi)
            xj = _augment(xj)
        ci = torch.from_numpy(self.context[i])
        cj = torch.from_numpy(self.context[j])
        return xi, xj, ci, cj


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class TinyCNNEmbedder(nn.Module):
    """Wraps a TinyCNN and returns its penultimate embedding.

    The embedding is the pooled conv features concatenated with the context
    channels, i.e. exactly the input to the linear score head.  It is what a
    downstream tabular model (e.g. LambdaMART) consumes in the hybrid
    CNN-encoder + GBDT design.
    """

    def __init__(self, model: TinyCNN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = self.model.features(_normalize(x))
        return torch.cat([z.mean(dim=2), z.max(dim=2).values, context], dim=1)


@torch.no_grad()
def embed_cnn(
    model: nn.Module,
    signals: np.ndarray,
    context: np.ndarray,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Penultimate embeddings of TinyCNN for the selected rows."""
    if indices is None:
        indices = np.arange(len(signals))
    indices = np.asarray(indices, dtype=int)
    device = device or torch.device("cpu")
    embedder = TinyCNNEmbedder(model).to(device)
    embedder.eval()
    dim = None
    out: Optional[np.ndarray] = None
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        x = torch.from_numpy(signals[idx].astype(np.float32)).to(device)
        c = torch.from_numpy(context[idx]).to(device)
        z = embedder(x, c).cpu().numpy().astype(np.float32)
        if out is None:
            dim = z.shape[1]
            out = np.empty((len(indices), dim), dtype=np.float32)
        out[start : start + len(idx)] = z
    assert out is not None
    return out


@torch.no_grad()
def score_cnn(
    model: nn.Module,
    signals: np.ndarray,
    context: np.ndarray,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if indices is None:
        indices = np.arange(len(signals))
    indices = np.asarray(indices, dtype=int)
    device = device or torch.device("cpu")
    model.eval()
    out = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        x = torch.from_numpy(signals[idx].astype(np.float32)).to(device)
        c = torch.from_numpy(context[idx]).to(device)
        out[start : start + len(idx)] = model(x, c).cpu().numpy()
    return out


def train_cnn(
    signals: np.ndarray,
    context: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    av: np.ndarray,
    vv: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    cfg: dict,
    seed: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train the tiny CNN with a Bradley-Terry pairwise loss."""
    bench = cfg["benchmark"]
    epochs = int(bench["cnn_epochs"])
    batch_size = int(bench["cnn_batch_pairs"])
    lr = float(bench["cnn_lr"])
    weight_decay = float(bench["cnn_weight_decay"])
    width = int(bench["cnn_width"])
    patience = int(bench["cnn_patience"])
    max_pairs = int(bench["cnn_max_pairs_per_group"])
    margin_frac = float(cfg["model"]["margin_frac"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx = np.where(train_mask)[0]
    pairs_local = build_pairs(
        y[train_mask],
        groups[train_mask],
        margin_frac=margin_frac,
        max_pairs_per_group=max_pairs,
        seed=seed,
    )
    if len(pairs_local[0]) < 2:
        raise RuntimeError("Need at least two training pairs.")
    pairs = np.stack([train_idx[pairs_local[0]], train_idx[pairs_local[1]]], axis=1)
    model = TinyCNN(
        in_channels=signals.shape[1],
        context_dim=context.shape[1],
        width=width,
        input_length=signals.shape[2],
    ).to(device)
    if verbose:
        print(
            f"  CNN: {len(pairs)} pairs, {count_parameters(model)} params, "
            f"device={device}"
        )

    # A final batch of size 1 breaks BatchNorm1d in training.  Drop it only
    # when it is the single leftover sample, not when the whole set is small.
    drop_last = len(pairs) > batch_size and len(pairs) % batch_size == 1
    loader = DataLoader(
        PairDataset(signals, context, pairs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    val_idx = np.where(val_mask)[0]
    best_state = copy.deepcopy(model.state_dict())
    best_regret = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xi, xj, ci, cj in loader:
            xi = xi.to(device)
            xj = xj.to(device)
            ci = ci.to(device)
            cj = cj.to(device)
            si = model(xi, ci)
            sj = model(xj, cj)
            loss = F.softplus(-(si - sj)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach()) * len(xi)
            n_seen += len(xi)
        scheduler.step()

        val_scores = score_cnn(model, signals, context, val_idx, device=device)
        val_metrics = metrics_from_scores(
            y[val_mask],
            groups[val_mask],
            val_scores,
            av[val_mask],
            vv[val_mask],
            top_k=int(cfg["evaluation"]["top_k"]),
            regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
        )
        regret = val_metrics["mean_regret"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(n_seen, 1),
                "val_regret": regret,
                "val_near_optimal": val_metrics["near_optimal_fraction"],
                "val_spearman": val_metrics["mean_spearman"],
                "seconds": time.time() - t0,
            }
        )
        if verbose:
            print(
                f"  epoch {epoch:02d}: loss={total_loss / max(n_seen, 1):.4f} "
                f"val_regret={regret:.4f} val_near={val_metrics['near_optimal_fraction']:.3f}"
            )
        if np.isfinite(regret) and regret < best_regret:
            best_regret = regret
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best {best_epoch})")
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_regret": best_regret,
        "n_pairs": int(len(pairs)),
        "n_params": count_parameters(model),
        "train_time_s": time.time() - t0,
        "device": str(device),
        "seed": int(seed),
    }
