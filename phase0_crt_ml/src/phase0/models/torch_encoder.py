"""Optional PyTorch encoder + ranking head.

This file is imported only when torch is available.  It is the intended path
once raw multi-channel waveforms are used instead of hand-crafted features.

Usage sketch:
    from phase0.models.torch_encoder import HAS_TORCH, TorchRanker
    model = TorchRanker(n_channels=7, n_settings=30)
    # train with a pairwise logistic loss over (beat_encoder, setting_embedding)
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TORCH = False


if HAS_TORCH:

    class SignalEncoder(nn.Module):
        def __init__(self, in_channels: int = 7, hidden: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=15, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, hidden, kernel_size=9, padding=4),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TorchRanker(nn.Module):
        """Scores (signal window, candidate AV/VV) pairs."""

        def __init__(self, in_channels: int = 7, hidden: int = 64) -> None:
            super().__init__()
            self.encoder = SignalEncoder(in_channels, hidden)
            self.setting_mlp = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden + 32, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, signals: torch.Tensor, settings: torch.Tensor) -> torch.Tensor:
            z = self.encoder(signals)
            s = self.setting_mlp(settings)
            return self.head(torch.cat([z, s], dim=-1)).squeeze(-1)

    def pairwise_loss(scores_i, scores_j):
        return F.softplus(-(scores_i - scores_j)).mean()

else:

    class TorchRanker:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is not installed. `pip install torch` to use TorchRanker.")
