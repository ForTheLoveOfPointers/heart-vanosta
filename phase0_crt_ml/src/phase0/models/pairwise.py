"""Pairwise (Bradley-Terry) ranking model.

Why ranking?  We do not need to predict absolute stroke work; we need to know
which AV/VV setting is best *for this patient*.  Pairwise comparisons within a
patient are calibration-invariant and handle flat response curves gracefully.

The model is linear in the (standardised) features, which is enough for the
Phase 0 feasibility study and runs with numpy/scipy only.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def build_pairs(
    y: np.ndarray,
    groups: np.ndarray,
    margin_frac: float = 0.01,
    max_pairs_per_group: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return index pairs ``(i, j)`` where ``y[i]`` is better than ``y[j]``.

    Pairs are built *within* each group so that comparisons are patient-specific.
    """
    rng = np.random.default_rng(seed)
    ii: list[int] = []
    jj: list[int] = []

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        yg = y[idx]
        span = float(yg.max() - yg.min())
        margin = margin_frac * span
        pairs = [
            (i, j)
            for i in range(len(idx))
            for j in range(len(idx))
            if yg[i] > yg[j] + margin
        ]
        if len(pairs) > max_pairs_per_group:
            chosen = rng.choice(len(pairs), size=max_pairs_per_group, replace=False)
            pairs = [pairs[k] for k in chosen]
        for i, j in pairs:
            ii.append(int(idx[i]))
            jj.append(int(idx[j]))

    return np.array(ii, dtype=int), np.array(jj, dtype=int)


class PairwiseRanker:
    def __init__(
        self,
        l2: float = 1e-2,
        margin_frac: float = 0.01,
        max_pairs_per_group: int = 500,
        seed: int = 0,
    ) -> None:
        self.l2 = float(l2)
        self.margin_frac = float(margin_frac)
        self.max_pairs_per_group = int(max_pairs_per_group)
        self.seed = int(seed)
        self.w: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def _build_pairs(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        del X  # kept for API compatibility
        return build_pairs(
            y,
            groups,
            margin_frac=self.margin_frac,
            max_pairs_per_group=self.max_pairs_per_group,
            seed=self.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "PairwiseRanker":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X) == 0:
            raise ValueError("Cannot fit PairwiseRanker on an empty dataset.")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values.")
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        Xs = (X - self.mu) / self.sd

        ii, jj = self._build_pairs(Xs, y, groups)
        if len(ii) == 0:
            self.w = np.zeros(X.shape[1], dtype=float)
            return self

        D = Xs[ii] - Xs[jj]

        def objective(w: np.ndarray):
            s = D @ w
            loss = float(np.mean(np.logaddexp(0.0, -s)) + self.l2 * np.dot(w, w))
            coef = -1.0 / (1.0 + np.exp(s))
            grad = (coef[:, None] * D).mean(axis=0) + 2.0 * self.l2 * w
            return loss, grad

        res = minimize(
            objective,
            np.zeros(X.shape[1], dtype=float),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        self.w = res.x
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.mu is None or self.sd is None:
            raise RuntimeError("Model is not fitted.")
        Xs = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return Xs @ self.w
