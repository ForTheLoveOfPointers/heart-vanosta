"""LambdaMART ranking model (LightGBM ``lambdarank``).

This is the production-grade learning-to-rank counterpart of the linear
Bradley-Terry ranker in ``pairwise.py``: the same within-patient pairwise
comparisons, but the score is an additive tree ensemble over the features
instead of a dot product.  Labels are converted to within-patient rank grades
so the loss only sees patient-specific contrasts and stays invariant to the
absolute calibration of stroke work.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np


@contextlib.contextmanager
def _catch_lgb_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def group_rank_grades(
    y: np.ndarray, groups: np.ndarray, n_grades: int = 8
) -> np.ndarray:
    """Per-group rank grades 0..n_grades-1 (higher stroke work -> higher grade).

    Rank-based grades are robust to the label scale and to flat response
    regions: two settings in a near-tie get adjacent grades rather than
    arbitrary value splits.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    grades = np.zeros(len(y), dtype=int)
    for g in np.unique(groups):
        m = groups == g
        yg = y[m]
        order = yg.argsort().argsort()  # 0 = worst, n-1 = best
        grades[m] = np.floor(order * n_grades / max(len(yg), 1)).astype(int)
        grades[m] = np.clip(grades[m], 0, n_grades - 1)
    return grades


def _contiguous_groups(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort rows by group and return (order, group sizes) for LightGBM."""
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    sizes = np.diff(np.concatenate([[0], boundaries, [len(sorted_groups)]]))
    return order, sizes.astype(int)


class LambdaMARTRanker:
    def __init__(
        self,
        n_grades: int = 8,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 100,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.9,
        bagging_freq: int = 1,
        label_gain: str = "linear",
        seed: int = 0,
        verbose: int = -1,
    ) -> None:
        self.n_grades = int(n_grades)
        # Rank grades are ordinal, not exponential relevance: linear gains
        # weight each grade step equally, which matched the clinical regret
        # metric better than LightGBM's default 2^g - 1 gains.
        if label_gain == "linear":
            gains = list(range(max(self.n_grades, 2)))
        elif label_gain == "exponential":
            gains = None  # LightGBM default
        else:
            raise ValueError(f"Unknown label_gain: {label_gain!r}")
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [1, 3],
            "num_leaves": int(num_leaves),
            "learning_rate": float(learning_rate),
            "min_data_in_leaf": int(min_data_in_leaf),
            "feature_fraction": float(feature_fraction),
            "bagging_fraction": float(bagging_fraction),
            "bagging_freq": int(bagging_freq),
            "seed": int(seed),
            "verbose": int(verbose),
            "num_threads": 4,
        }
        if gains is not None:
            self.params["label_gain"] = gains
        self.n_estimators = int(n_estimators)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.model = None
        self.best_iteration: int | None = None

    def _prepare_masked(self, X, y, groups, mask):
        """LightGBM Dataset for the masked rows, rows sorted by group."""
        import lightgbm as lgb

        X = np.asarray(X, dtype=np.float32)[mask]
        y = np.asarray(y, dtype=float)[mask]
        g = np.asarray(groups)[mask]
        order, sizes = _contiguous_groups(g)
        grades = group_rank_grades(y[order], g[order], self.n_grades)
        return lgb.Dataset(X[order], label=grades, group=sizes), (order,)

    def _prepare(self, X, y, groups):
        X = np.asarray(X, dtype=np.float32)
        order, group_sizes = _contiguous_groups(groups)
        grades = group_rank_grades(
            np.asarray(y, dtype=float)[order],
            np.asarray(groups)[order],
            self.n_grades,
        )
        return X[order], grades, group_sizes, order

    def _regret_feval(self, y_val, groups_val):
        """Early-stop on the deployment metric: mean within-patient regret."""

        def feval(preds, _dataset):
            from ..evaluate import metrics_from_scores

            preds = np.asarray(preds, dtype=float)
            met = metrics_from_scores(
                y_val,
                groups_val,
                preds,
                np.zeros(len(y_val)),
                np.zeros(len(y_val)),
                top_k=3,
                regret_threshold=0.02,
            )
            # LightGBM metrics are "higher is better"; negate the regret.
            return "neg_regret", float(-met["mean_regret"]), True

        return feval

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
    ) -> "LambdaMARTRanker":
        import lightgbm as lgb

        if len(X) == 0:
            raise ValueError("Cannot fit LambdaMARTRanker on an empty dataset.")
        if not np.isfinite(np.asarray(X, dtype=float)).all():
            raise ValueError("X contains non-finite values.")

        Xs, grades, sizes, _ = self._prepare(X, y, groups)
        train_set = lgb.Dataset(Xs, label=grades, group=sizes)

        has_val = X_val is not None and y_val is not None and groups_val is not None
        valid_sets = None
        valid_names = None
        feval = None
        callbacks = []
        if has_val:
            # Early stopping on validation NDCG is unreliable here: NDCG@1
            # saturates long before the clinical metric (within-patient
            # regret) stops improving, so we early-stop on the deployment
            # metric instead, via a custom evaluation function.
            Xv, grades_v, sizes_v, order_v = self._prepare(X_val, y_val, groups_val)
            val_set = lgb.Dataset(
                Xv, label=grades_v, group=sizes_v, reference=train_set
            )
            valid_sets = [val_set]
            valid_names = ["valid"]
            feval = self._regret_feval(
                np.asarray(y_val, dtype=float)[order_v],
                np.asarray(groups_val)[order_v],
            )
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            )

        params = dict(self.params)
        if has_val:
            params["metric"] = "None"

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval,
            callbacks=callbacks,
        )
        self.best_iteration = (
            int(self.model.best_iteration) if has_val else None
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float64)
        n = self.best_iteration or self.model.num_trees()
        batch = 8192
        for start in range(0, len(X), batch):
            out[start : start + batch] = self.model.predict(
                X[start : start + batch], num_iteration=n
            )
        return out

    def feature_importance(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        n = self.best_iteration or self.model.num_trees()
        return self.model.feature_importance(
            importance_type="gain", iteration=n
        ).astype(float)


def fit_lambda_mart_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    num_boost_round: int = 600,
    seed: int = 0,
    **ranker_kwargs,
) -> tuple[LambdaMARTRanker, dict]:
    """Train LambdaMART with grouped-CV selection of the round count.

    A single small validation split makes the per-round validation regret too
    noisy for reliable early stopping, so we average the per-round validation
    regret over k patient-level folds of the training data, pick the best
    round, and retrain on the full training set with that many rounds.  This
    is the standard production recipe for small-data GBDT round selection.

    Returns (fitted ranker, cv_diagnostics).
    """
    import lightgbm as lgb

    ranker = LambdaMARTRanker(
        n_estimators=num_boost_round,
        early_stopping_rounds=10**9,  # no per-fold early stopping
        seed=seed,
        **ranker_kwargs,
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    ids = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds = np.array_split(ids, int(n_folds))

    curves = []
    for f in folds:
        held = np.isin(groups, f)
        dtr, (order_h,) = ranker._prepare_masked(X, y, groups, ~held)
        dva, (order_v,) = ranker._prepare_masked(X, y, groups, held)
        dva = lgb.Dataset(dva.data, label=dva.label, group=dva.group, reference=dtr)
        evals: dict = {}
        with _catch_lgb_warnings():
            lgb.train(
                ranker.params,
                dtr,
                num_boost_round=num_boost_round,
                valid_sets=[dva],
                valid_names=["v"],
                feval=ranker._regret_feval(y[held][order_v], groups[held][order_v]),
                callbacks=[lgb.record_evaluation(evals)],
            )
        curves.append(np.asarray(evals["v"]["neg_regret"]))

    mean_curve = np.vstack(curves).mean(axis=0)
    best_round = int(mean_curve.argmax()) + 1

    with _catch_lgb_warnings():
        full, _ = ranker._prepare_masked(X, y, groups, np.ones(len(y), dtype=bool))
        ranker.model = lgb.train(ranker.params, full, num_boost_round=best_round)
    ranker.best_iteration = best_round
    diagnostics = {
        "cv_folds": int(n_folds),
        "cv_best_round": best_round,
        "cv_mean_val_regret_at_best": float(-mean_curve.max()),
        "cv_curve": mean_curve.tolist(),
    }
    return ranker, diagnostics
