"""Evaluation metrics and feature masks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from .models.pairwise import PairwiseRanker


def sensor_feature_masks(feature_names: List[str]) -> Dict[str, np.ndarray]:
    names = np.array(feature_names)

    def pick(prefixes):
        m = np.zeros(len(names), dtype=bool)
        for p in prefixes:
            m |= np.char.startswith(names.astype(str), p)
        return m

    def pick_interactions(substrings):
        m = np.zeros(len(names), dtype=bool)
        for i, n in enumerate(names):
            s = str(n)
            if s.startswith("ix_") and any(sub in s for sub in substrings):
                m[i] = True
        return m

    context = pick(["set_", "ctx_"])
    impedance = context | pick(["imp_"]) | pick_interactions(["ctx_", "imp_"])
    return {
        "all": np.ones(len(names), dtype=bool),
        "impedance_only": impedance,
        "no_sensors": context | pick_interactions(["ctx_"]),
        "setting_only": pick(["set_"]),
    }


def metrics_from_scores(
    y: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    av: np.ndarray,
    vv: np.ndarray,
    eval_mask: np.ndarray | None = None,
    top_k: int = 3,
    regret_threshold: float = 0.02,
) -> Dict[str, float]:
    """Patient-level ranking metrics for an arbitrary score vector."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    scores = np.asarray(scores)
    av = np.asarray(av)
    vv = np.asarray(vv)
    lengths = {len(y), len(groups), len(scores), len(av), len(vv)}
    if len(lengths) != 1:
        raise ValueError(
            "y, groups, scores, av and vv must have the same length; "
            f"got {sorted(lengths)}"
        )
    if eval_mask is None:
        eval_mask = np.ones(len(y), dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask, dtype=bool)
        if len(eval_mask) != len(y):
            raise ValueError(
                "eval_mask must have the same length as y; "
                f"got {len(eval_mask)} and {len(y)}"
            )

    exact, near_opt, topk_hit, regrets, rhos = [], [], [], [], []
    av_err, vv_err = [], []

    for g in np.unique(groups[eval_mask]):
        m = (groups == g) & eval_mask
        if m.sum() < 2:
            continue
        yg = y[m]
        sg = scores[m]
        avg = av[m]
        vvg = vv[m]

        i_best = int(np.argmax(yg))
        i_pred = int(np.argmax(sg))
        best_sw = abs(float(yg[i_best])) + 1e-9
        regret = (float(yg[i_best]) - float(yg[i_pred])) / best_sw

        regrets.append(regret)
        exact.append(float(i_best == i_pred))
        near_opt.append(float(regret <= regret_threshold))

        order = np.argsort(sg)[::-1][:top_k]
        topk_hit.append(float(i_best in order.tolist()))

        rho = spearmanr(sg, yg).correlation
        if np.isfinite(rho):
            rhos.append(float(rho))

        av_err.append(abs(float(avg[i_best]) - float(avg[i_pred])))
        vv_err.append(abs(float(vvg[i_best]) - float(vvg[i_pred])))

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    return {
        "exact_argmax_accuracy": mean(exact),
        "near_optimal_fraction": mean(near_opt),
        f"top{top_k}_hit": mean(topk_hit),
        "mean_regret": mean(regrets),
        "median_regret": float(np.median(regrets)) if regrets else float("nan"),
        "mean_spearman": mean(rhos),
        "mean_abs_av_error_ms": mean(av_err),
        "mean_abs_vv_error_ms": mean(vv_err),
        "n_patients": float(len(regrets)),
    }


def evaluate_ranker(
    ranker: PairwiseRanker,
    ds,
    feature_mask: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    top_k: int = 3,
    regret_threshold: float = 0.02,
) -> Dict[str, float]:
    X = ds.X if feature_mask is None else ds.X[:, feature_mask]
    scores = ranker.score(X)
    return metrics_from_scores(
        ds.y,
        ds.groups,
        scores,
        ds.av,
        ds.vv,
        eval_mask=eval_mask,
        top_k=top_k,
        regret_threshold=regret_threshold,
    )


def three_way_split(
    groups: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
):
    """Patient-level train/validation/test split."""
    ids = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_test = max(1, int(round(test_frac * len(ids))))
    n_val = max(1, int(round(val_frac * len(ids))))
    if n_test + n_val >= len(ids):
        raise ValueError("Not enough groups for the requested split.")

    test_ids = set(ids[:n_test].tolist())
    val_ids = set(ids[n_test : n_test + n_val].tolist())
    train_ids = set(ids[n_test + n_val :].tolist())

    train_mask = np.isin(groups, list(train_ids))
    val_mask = np.isin(groups, list(val_ids))
    test_mask = np.isin(groups, list(test_ids))
    return train_mask, val_mask, test_mask
