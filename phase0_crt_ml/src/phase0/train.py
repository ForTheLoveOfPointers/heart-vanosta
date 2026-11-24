"""End-to-end Phase 0 experiment: train, evaluate, ablate, learning curve."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .dataset import Dataset
from .evaluate import (
    evaluate_ranker,
    learning_curve,
    run_ablations,
    split_groups,
)
from .models.pairwise import PairwiseRanker


def run_experiment(ds: Dataset, cfg: dict) -> Dict[str, object]:
    ev = cfg["evaluation"]
    mdl = cfg["model"]

    train_mask, test_mask = split_groups(
        ds.groups, test_frac=ev["test_frac"], seed=int(cfg.get("seed", 0))
    )

    model_kwargs = {
        "l2": float(mdl["l2"]),
        "margin_frac": float(mdl["margin_frac"]),
        "max_pairs_per_group": int(mdl["max_pairs_per_group"]),
        "seed": int(cfg.get("seed", 0)),
    }

    ranker = PairwiseRanker(**model_kwargs).fit(
        ds.X[train_mask], ds.y[train_mask], ds.groups[train_mask]
    )
    main_metrics = evaluate_ranker(
        ranker,
        ds,
        eval_mask=test_mask,
        top_k=int(ev["top_k"]),
        regret_threshold=float(ev["regret_threshold"]),
    )

    ablations = run_ablations(
        ds,
        train_mask,
        test_mask,
        model_kwargs,
        top_k=int(ev["top_k"]),
        regret_threshold=float(ev["regret_threshold"]),
    )

    curve = learning_curve(
        ds,
        train_mask,
        test_mask,
        sizes=[int(x) for x in ev["learning_curve_patients"]],
        model_kwargs=model_kwargs,
        seed=int(cfg.get("seed", 0)),
        top_k=int(ev["top_k"]),
        regret_threshold=float(ev["regret_threshold"]),
    )

    return {
        "main": main_metrics,
        "ablations": ablations,
        "learning_curve": curve,
        "ranker": ranker,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "n_features": len(ds.feature_names),
        "n_samples": int(len(ds.y)),
        "n_patients": int(len(np.unique(ds.groups))),
    }
