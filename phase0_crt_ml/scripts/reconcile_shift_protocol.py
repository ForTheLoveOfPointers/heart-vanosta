#!/usr/bin/env python3
"""Recompute the sealed shifted-domain protocol at 2% and 5% tolerances.

Reports:
- source-trained linear, evaluated on source and shifted test;
- source-trained linear with within-patient impedance normalization;
- target-adapted 5-fold grouped CV within the shifted domain (oracle bound);
- target-adapted + within-patient normalization.

This is a reconciliation script for the preprint. It uses the same dataset,
split, model hyperparameters, and seeds as the primary ladder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase0.benchmark import BenchmarkDataset
from phase0.evaluate import metrics_from_scores
from phase0.features import patient_normalize
from phase0.models.pairwise import PairwiseRanker


def _load_split(path: Path, groups: np.ndarray):
    split = json.loads(path.read_text(encoding="utf-8"))
    ids = {name: set(int(x) for x in split[f"{name}_ids"]) for name in ("train", "val", "test")}
    masks = {name: np.isin(groups, list(v)) for name, v in ids.items()}
    if set(np.unique(groups).tolist()) != ids["train"] | ids["val"] | ids["test"]:
        raise RuntimeError("Split file does not cover the dataset patient ids.")
    return masks


def _metrics(y, groups, scores, av, vv, threshold):
    return metrics_from_scores(
        y, groups, scores, av, vv, top_k=3, regret_threshold=threshold
    )


def _fit(train_X, train_y, train_groups, cfg, seed=0):
    return PairwiseRanker(
        l2=float(cfg["model"]["l2"]),
        margin_frac=float(cfg["model"]["margin_frac"]),
        max_pairs_per_group=int(cfg["model"]["max_pairs_per_group"]),
        seed=seed,
    ).fit(train_X, train_y, train_groups)


def _grouped_folds(groups: np.ndarray, n_folds: int, seed: int):
    unique = np.unique(groups).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return np.array_split(unique, n_folds)


def _target_adapted(X, y, groups, av, vv, thresholds, cfg, seed=0):
    oof = np.full(len(y), np.nan, dtype=float)
    for fold in _grouped_folds(groups, 5, seed):
        val = np.isin(groups, fold)
        train = ~val
        model = _fit(X[train], y[train], groups[train], cfg, seed=seed)
        oof[val] = model.score(X[val])
    if not np.isfinite(oof).all():
        raise RuntimeError("Target-adapted CV did not produce scores for every row.")
    return {f"{th:g}": _metrics(y, groups, oof, av, vv, th) for th in thresholds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument("--dataset-tag", default="bench500_imp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    ds = BenchmarkDataset.load(out / f"benchmark_dataset_{args.dataset_tag}.npz")
    shifted = BenchmarkDataset.load(out / f"benchmark_shifted_{args.dataset_tag}.npz")
    masks = _load_split(out / f"benchmark_split_{args.dataset_tag}.json", ds.groups)
    train_mask, test_mask = masks["train"], masks["test"]
    if len(shifted.y) != int(test_mask.sum()):
        raise RuntimeError("Shifted rows do not match the test split.")

    results = {}
    for th in args.thresholds:
        model = _fit(ds.X[train_mask], ds.y[train_mask], ds.groups[train_mask], cfg, args.seed)
        results[f"source_trained_source_{th:g}"] = _metrics(
            ds.y[test_mask], ds.groups[test_mask], model.score(ds.X[test_mask]),
            ds.av[test_mask], ds.vv[test_mask], th,
        )
        results[f"source_trained_shifted_{th:g}"] = _metrics(
            shifted.y, shifted.groups, model.score(shifted.X),
            shifted.av, shifted.vv, th,
        )

    def norm(X, groups, names):
        return patient_normalize(X, groups, names, ["imp_"])

    X_all = norm(ds.X, ds.groups, ds.feature_names)
    X_test = norm(ds.X[test_mask], ds.groups[test_mask], ds.feature_names)
    X_shift = norm(shifted.X, shifted.groups, shifted.feature_names)
    model_n = _fit(X_all[train_mask], ds.y[train_mask], ds.groups[train_mask], cfg, args.seed)
    for th in args.thresholds:
        results[f"source_trained_norm_source_{th:g}"] = _metrics(
            ds.y[test_mask], ds.groups[test_mask], model_n.score(X_test),
            ds.av[test_mask], ds.vv[test_mask], th,
        )
        results[f"source_trained_norm_shifted_{th:g}"] = _metrics(
            shifted.y, shifted.groups, model_n.score(X_shift),
            shifted.av, shifted.vv, th,
        )

    for th in args.thresholds:
        adapted = _target_adapted(
            shifted.X, shifted.y, shifted.groups, shifted.av, shifted.vv,
            [th], cfg, args.seed,
        )[f"{th:g}"]
        results[f"target_adapted_shifted_{th:g}"] = adapted
        adapted_n = _target_adapted(
            X_shift, shifted.y, shifted.groups, shifted.av, shifted.vv,
            [th], cfg, args.seed,
        )[f"{th:g}"]
        results[f"target_adapted_norm_shifted_{th:g}"] = adapted_n

    report = {
        "dataset_tag": args.dataset_tag,
        "seed": args.seed,
        "thresholds": args.thresholds,
        "results": results,
    }
    path = out / "reconcile_shift_protocol.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
