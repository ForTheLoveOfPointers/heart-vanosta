#!/usr/bin/env python3
"""Shuffled-label negative control for the benchmark.

If the model can still find near-optimal settings after breaking the
within-patient label association, there is leakage or an evaluation bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from phase0.benchmark import BenchmarkDataset
from phase0.evaluate import metrics_from_scores, three_way_split
from phase0.features import patient_normalize
from phase0.models.pairwise import PairwiseRanker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="bench500_imp")
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    ds = BenchmarkDataset.load(out / f"benchmark_dataset_{args.tag}.npz")
    linear_path = out / f"linear_{args.tag}.npz"
    pp = ()
    if linear_path.exists():
        lin = np.load(linear_path, allow_pickle=True)
        if "patient_normalize" in lin:
            pp = tuple(str(x) for x in lin["patient_normalize"])
    elif cfg.get("features", {}).get("patient_normalize"):
        pp = tuple(str(x) for x in cfg["features"]["patient_normalize"])
    if pp:
        ds.X = patient_normalize(ds.X, ds.groups, ds.feature_names, pp)

    split_path = out / f"benchmark_split_{args.tag}.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_mask = np.isin(ds.groups, [int(x) for x in split["train_ids"]])
        test_mask = np.isin(ds.groups, [int(x) for x in split["test_ids"]])
    else:
        train_mask, _, test_mask = three_way_split(
            ds.groups, val_frac=0.15, test_frac=0.15, seed=int(cfg.get("seed", 0))
        )

    rng = np.random.default_rng(args.seed)
    y_shuffled = ds.y.copy()
    for g in np.unique(ds.groups):
        m = ds.groups == g
        y_shuffled[m] = rng.permutation(y_shuffled[m])

    model_kwargs = {
        "l2": float(cfg["model"]["l2"]),
        "margin_frac": float(cfg["model"]["margin_frac"]),
        "max_pairs_per_group": int(cfg["model"]["max_pairs_per_group"]),
        "seed": args.seed,
    }
    ranker = PairwiseRanker(**model_kwargs).fit(
        ds.X[train_mask], y_shuffled[train_mask], ds.groups[train_mask]
    )
    metrics = metrics_from_scores(
        y_shuffled[test_mask],
        ds.groups[test_mask],
        ranker.score(ds.X[test_mask]),
        ds.av[test_mask],
        ds.vv[test_mask],
        top_k=int(cfg["evaluation"]["top_k"]),
        regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
    )
    result = {
        "tag": args.tag,
        "seed": args.seed,
        "patient_normalize": list(pp),
        "metrics": metrics,
    }
    path = out / f"benchmark_negative_control_{args.tag}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
