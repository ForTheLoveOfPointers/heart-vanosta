#!/usr/bin/env python3
"""Tiny end-to-end CircAdapt check. Run this first after setup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from phase0.benchmark import build_benchmark_dataset, sample_patients
from phase0.evaluate import metrics_from_scores
from phase0.models.pairwise import PairwiseRanker
from phase0.physiology import CircAdaptSimulator
from phase0.sensors import SOURCE_DOMAIN


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    n_patients = 3
    cfg["simulation"].update(
        {
            "n_patients": n_patients,
            "av_grid_ms": [90, 150, 210],
            "vv_grid_ms": [-30, 0, 30],
        }
    )

    sim = CircAdaptSimulator(
        model=cfg["simulation"].get("circadapt_model", "VanOsta2024"),
        n_beats=int(cfg["simulation"].get("circadapt_beats", 3)),
        vv_penalty_strength=float(
            cfg["simulation"].get("vv_penalty_strength", 1.0)
        ),
    )
    patients = sample_patients(cfg, n_patients, seed=0)
    ds = build_benchmark_dataset(
        cfg,
        sim,
        patients,
        SOURCE_DOMAIN,
        signal_length=64,
        seed=0,
        verbose=False,
    )

    assert np.isfinite(ds.X).all(), "non-finite features"
    assert np.isfinite(ds.y).all(), "non-finite labels"
    assert ds.X.shape[0] == n_patients * 3 * 3

    model = PairwiseRanker(
        l2=float(cfg["model"]["l2"]),
        margin_frac=float(cfg["model"]["margin_frac"]),
        max_pairs_per_group=int(cfg["model"]["max_pairs_per_group"]),
        seed=0,
    ).fit(ds.X, ds.y, ds.groups)
    metrics = metrics_from_scores(
        ds.y, ds.groups, model.score(ds.X), ds.av, ds.vv, regret_threshold=0.02
    )
    for key, value in metrics.items():
        assert np.isfinite(value), f"non-finite metric {key}: {value}"

    print("SMOKE OK")
    print(f"samples={ds.X.shape[0]} features={ds.X.shape[1]}")
    print(
        "exact_argmax={:.3f} near_optimal={:.3f} mean_regret={:.3f}".format(
            metrics["exact_argmax_accuracy"],
            metrics["near_optimal_fraction"],
            metrics["mean_regret"],
        )
    )


if __name__ == "__main__":
    main()
