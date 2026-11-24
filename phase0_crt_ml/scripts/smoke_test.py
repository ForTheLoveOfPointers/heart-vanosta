#!/usr/bin/env python3
"""Tiny end-to-end CircAdapt check.  Run this first after setup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from phase0.dataset import build_dataset
from phase0.physiology import CircAdaptSimulator
from phase0.train import run_experiment


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    n_patients = 4
    cfg["simulation"].update(
        {
            "n_patients": n_patients,
            "av_grid_ms": [90, 150, 210],
            "vv_grid_ms": [-30, 0, 30],
        }
    )
    cfg["evaluation"]["learning_curve_patients"] = [2, 4]

    sim = CircAdaptSimulator(
        model=cfg["simulation"].get("circadapt_model", "VanOsta2024"),
        n_beats=int(cfg["simulation"].get("circadapt_beats", 3)),
        vv_penalty_strength=float(
            cfg["simulation"].get("vv_penalty_strength", 1.0)
        ),
    )
    ds = build_dataset(cfg, sim, n_patients=n_patients, seed=0, verbose=False)

    assert np.isfinite(ds.X).all(), "non-finite features"
    assert np.isfinite(ds.y).all(), "non-finite labels"
    assert ds.X.shape[0] == n_patients * 3 * 3

    res = run_experiment(ds, cfg)
    for key, value in res["main"].items():
        assert np.isfinite(value), f"non-finite metric {key}: {value}"

    print("SMOKE OK")
    print(f"samples={ds.X.shape[0]} features={ds.X.shape[1]}")
    print(
        "exact_argmax={:.3f} near_optimal={:.3f} mean_regret={:.3f}".format(
            res["main"]["exact_argmax_accuracy"],
            res["main"]["near_optimal_fraction"],
            res["main"]["mean_regret"],
        )
    )


if __name__ == "__main__":
    main()
