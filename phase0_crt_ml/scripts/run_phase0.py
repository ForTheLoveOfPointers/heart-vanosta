#!/usr/bin/env python3
"""Run the Phase 0 learnability experiment end to end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from phase0.dataset import build_dataset
from phase0.physiology import CircAdaptSimulator
from phase0.train import run_experiment


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _plot_learning_curve(curve, path: Path) -> None:
    ns = [c["n_train_patients"] for c in curve]
    regret = [c["mean_regret"] for c in curve]
    near = [c["near_optimal_fraction"] for c in curve]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ns, regret, "o-", color="tab:red", label="mean regret")
    ax1.set_xlabel("training patients")
    ax1.set_ylabel("mean regret", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(ns, near, "s-", color="tab:blue", label="near-optimal fraction")
    ax2.set_ylabel("near-optimal fraction", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ablations(ablations, path: Path) -> None:
    names = list(ablations.keys())
    near = [ablations[n]["near_optimal_fraction"] for n in names]
    regret = [ablations[n]["mean_regret"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, near, width=0.4, label="near-optimal fraction")
    ax.bar(x + 0.2, regret, width=0.4, label="mean regret")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_example_surface(ds, ranker, test_mask, path: Path) -> None:
    test_ids = np.unique(ds.groups[test_mask])
    if len(test_ids) == 0:
        return
    g = test_ids[0]
    m = ds.groups == g
    scores = ranker.score(ds.X[m])
    avs = np.unique(ds.av)
    vvs = np.unique(ds.vv)
    if len(avs) * len(vvs) != int(m.sum()):
        return

    true_grid = ds.y[m].reshape(len(avs), len(vvs))
    pred_grid = scores.reshape(len(avs), len(vvs))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, grid, title in (
        (axes[0], true_grid, "true stroke work"),
        (axes[1], pred_grid, "predicted score"),
    ):
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(vvs)))
        ax.set_xticklabels([f"{v:+.0f}" for v in vvs])
        ax.set_yticks(range(len(avs)))
        ax.set_yticklabels([f"{a:.0f}" for a in avs])
        ax.set_xlabel("VV delay (ms)")
        ax.set_ylabel("AV delay (ms)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"patient {int(g)}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--patients", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.patients is not None:
        cfg["simulation"]["n_patients"] = args.patients
    n = int(cfg["simulation"].get("n_patients", 100))

    print(f"[phase0] backend=circadapt patients={n}")
    sim = CircAdaptSimulator(
        model=cfg["simulation"].get("circadapt_model", "VanOsta2024"),
        n_beats=int(cfg["simulation"].get("circadapt_beats", 3)),
        vv_penalty_strength=float(
            cfg["simulation"].get("vv_penalty_strength", 1.0)
        ),
    )

    ds = build_dataset(cfg, sim, n_patients=n, seed=int(cfg["seed"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds.save(out / f"dataset_{args.tag}.npz")

    res = run_experiment(ds, cfg)
    report = {k: v for k, v in res.items() if k not in ("ranker", "train_mask", "test_mask")}
    (out / f"report_{args.tag}.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )

    print(json.dumps(_jsonable(res["main"]), indent=2))

    _plot_learning_curve(res["learning_curve"], out / f"learning_curve_{args.tag}.png")
    _plot_ablations(res["ablations"], out / f"ablations_{args.tag}.png")
    _plot_example_surface(ds, res["ranker"], res["test_mask"], out / f"surface_{args.tag}.png")
    print(f"[phase0] results written to {out}")


if __name__ == "__main__":
    main()
