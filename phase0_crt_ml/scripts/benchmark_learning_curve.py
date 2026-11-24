#!/usr/bin/env python3
"""Learning curve for the linear ranker and the tiny CNN.

Uses the already-generated benchmark dataset.  For each training size, both
models are trained from scratch on the same patient subset and evaluated on the
same held-out test patients.
"""

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
import torch
import yaml

from phase0.benchmark import BenchmarkDataset
from phase0.evaluate import metrics_from_scores, three_way_split
from phase0.models.cnn import make_context, score_cnn, train_cnn
from phase0.models.pairwise import PairwiseRanker


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="bench500_imp")
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80, 150, 350])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    if args.epochs is not None:
        cfg["benchmark"]["cnn_epochs"] = args.epochs
    ds = BenchmarkDataset.load(out / f"benchmark_dataset_{args.tag}.npz")
    split_path = out / f"benchmark_split_{args.tag}.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_mask = np.isin(ds.groups, [int(x) for x in split["train_ids"]])
        val_mask = np.isin(ds.groups, [int(x) for x in split["val_ids"]])
        test_mask = np.isin(ds.groups, [int(x) for x in split["test_ids"]])
    else:
        print("[curve] WARNING: split file missing; falling back to config seed")
        train_mask, val_mask, test_mask = three_way_split(
            ds.groups, val_frac=0.15, test_frac=0.15, seed=int(cfg.get("seed", 0))
        )
    context = make_context(ds.av, ds.vv, ds.hr)

    train_ids = np.unique(ds.groups[train_mask])
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    rng.shuffle(train_ids)

    model_kwargs = {
        "l2": float(cfg["model"]["l2"]),
        "margin_frac": float(cfg["model"]["margin_frac"]),
        "max_pairs_per_group": int(cfg["model"]["max_pairs_per_group"]),
        "seed": int(cfg.get("seed", 0)),
    }
    top_k = int(cfg["evaluation"]["top_k"])
    threshold = float(cfg["evaluation"]["regret_threshold"])

    curve = []
    seen_sizes: set[int] = set()
    for n in args.sizes:
        n = min(int(n), len(train_ids))
        if n in seen_sizes:
            continue
        seen_sizes.add(n)
        chosen = train_ids[:n]
        sub = np.isin(ds.groups, chosen)
        print(f"[curve] n={n} patients")

        lin = PairwiseRanker(**model_kwargs).fit(
            ds.X[sub], ds.y[sub], ds.groups[sub]
        )
        lin_scores = lin.score(ds.X[test_mask])
        lin_metrics = metrics_from_scores(
            ds.y[test_mask],
            ds.groups[test_mask],
            lin_scores,
            ds.av[test_mask],
            ds.vv[test_mask],
            top_k=top_k,
            regret_threshold=threshold,
        )

        cnn = train_cnn(
            ds.signals,
            context,
            ds.y,
            ds.groups,
            ds.av,
            ds.vv,
            sub,
            val_mask,
            cfg,
            seed=int(cfg.get("seed", 0)),
            verbose=False,
        )
        cnn_scores = score_cnn(
            cnn["model"], ds.signals, context, np.where(test_mask)[0]
        )
        cnn_metrics = metrics_from_scores(
            ds.y[test_mask],
            ds.groups[test_mask],
            cnn_scores,
            ds.av[test_mask],
            ds.vv[test_mask],
            top_k=top_k,
            regret_threshold=threshold,
        )
        curve.append(
            {
                "n_train_patients": int(n),
                "linear": lin_metrics,
                "cnn": cnn_metrics,
                "cnn_best_epoch": int(cnn["best_epoch"]),
                "cnn_params": int(cnn["n_params"]),
                "cnn_epochs": int(cfg["benchmark"]["cnn_epochs"]),
            }
        )
        print(
            f"  linear near={lin_metrics['near_optimal_fraction']:.3f} "
            f"regret={lin_metrics['mean_regret']:.4f} | "
            f"cnn near={cnn_metrics['near_optimal_fraction']:.3f} "
            f"regret={cnn_metrics['mean_regret']:.4f} "
            f"(best epoch {cnn['best_epoch']})"
        )

    (out / f"benchmark_learning_curve_{args.tag}.json").write_text(
        json.dumps(_jsonable(curve), indent=2), encoding="utf-8"
    )

    ns = [c["n_train_patients"] for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, color in (("linear", "tab:blue"), ("cnn", "tab:orange")):
        axes[0].plot(
            ns,
            [c[name]["near_optimal_fraction"] for c in curve],
            "o-",
            color=color,
            label=name,
        )
        axes[1].plot(
            ns,
            [c[name]["mean_regret"] for c in curve],
            "o-",
            color=color,
            label=name,
        )
    axes[0].set_xlabel("training patients")
    axes[0].set_ylabel("near-optimal fraction")
    axes[0].set_title("Higher is better")
    axes[1].set_xlabel("training patients")
    axes[1].set_ylabel("mean regret")
    axes[1].set_title("Lower is better")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Learning curve: linear vs tiny CNN "
        f"({len(np.unique(ds.groups))} virtual patients)"
    )
    fig.tight_layout()
    fig.savefig(out / f"benchmark_learning_curve_{args.tag}.png", dpi=150)
    plt.close(fig)
    print(f"[curve] written to {out}")


if __name__ == "__main__":
    main()
