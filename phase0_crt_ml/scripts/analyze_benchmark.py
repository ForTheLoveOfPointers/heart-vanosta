#!/usr/bin/env python3
"""Bootstrap confidence intervals for the 1,000-patient benchmark."""

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
from phase0.evaluate import three_way_split
from phase0.features import patient_normalize
from phase0.models.cnn import TinyCNN, make_context, score_cnn


def _per_patient(y, groups, scores, threshold):
    regrets = {}
    near = {}
    for g in np.unique(groups):
        m = groups == g
        yg = y[m]
        sg = scores[m]
        ib = int(np.argmax(yg))
        ip = int(np.argmax(sg))
        r = (float(yg[ib]) - float(yg[ip])) / (abs(float(yg[ib])) + 1e-9)
        regrets[int(g)] = r
        near[int(g)] = float(r <= threshold)
    return regrets, near


def _bootstrap(values: dict, n_boot: int, rng):
    ids = np.array(sorted(values.keys()))
    arr = np.array([values[int(i)] for i in ids], dtype=float)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[b] = sample.mean()
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_diff(a: dict, b: dict, n_boot: int, rng):
    ids = np.array(sorted(set(a) & set(b)))
    av = np.array([a[int(i)] for i in ids], dtype=float)
    bv = np.array([b[int(i)] for i in ids], dtype=float)
    diffs = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        diffs[k] = (av[idx] - bv[idx]).mean()
    return float((av - bv).mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="bench1000")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    out = Path(args.out)
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    ds = BenchmarkDataset.load(out / f"benchmark_dataset_{args.tag}.npz")
    shifted = BenchmarkDataset.load(out / f"benchmark_shifted_{args.tag}.npz")

    split_path = out / f"benchmark_split_{args.tag}.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        test_ids = set(int(x) for x in split["test_ids"])
        train_ids = set(int(x) for x in split["train_ids"])
        val_ids = set(int(x) for x in split["val_ids"])
        train_mask = np.isin(ds.groups, list(train_ids))
        val_mask = np.isin(ds.groups, list(val_ids))
        test_mask = np.isin(ds.groups, list(test_ids))
    else:
        print("[analyze] WARNING: split file missing; falling back to config seed")
        train_mask, val_mask, test_mask = three_way_split(
            ds.groups, val_frac=0.15, test_frac=0.15, seed=int(cfg.get("seed", 0))
        )

    lin = np.load(out / f"linear_{args.tag}.npz", allow_pickle=True)
    w, mu, sd = lin["w"], lin["mu"], lin["sd"]
    saved_features = [str(x) for x in lin["feature_names"]]
    if saved_features != list(ds.feature_names):
        raise RuntimeError("Linear checkpoint feature order does not match the dataset.")
    pp = tuple(str(x) for x in lin["patient_normalize"]) if "patient_normalize" in lin else ()
    if pp:
        ds.X = patient_normalize(ds.X, ds.groups, ds.feature_names, pp)
        shifted.X = patient_normalize(
            shifted.X, shifted.groups, shifted.feature_names, pp
        )

    def linear_scores(X):
        return ((X - mu) / sd) @ w

    training_path = out / f"cnn_training_{args.tag}.json"
    width = int(cfg["benchmark"]["cnn_width"])
    if training_path.exists():
        training_meta = json.loads(training_path.read_text(encoding="utf-8"))
        model_cfg = training_meta.get("model_config", {})
        width = int(model_cfg.get("width", width))
        for key, actual in (
            ("in_channels", ds.signals.shape[1]),
            ("context_dim", 3),
            ("input_length", ds.signals.shape[2]),
        ):
            saved = model_cfg.get(key)
            if saved is not None and int(saved) != int(actual):
                raise RuntimeError(
                    f"CNN checkpoint {key}={saved} does not match data ({actual})."
                )
    model = TinyCNN(
        in_channels=ds.signals.shape[1],
        context_dim=3,
        width=width,
        input_length=ds.signals.shape[2],
    )
    model.load_state_dict(torch.load(out / f"cnn_{args.tag}.pt", map_location="cpu"))
    model.eval()
    context = make_context(ds.av, ds.vv, ds.hr)
    shifted_context = make_context(shifted.av, shifted.vv, shifted.hr)

    scores = {
        "linear_source": linear_scores(ds.X[test_mask]),
        "cnn_source": score_cnn(model, ds.signals, context, np.where(test_mask)[0]),
        "linear_shifted": linear_scores(shifted.X),
        "cnn_shifted": score_cnn(model, shifted.signals, shifted_context),
    }
    y_source = ds.y[test_mask]
    g_source = ds.groups[test_mask]
    y_shifted = shifted.y
    g_shifted = shifted.groups

    threshold = float(cfg["evaluation"]["regret_threshold"])
    rng = np.random.default_rng(args.seed)
    report = {"n_patients": int(len(np.unique(g_source))), "n_boot": args.n_boot, "metrics": {}}

    for name, sc in scores.items():
        y, g = (y_source, g_source) if "source" in name else (y_shifted, g_shifted)
        regrets, near = _per_patient(y, g, sc, threshold)
        report["metrics"][name] = {
            "near": dict(zip(("mean", "lo", "hi"), _bootstrap(near, args.n_boot, rng))),
            "regret": dict(zip(("mean", "lo", "hi"), _bootstrap(regrets, args.n_boot, rng))),
        }

    report["paired_diff_cnn_minus_linear"] = {}
    for domain in ("source", "shifted"):
        y, g = (y_source, g_source) if domain == "source" else (y_shifted, g_shifted)
        lin_r, lin_n = _per_patient(y, g, scores[f"linear_{domain}"], threshold)
        cnn_r, cnn_n = _per_patient(y, g, scores[f"cnn_{domain}"], threshold)
        report["paired_diff_cnn_minus_linear"][domain] = {
            "near": dict(zip(("mean", "lo", "hi"), _bootstrap_diff(cnn_n, lin_n, args.n_boot, rng))),
            "regret": dict(zip(("mean", "lo", "hi"), _bootstrap_diff(cnn_r, lin_r, args.n_boot, rng))),
        }

    (out / f"benchmark_ci_{args.tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    labels = ["linear\nsource", "CNN\nsource", "linear\nshifted", "CNN\nshifted"]
    keys = ["linear_source", "cnn_source", "linear_shifted", "cnn_shifted"]
    near = [report["metrics"][k]["near"]["mean"] for k in keys]
    near_err = [
        [
            report["metrics"][k]["near"]["mean"] - report["metrics"][k]["near"]["lo"]
            for k in keys
        ],
        [
            report["metrics"][k]["near"]["hi"] - report["metrics"][k]["near"]["mean"]
            for k in keys
        ],
    ]
    regret = [report["metrics"][k]["regret"]["mean"] for k in keys]
    regret_err = [
        [
            report["metrics"][k]["regret"]["mean"] - report["metrics"][k]["regret"]["lo"]
            for k in keys
        ],
        [
            report["metrics"][k]["regret"]["hi"] - report["metrics"][k]["regret"]["mean"]
            for k in keys
        ],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(keys))
    colors = ["tab:blue", "tab:orange", "tab:blue", "tab:orange"]
    axes[0].bar(x, near, color=colors, yerr=near_err, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("near-optimal fraction")
    axes[1].bar(x, regret, color=colors, yerr=regret_err, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("mean regret")
    fig.suptitle("Bootstrap 95% CI (1,000 patients, 150 test)")
    fig.tight_layout()
    fig.savefig(out / f"benchmark_ci_{args.tag}.png", dpi=150)
    plt.close(fig)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
