#!/usr/bin/env python3
"""Bootstrap CIs for a linear-only benchmark, including patient normalization."""

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
from phase0.evaluate import sensor_feature_masks
from phase0.features import patient_normalize
from phase0.models.pairwise import PairwiseRanker


def _per_patient(y, groups, scores, threshold=0.02):
    regrets, near = {}, {}
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


def _bootstrap(values, n_boot, rng):
    ids = np.array(sorted(values))
    arr = np.array([values[int(i)] for i in ids], dtype=float)
    means = np.array(
        [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    )
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_diff(a, b, n_boot, rng):
    ids = np.array(sorted(set(a) & set(b)))
    av = np.array([a[int(i)] for i in ids], dtype=float)
    bv = np.array([b[int(i)] for i in ids], dtype=float)
    diffs = av - bv
    boot = np.array(
        [diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(n_boot)]
    )
    return float(diffs.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="bench500_pp")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    out = Path(args.out)
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    ds = BenchmarkDataset.load(out / f"benchmark_dataset_{args.tag}.npz")
    shifted = BenchmarkDataset.load(out / f"benchmark_shifted_{args.tag}.npz")
    split = json.loads((out / f"benchmark_split_{args.tag}.json").read_text())
    train_ids = set(int(x) for x in split["train_ids"])
    test_ids = set(int(x) for x in split["test_ids"])
    train_mask = np.isin(ds.groups, list(train_ids))
    test_mask = np.isin(ds.groups, list(test_ids))

    lin = np.load(out / f"linear_{args.tag}.npz", allow_pickle=True)
    w, mu, sd = lin["w"], lin["mu"], lin["sd"]
    pp = tuple(str(x) for x in lin["patient_normalize"]) if "patient_normalize" in lin else ()
    if pp:
        ds.X = patient_normalize(ds.X, ds.groups, ds.feature_names, pp)
        shifted.X = patient_normalize(
            shifted.X, shifted.groups, shifted.feature_names, pp
        )

    masks = sensor_feature_masks(ds.feature_names)
    no_sensors = masks["no_sensors"]
    model_kwargs = {
        "l2": float(cfg["model"]["l2"]),
        "margin_frac": float(cfg["model"]["margin_frac"]),
        "max_pairs_per_group": int(cfg["model"]["max_pairs_per_group"]),
        "seed": args.seed,
    }
    ns_model = PairwiseRanker(**model_kwargs).fit(
        ds.X[train_mask][:, no_sensors],
        ds.y[train_mask],
        ds.groups[train_mask],
    )

    scores = {
        "linear_source": ((ds.X[test_mask] - mu) / sd) @ w,
        "linear_shifted": ((shifted.X - mu) / sd) @ w,
        "no_sensors_source": ns_model.score(ds.X[test_mask][:, no_sensors]),
        "no_sensors_shifted": ns_model.score(shifted.X[:, no_sensors]),
    }
    target = {
        "linear_source": (ds.y[test_mask], ds.groups[test_mask]),
        "linear_shifted": (shifted.y, shifted.groups),
        "no_sensors_source": (ds.y[test_mask], ds.groups[test_mask]),
        "no_sensors_shifted": (shifted.y, shifted.groups),
    }

    threshold = float(cfg["evaluation"]["regret_threshold"])
    rng = np.random.default_rng(args.seed)
    report = {
        "tag": args.tag,
        "n_test_patients": int(len(np.unique(ds.groups[test_mask]))),
        "n_boot": args.n_boot,
        "patient_normalize": list(pp),
        "metrics": {},
        "paired_all_minus_no": {},
    }
    per = {}
    for name, sc in scores.items():
        y, g = target[name]
        regrets, near = _per_patient(y, g, sc, threshold)
        per[name] = (regrets, near)
        report["metrics"][name] = {
            "near": dict(zip(("mean", "lo", "hi"), _bootstrap(near, args.n_boot, rng))),
            "regret": dict(zip(("mean", "lo", "hi"), _bootstrap(regrets, args.n_boot, rng))),
        }

    for domain in ("source", "shifted"):
        lin_r, lin_n = per[f"linear_{domain}"]
        ns_r, ns_n = per[f"no_sensors_{domain}"]
        report["paired_all_minus_no"][domain] = {
            "near": dict(zip(("mean", "lo", "hi"), _bootstrap_diff(lin_n, ns_n, args.n_boot, rng))),
            "regret": dict(zip(("mean", "lo", "hi"), _bootstrap_diff(lin_r, ns_r, args.n_boot, rng))),
        }

    path = out / f"benchmark_ci_{args.tag}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
