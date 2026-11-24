#!/usr/bin/env python3
"""Benchmark: linear ranker vs tiny raw-signal CNN.

The script builds (or reuses) a domain-randomised CircAdapt dataset, evaluates a
linear pairwise ranker and a tiny 1-D CNN on the same patient-level split, and
tests both on an in-domain and a shifted sensor domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from phase0.benchmark import BenchmarkDataset, build_benchmark_dataset, sample_patients
from phase0.evaluate import (
    evaluate_ranker,
    metrics_from_scores,
    sensor_feature_masks,
    three_way_split,
)
from phase0.features import patient_normalize
from phase0.models.pairwise import PairwiseRanker
from phase0.sensors import SHIFTED_DOMAIN, SOURCE_DOMAIN
from phase0.types import PatientParams


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


def _load_patients(path: Path) -> list[PatientParams]:
    return [PatientParams(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _save_patients(patients: list[PatientParams], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(p) for p in patients], indent=2), encoding="utf-8"
    )


def _config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_path(out: Path, tag: str) -> Path:
    return out / f"benchmark_split_{tag}.json"


def _save_split(path: Path, seed: int, groups: np.ndarray, train_mask, val_mask, test_mask) -> None:
    payload = {
        "seed": int(seed),
        "train_ids": sorted(np.unique(groups[train_mask]).astype(int).tolist()),
        "val_ids": sorted(np.unique(groups[val_mask]).astype(int).tolist()),
        "test_ids": sorted(np.unique(groups[test_mask]).astype(int).tolist()),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _label_surface_stats(ds, mask) -> dict:
    avs = np.unique(ds.av[mask])
    vvs = np.unique(ds.vv[mask])
    av_opts, vv_opts = [], []
    for g in np.unique(ds.groups[mask]):
        m = (ds.groups == g) & mask
        if m.sum() < 2:
            continue
        i = int(np.argmax(ds.y[m]))
        av_opts.append(float(ds.av[m][i]))
        vv_opts.append(float(ds.vv[m][i]))
    n = len(av_opts)
    if n == 0:
        return {"n_patients": 0}
    av_opts = np.asarray(av_opts)
    vv_opts = np.asarray(vv_opts)
    return {
        "n_patients": int(n),
        "av_min": float(avs.min()),
        "av_max": float(avs.max()),
        "vv_min": float(vvs.min()),
        "vv_max": float(vvs.max()),
        "av_opt_at_min_fraction": float(np.mean(av_opts == avs.min())),
        "av_opt_at_max_fraction": float(np.mean(av_opts == avs.max())),
        "vv_opt_at_min_fraction": float(np.mean(vv_opts == vvs.min())),
        "vv_opt_at_max_fraction": float(np.mean(vv_opts == vvs.max())),
    }


def _subset_metrics(ds, mask, scores, cfg):
    return metrics_from_scores(
        ds.y[mask],
        ds.groups[mask],
        scores,
        ds.av[mask],
        ds.vv[mask],
        top_k=int(cfg["evaluation"]["top_k"]),
        regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
    )


def _plot_summary(report: dict, path: Path) -> None:
    models = [
        ("Linear", "linear"),
        ("Linear\nno sensors", "linear_no_sensors"),
        ("Tiny CNN", "cnn"),
    ]
    labels = [m[0] for m in models]
    source_near = [report[m[1]]["source_test"]["near_optimal_fraction"] for m in models]
    shifted_near = [report[m[1]]["shifted_test"]["near_optimal_fraction"] for m in models]
    source_regret = [report[m[1]]["source_test"]["mean_regret"] for m in models]
    shifted_regret = [report[m[1]]["shifted_test"]["mean_regret"] for m in models]

    x = np.arange(len(models))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(x - width / 2, source_near, width, label="in-domain", color="tab:blue")
    axes[0].bar(x + width / 2, shifted_near, width, label="shifted", color="tab:orange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("near-optimal fraction")
    axes[0].set_title("Higher is better")
    axes[0].legend()

    axes[1].bar(x - width / 2, source_regret, width, label="in-domain", color="tab:blue")
    axes[1].bar(x + width / 2, shifted_regret, width, label="shifted", color="tab:orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("mean regret")
    axes[1].set_title("Lower is better")
    axes[1].legend()

    fig.suptitle(
        f"Phase 0 benchmark: {report.get('dataset', {}).get('n_patients', '?')} virtual patients"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_surface(ds, test_mask, linear_scores, cnn_scores, path: Path) -> None:
    test_ids = np.unique(ds.groups[test_mask])
    if len(test_ids) == 0:
        return
    g = test_ids[0]
    m = ds.groups == g
    avs = np.unique(ds.av)
    vvs = np.unique(ds.vv)
    if len(avs) * len(vvs) != int(m.sum()):
        return

    true_grid = ds.y[m].reshape(len(avs), len(vvs))
    linear_grid = linear_scores.reshape(len(avs), len(vvs))
    cnn_grid = cnn_scores.reshape(len(avs), len(vvs))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, grid, title in (
        (axes[0], true_grid, "true stroke work"),
        (axes[1], linear_grid, "linear score"),
        (axes[2], cnn_grid, "tiny CNN score"),
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
    fig.suptitle(f"example test patient {int(g)}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_markdown(report: dict, path: Path) -> None:
    def row(name: str) -> str:
        r = report[name]
        return (
            f"| {name} | {r['source_test']['near_optimal_fraction']:.3f} | "
            f"{r['source_test']['mean_regret']:.4f} | "
            f"{r['shifted_test']['near_optimal_fraction']:.3f} | "
            f"{r['shifted_test']['mean_regret']:.4f} |"
        )

    n_patients = report.get("dataset", {}).get("n_patients", "?")
    lines = [
        f"# Phase 0 benchmark: {n_patients} virtual patients",
        "",
        "CircAdapt physiology, domain-randomised device signals, patient-level split.",
        "",
        "| Model | In-domain near-optimal | In-domain regret | Shifted near-optimal | Shifted regret |",
        "|---|---:|---:|---:|---:|",
        row("linear"),
        row("linear_no_sensors"),
        row("cnn"),
        "",
        "The shifted domain is a synthetic stress test (noisier signals, different",
        "impedance calibration, broader EGM wavelets), not real patient data.",
        "",
        (
            f"CNN: about {report['cnn']['n_params']} parameters, "
            f"best epoch {report['cnn']['best_epoch']}."
            if report["cnn"].get("n_params", 0)
            else "CNN: skipped for this linear-only run."
        ),
        "",
        "## Linear sensor ablations",
        "",
        "| Feature set | In-domain near-optimal | In-domain regret | Shifted near-optimal | Shifted regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in report.get("linear_ablations", {}):
        r = report["linear_ablations"][name]
        lines.append(
            f"| {name} | {r['source_test']['near_optimal_fraction']:.3f} | "
            f"{r['source_test']['mean_regret']:.4f} | "
            f"{r['shifted_test']['near_optimal_fraction']:.3f} | "
            f"{r['shifted_test']['mean_regret']:.4f} |"
        )
    surf = report.get("label_surface", {}).get("source_test", {})
    if surf:
        lines.extend(
            [
                "",
                "## Label-surface diagnostics",
                "",
                f"AV grid: {surf['av_min']:.0f}-{surf['av_max']:.0f} ms; "
                f"optimum at max: {surf['av_opt_at_max_fraction']:.1%}. "
                f"VV grid: {surf['vv_min']:.0f}-{surf['vv_max']:.0f} ms; "
                f"optimum at max: {surf['vv_opt_at_max_fraction']:.1%}.",
            ]
        )
    lines.extend(
        [
            "",
            "This demonstrates *learnability in simulation* and robustness to a",
            "controlled domain shift. It is not clinical validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--patients", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument("--tag", default="benchmark")
    parser.add_argument("--signal-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-cnn", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument(
        "--features",
        choices=("raw", "normalized"),
        default=None,
        help="Feature mode; defaults to config features.mode.",
    )
    parser.add_argument(
        "--patient-normalize",
        choices=("none", "imp"),
        default=None,
        help="Within-patient impedance-feature standardization; defaults to config.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.features is not None:
        cfg.setdefault("features", {})["mode"] = args.features
    if args.patient_normalize is not None:
        prefix_map = {
            "none": [],
            "imp": ["imp_"],
        }
        cfg.setdefault("features", {})["patient_normalize"] = prefix_map[
            args.patient_normalize
        ]
    bench_cfg = cfg.setdefault("benchmark", {})
    if args.patients is not None:
        bench_cfg["n_patients"] = args.patients
    if args.signal_length is not None:
        bench_cfg["signal_length"] = args.signal_length
    if args.epochs is not None:
        bench_cfg["cnn_epochs"] = args.epochs
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    n_patients = int(bench_cfg.get("n_patients", 1000))
    signal_length = int(bench_cfg.get("signal_length", 512))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds_path = out / f"benchmark_dataset_{args.tag}.npz"
    shifted_path = out / f"benchmark_shifted_{args.tag}.npz"
    patients_path = out / f"benchmark_patients_{args.tag}.json"
    split_path = _split_path(out, args.tag)
    config_hash = _config_hash({"config": cfg, "seed": seed})

    import torch

    torch.set_num_threads(int(args.threads))

    t_start = time.time()
    if args.reuse_data and ds_path.exists() and patients_path.exists():
        print(f"[benchmark] reusing {ds_path}")
        ds = BenchmarkDataset.load(ds_path)
        old_hash = ds.meta.get("config_hash")
        if old_hash is None:
            print("[benchmark] WARNING: reused dataset has no config hash")
        elif old_hash != config_hash:
            raise RuntimeError(
                "Reused dataset was built with a different config. "
                "Delete the reports for this tag or change --tag."
            )
        patients = _load_patients(patients_path)
    else:
        from phase0.physiology.circadapt_backend import CircAdaptSimulator

        print(f"[benchmark] building {n_patients} patients, signal_length={signal_length}")
        patients = sample_patients(cfg, n_patients, seed=seed)
        sim = CircAdaptSimulator(
            model=cfg["simulation"].get("circadapt_model", "VanOsta2024"),
            n_beats=int(cfg["simulation"].get("circadapt_beats", 3)),
            vv_penalty_strength=float(
                cfg["simulation"].get("vv_penalty_strength", 1.0)
            ),
        )
        ds = build_benchmark_dataset(
            cfg,
            sim,
            patients,
            SOURCE_DOMAIN,
            signal_length=signal_length,
            seed=seed,
            verbose=True,
        )
        ds.meta["config_hash"] = config_hash
        ds.save(ds_path)
        patients = ds.effective_patients
        _save_patients(patients, patients_path)
        print(f"[benchmark] dataset saved to {ds_path}")

    print(
        f"[benchmark] rows={len(ds.y)} patients={len(np.unique(ds.groups))} "
        f"skipped={ds.meta.get('n_skipped', 0)}"
    )

    if args.reuse_data and split_path.exists():
        split = _load_split(split_path)
        train_ids = set(int(x) for x in split["train_ids"])
        val_ids = set(int(x) for x in split["val_ids"])
        test_ids = set(int(x) for x in split["test_ids"])
        train_mask = np.isin(ds.groups, list(train_ids))
        val_mask = np.isin(ds.groups, list(val_ids))
        test_mask = np.isin(ds.groups, list(test_ids))
        split_seed = int(split["seed"])
        if set(np.unique(ds.groups).tolist()) != train_ids | val_ids | test_ids:
            raise RuntimeError("Split file does not cover the dataset patient ids.")
    else:
        split_seed = seed
        train_mask, val_mask, test_mask = three_way_split(
            ds.groups,
            val_frac=float(bench_cfg.get("val_frac", 0.15)),
            test_frac=float(bench_cfg.get("test_frac", 0.15)),
            seed=split_seed,
        )
        _save_split(split_path, split_seed, ds.groups, train_mask, val_mask, test_mask)

    test_ids = sorted(np.unique(ds.groups[test_mask]).tolist())
    by_id = {int(p.patient_id): p for p in patients}
    missing = [i for i in test_ids if i not in by_id]
    if missing:
        raise RuntimeError(f"Missing PatientParams for test ids: {missing[:5]}")
    test_patients = [by_id[i] for i in test_ids]
    print(
        f"[benchmark] train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"test={int(test_mask.sum())} rows"
    )

    if args.reuse_data and shifted_path.exists():
        print(f"[benchmark] reusing {shifted_path}")
        shifted = BenchmarkDataset.load(shifted_path)
        old_hash = shifted.meta.get("config_hash")
        if old_hash is None:
            print("[benchmark] WARNING: reused shifted dataset has no config hash")
        elif old_hash != config_hash:
            raise RuntimeError(
                "Reused shifted dataset was built with a different config. "
                "Delete the reports for this tag or change --tag."
            )
    else:
        from phase0.physiology.circadapt_backend import CircAdaptSimulator

        print("[benchmark] building shifted test domain")
        sim = CircAdaptSimulator(
            model=cfg["simulation"].get("circadapt_model", "VanOsta2024"),
            n_beats=int(cfg["simulation"].get("circadapt_beats", 3)),
            vv_penalty_strength=float(
                cfg["simulation"].get("vv_penalty_strength", 1.0)
            ),
        )
        shifted = build_benchmark_dataset(
            cfg,
            sim,
            test_patients,
            SHIFTED_DOMAIN,
            signal_length=signal_length,
            seed=seed + 10_000,
            verbose=True,
            max_attempts=1,
            resample_on_failure=False,
        )
        shifted.meta["config_hash"] = config_hash
        shifted.save(shifted_path)
        print(f"[benchmark] shifted dataset saved to {shifted_path}")

    if len(shifted.y) != int(test_mask.sum()):
        raise RuntimeError(
            f"Shifted rows ({len(shifted.y)}) do not match test rows ({int(test_mask.sum())})."
        )
    if not np.array_equal(shifted.groups, ds.groups[test_mask]) or not np.allclose(
        shifted.av, ds.av[test_mask]
    ) or not np.allclose(shifted.vv, ds.vv[test_mask]):
        raise RuntimeError("Shifted test rows are not aligned with the source test rows.")
    if not np.allclose(shifted.y, ds.y[test_mask]) or not np.allclose(
        shifted.hr, ds.hr[test_mask]
    ):
        raise RuntimeError(
            "Shifted physiology labels/HR do not match the source test rows; "
            "the shift must change only the sensor observation, not the patient."
        )

    pp_prefixes = tuple(cfg.get("features", {}).get("patient_normalize", []) or [])
    if pp_prefixes:
        print(f"[benchmark] within-patient normalization: {list(pp_prefixes)}")
        ds.X = patient_normalize(ds.X, ds.groups, ds.feature_names, pp_prefixes)
        shifted.X = patient_normalize(
            shifted.X, shifted.groups, shifted.feature_names, pp_prefixes
        )

    model_kwargs = {
        "l2": float(cfg["model"]["l2"]),
        "margin_frac": float(cfg["model"]["margin_frac"]),
        "max_pairs_per_group": int(cfg["model"]["max_pairs_per_group"]),
        "seed": seed,
    }

    print("[benchmark] training linear ranker")
    ranker = PairwiseRanker(**model_kwargs).fit(
        ds.X[train_mask], ds.y[train_mask], ds.groups[train_mask]
    )
    linear_report = {
        "source_val": evaluate_ranker(
            ranker, ds, eval_mask=val_mask,
            top_k=int(cfg["evaluation"]["top_k"]),
            regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
        ),
        "source_test": evaluate_ranker(
            ranker, ds, eval_mask=test_mask,
            top_k=int(cfg["evaluation"]["top_k"]),
            regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
        ),
        "shifted_test": metrics_from_scores(
            shifted.y,
            shifted.groups,
            ranker.score(shifted.X),
            shifted.av,
            shifted.vv,
            top_k=int(cfg["evaluation"]["top_k"]),
            regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
        ),
    }

    print("[benchmark] training linear sensor ablations")
    masks = sensor_feature_masks(ds.feature_names)
    linear_ablations = {}
    for name, mask in masks.items():
        ablation_ranker = PairwiseRanker(**model_kwargs).fit(
            ds.X[train_mask][:, mask],
            ds.y[train_mask],
            ds.groups[train_mask],
        )
        linear_ablations[name] = {
            "source_test": evaluate_ranker(
                ablation_ranker,
                ds,
                feature_mask=mask,
                eval_mask=test_mask,
                top_k=int(cfg["evaluation"]["top_k"]),
                regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
            ),
            "shifted_test": metrics_from_scores(
                shifted.y,
                shifted.groups,
                ablation_ranker.score(shifted.X[:, mask]),
                shifted.av,
                shifted.vv,
                top_k=int(cfg["evaluation"]["top_k"]),
                regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
            ),
        }

    linear_ns_report = linear_ablations["no_sensors"]
    report = {
        "config": _jsonable(cfg),
        "config_hash": config_hash,
        "dataset": {
            **ds.meta,
            "n_train_patients": int(len(np.unique(ds.groups[train_mask]))),
            "n_val_patients": int(len(np.unique(ds.groups[val_mask]))),
            "n_test_patients": int(len(np.unique(ds.groups[test_mask]))),
        },
        "split": {
            "seed": int(split_seed),
            "train_ids": sorted(np.unique(ds.groups[train_mask]).astype(int).tolist()),
            "val_ids": sorted(np.unique(ds.groups[val_mask]).astype(int).tolist()),
            "test_ids": sorted(np.unique(ds.groups[test_mask]).astype(int).tolist()),
        },
        "label_surface": {
            "source_test": _label_surface_stats(ds, test_mask),
            "shifted_test": _label_surface_stats(
                shifted, np.ones(len(shifted.y), dtype=bool)
            ),
        },
        "linear": linear_report,
        "linear_no_sensors": linear_ns_report,
        "linear_ablations": linear_ablations,
        "domain": {
            "source": asdict(SOURCE_DOMAIN),
            "shifted": asdict(SHIFTED_DOMAIN),
        },
        "feature_mode": str(ds.meta.get("feature_mode", "raw")),
        "patient_normalize": list(pp_prefixes),
    }

    np.savez(
        out / f"linear_{args.tag}.npz",
        w=ranker.w,
        mu=ranker.mu,
        sd=ranker.sd,
        feature_names=np.array(ds.feature_names),
        feature_mode=np.array(str(ds.meta.get("feature_mode", "raw"))),
        patient_normalize=np.array(list(pp_prefixes)),
    )

    if not args.skip_cnn:
        from phase0.models.cnn import TinyCNN, make_context, score_cnn, train_cnn

        context = make_context(ds.av, ds.vv, ds.hr)
        shifted_context = make_context(shifted.av, shifted.vv, shifted.hr)

        cnn_path = out / f"cnn_{args.tag}.pt"
        training_path = out / f"cnn_training_{args.tag}.json"
        if args.reuse_cnn and cnn_path.exists() and training_path.exists():
            print(f"[benchmark] reusing CNN {cnn_path}")
            cnn_meta = json.loads(training_path.read_text(encoding="utf-8"))
            model_cfg = cnn_meta.get("model_config", {})
            width = int(model_cfg.get("width", bench_cfg.get("cnn_width", 8)))
            for key, actual in (
                ("in_channels", ds.signals.shape[1]),
                ("context_dim", context.shape[1]),
                ("input_length", ds.signals.shape[2]),
            ):
                saved = model_cfg.get(key)
                if saved is not None and int(saved) != int(actual):
                    raise RuntimeError(
                        f"CNN checkpoint {key}={saved} does not match data ({actual})."
                    )
            model = TinyCNN(
                in_channels=ds.signals.shape[1],
                context_dim=context.shape[1],
                width=width,
                input_length=ds.signals.shape[2],
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
            model.eval()
            cnn_meta["model"] = model
        else:
            print("[benchmark] training tiny CNN")
            cnn_meta = train_cnn(
                ds.signals,
                context,
                ds.y,
                ds.groups,
                ds.av,
                ds.vv,
                train_mask,
                val_mask,
                cfg,
                seed=seed,
                verbose=True,
            )
            cnn_meta["model_config"] = {
                "width": int(cfg["benchmark"]["cnn_width"]),
                "in_channels": int(ds.signals.shape[1]),
                "context_dim": int(context.shape[1]),
                "input_length": int(ds.signals.shape[2]),
            }
            torch.save(cnn_meta["model"].state_dict(), cnn_path)
            training_path.write_text(
                json.dumps(
                    _jsonable({k: v for k, v in cnn_meta.items() if k != "model"}),
                    indent=2,
                ),
                encoding="utf-8",
            )

        model = cnn_meta["model"]
        val_scores = score_cnn(model, ds.signals, context, np.where(val_mask)[0])
        test_scores = score_cnn(model, ds.signals, context, np.where(test_mask)[0])
        shifted_scores = score_cnn(model, shifted.signals, shifted_context)
        report["cnn"] = {
            "source_val": _subset_metrics(ds, val_mask, val_scores, cfg),
            "source_test": _subset_metrics(ds, test_mask, test_scores, cfg),
            "shifted_test": metrics_from_scores(
                shifted.y,
                shifted.groups,
                shifted_scores,
                shifted.av,
                shifted.vv,
                top_k=int(cfg["evaluation"]["top_k"]),
                regret_threshold=float(cfg["evaluation"]["regret_threshold"]),
            ),
            "n_params": int(cnn_meta.get("n_params", 0)),
            "best_epoch": int(cnn_meta.get("best_epoch", 0)),
            "best_val_regret": float(cnn_meta.get("best_val_regret", float("nan"))),
            "train_time_s": float(cnn_meta.get("train_time_s", 0.0)),
            "n_pairs": int(cnn_meta.get("n_pairs", 0)),
            "model_config": cnn_meta.get("model_config", {}),
            "history": cnn_meta.get("history", []),
        }
    else:
        report["cnn"] = {
            "source_test": {"near_optimal_fraction": float("nan"), "mean_regret": float("nan")},
            "shifted_test": {"near_optimal_fraction": float("nan"), "mean_regret": float("nan")},
            "n_params": 0,
            "best_epoch": 0,
        }

    report["runtime_s"] = time.time() - t_start
    (out / f"benchmark_report_{args.tag}.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )
    _plot_summary(report, out / f"benchmark_summary_{args.tag}.png")
    _write_markdown(report, out / f"benchmark_summary_{args.tag}.md")

    if not args.skip_cnn:
        g = int(np.unique(ds.groups[test_mask])[0])
        m = ds.groups == g
        _plot_surface(
            ds,
            test_mask,
            ranker.score(ds.X[m]),
            score_cnn(model, ds.signals, context, np.where(m)[0]),
            out / f"benchmark_surface_{args.tag}.png",
        )

    print(json.dumps(_jsonable({
        "linear": report["linear"]["source_test"],
        "linear_no_sensors": report["linear_no_sensors"]["source_test"],
        "cnn": report["cnn"]["source_test"],
    }), indent=2))
    print(f"[benchmark] results written to {out}")


if __name__ == "__main__":
    main()
