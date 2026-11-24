#!/usr/bin/env python3
"""Model-ladder benchmark: linear ranker vs LambdaMART vs 1D CNN vs hybrids.

Reuses an existing impedance-only benchmark dataset (source + shifted npz and
the saved patient-level split), then trains and compares:

- linear Bradley-Terry ranker (baseline, features),
- linear no-sensor ablation (context/setting only),
- LambdaMART (LightGBM lambdarank) on the 42 features,
- tiny 1D CNN on the raw signal (pairwise loss),
- LambdaMART on CNN embeddings (encoder only),
- hybrid LambdaMART on features + CNN embeddings.

This implements the RESEARCH_PLAN.md model ladder on the primary baseline
(no within-patient normalization).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from phase0.benchmark import BenchmarkDataset
from phase0.evaluate import metrics_from_scores, sensor_feature_masks
from phase0.features import patient_normalize
from phase0.models.pairwise import PairwiseRanker
from phase0.models.gbdt import LambdaMARTRanker, fit_lambda_mart_cv


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


def _load_split(path: Path, groups: np.ndarray):
    split = json.loads(path.read_text(encoding="utf-8"))
    train_ids = set(int(x) for x in split["train_ids"])
    val_ids = set(int(x) for x in split["val_ids"])
    test_ids = set(int(x) for x in split["test_ids"])
    masks = {}
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        masks[name] = np.isin(groups, list(ids))
    if set(np.unique(groups).tolist()) != train_ids | val_ids | test_ids:
        raise RuntimeError("Split file does not cover the dataset patient ids.")
    return masks


def _per_patient_regret(y, groups, scores, eval_mask, regret_threshold):
    """Per-patient (regret, near_optimal) so differences can be bootstrapped."""
    if eval_mask is None:
        eval_mask = np.ones(len(y), dtype=bool)
    else:
        eval_mask = np.asarray(eval_mask, dtype=bool)
    regrets, near = [], []
    for g in np.unique(groups[eval_mask]):
        m = (groups == g) & eval_mask
        yg, sg = y[m], scores[m]
        i_best, i_pred = int(np.argmax(yg)), int(np.argmax(sg))
        best_sw = abs(float(yg[i_best])) + 1e-9
        r = (float(yg[i_best]) - float(yg[i_pred])) / best_sw
        regrets.append(r)
        near.append(float(r <= regret_threshold))
    return np.array(regrets), np.array(near)


def _bootstrap_paired_diff(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    means = np.array(
        [d[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)]
    )
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def _rank_within_patient(scores, groups):
    """Map scores to within-patient rank percentiles in [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    out = np.empty_like(scores)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    for i in range(inverse.max() + 1):
        idx = np.where(inverse == i)[0]
        if len(idx) > 1:
            out[idx] = scores[idx].argsort().argsort() / (len(idx) - 1)
        else:
            out[idx] = 0.5
    return out


def _plot_summary(results, path, model_order):
    labels, src_near, sh_near, src_reg, sh_reg = [], [], [], [], []
    for key, label in model_order:
        if key not in results:
            continue
        r = results[key]
        labels.append(label)
        src_near.append(r["source_test"]["near_optimal_fraction"])
        sh_near.append(r["shifted_test"]["near_optimal_fraction"])
        src_reg.append(r["source_test"]["mean_regret"])
        sh_reg.append(r["shifted_test"]["mean_regret"])

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(x - width / 2, src_near, width, label="in-domain", color="tab:blue")
    axes[0].bar(x + width / 2, sh_near, width, label="shifted", color="tab:orange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("near-optimal fraction")
    axes[0].set_title("Higher is better")
    axes[0].legend()
    axes[1].bar(x - width / 2, src_reg, width, label="in-domain", color="tab:blue")
    axes[1].bar(x + width / 2, sh_reg, width, label="shifted", color="tab:orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("mean regret")
    axes[1].set_title("Lower is better")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument(
        "--dataset-tag",
        default="bench500_imp",
        help="tag of the existing source/shifted/split files to reuse",
    )
    parser.add_argument(
        "--out-tag",
        default=None,
        help="tag for this run's outputs (default: ladder_<dataset-tag>)",
    )
    parser.add_argument(
        "--patient-normalize",
        choices=("none", "imp"),
        default="none",
        help="within-patient normalization; primary baseline is none",
    )
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--reuse-cnn", action="store_true")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gbt-seed", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--secondary-threshold",
        type=float,
        default=0.05,
        help="additional near-optimal tolerance to report (default: 0.05)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out_tag = args.out_tag or f"ladder_{args.dataset_tag}"
    top_k = int(cfg["evaluation"]["top_k"])
    regret_threshold = float(cfg["evaluation"]["regret_threshold"])
    secondary_threshold = float(args.secondary_threshold)

    import torch

    torch.set_num_threads(int(args.threads))

    ds = BenchmarkDataset.load(out / f"benchmark_dataset_{args.dataset_tag}.npz")
    shifted = BenchmarkDataset.load(
        out / f"benchmark_shifted_{args.dataset_tag}.npz"
    )
    masks = _load_split(
        out / f"benchmark_split_{args.dataset_tag}.json", ds.groups
    )
    train_mask, val_mask, test_mask = (
        masks["train"],
        masks["val"],
        masks["test"],
    )
    if len(shifted.y) != int(test_mask.sum()):
        raise RuntimeError("Shifted rows do not match the test split.")
    if not np.array_equal(shifted.groups, ds.groups[test_mask]):
        raise RuntimeError("Shifted rows are not aligned with the test rows.")
    if not np.allclose(shifted.y, ds.y[test_mask]):
        raise RuntimeError("Shifted labels do not match the source labels.")

    pp_prefixes = {"none": [], "imp": ["imp_"]}[args.patient_normalize]
    if pp_prefixes:
        print(f"[ladder] within-patient normalization: {pp_prefixes}")
        ds.X = patient_normalize(ds.X, ds.groups, ds.feature_names, pp_prefixes)
        shifted.X = patient_normalize(
            shifted.X, shifted.groups, shifted.feature_names, pp_prefixes
        )

    score_artifacts = {}

    def eval_scores(scores_test, scores_shifted, scores_val=None, key=None):
        if key is not None:
            score_artifacts[f"{key}_source_test"] = np.asarray(scores_test)
            score_artifacts[f"{key}_shifted_test"] = np.asarray(scores_shifted)
            if scores_val is not None:
                score_artifacts[f"{key}_source_val"] = np.asarray(scores_val)
        rep = {
            "source_test": metrics_from_scores(
                ds.y[test_mask],
                ds.groups[test_mask],
                scores_test,
                ds.av[test_mask],
                ds.vv[test_mask],
                top_k=top_k,
                regret_threshold=regret_threshold,
            ),
            "shifted_test": metrics_from_scores(
                shifted.y,
                shifted.groups,
                scores_shifted,
                shifted.av,
                shifted.vv,
                top_k=top_k,
                regret_threshold=regret_threshold,
            ),
            f"source_test_{secondary_threshold:g}": metrics_from_scores(
                ds.y[test_mask],
                ds.groups[test_mask],
                scores_test,
                ds.av[test_mask],
                ds.vv[test_mask],
                top_k=top_k,
                regret_threshold=secondary_threshold,
            ),
            f"shifted_test_{secondary_threshold:g}": metrics_from_scores(
                shifted.y,
                shifted.groups,
                scores_shifted,
                shifted.av,
                shifted.vv,
                top_k=top_k,
                regret_threshold=secondary_threshold,
            ),
        }
        if scores_val is not None:
            rep["source_val"] = metrics_from_scores(
                ds.y[val_mask],
                ds.groups[val_mask],
                scores_val,
                ds.av[val_mask],
                ds.vv[val_mask],
                top_k=top_k,
                regret_threshold=regret_threshold,
            )
            rep[f"source_val_{secondary_threshold:g}"] = metrics_from_scores(
                ds.y[val_mask],
                ds.groups[val_mask],
                scores_val,
                ds.av[val_mask],
                ds.vv[val_mask],
                top_k=top_k,
                regret_threshold=secondary_threshold,
            )
        return rep

    gbt_seed = int(args.seed if args.gbt_seed is None else args.gbt_seed)
    results = {}
    timings = {}
    t_start = time.time()

    # ------------------------------------------------------------------ linear
    print("[ladder] linear Bradley-Terry ranker")
    t0 = time.time()
    model_kwargs = {
        "l2": float(cfg["model"]["l2"]),
        "margin_frac": float(cfg["model"]["margin_frac"]),
        "max_pairs_per_group": int(cfg["model"]["max_pairs_per_group"]),
        "seed": args.seed,
    }
    linear = PairwiseRanker(**model_kwargs).fit(
        ds.X[train_mask], ds.y[train_mask], ds.groups[train_mask]
    )
    results["linear"] = eval_scores(
        linear.score(ds.X[test_mask]),
        linear.score(shifted.X),
        linear.score(ds.X[val_mask]),
        key="linear",
    )
    timings["linear"] = time.time() - t0

    print("[ladder] linear no-sensors ablation")
    masks_feat = sensor_feature_masks(ds.feature_names)
    ns_mask = masks_feat["no_sensors"]
    linear_ns = PairwiseRanker(**model_kwargs).fit(
        ds.X[train_mask][:, ns_mask],
        ds.y[train_mask],
        ds.groups[train_mask],
    )
    results["linear_no_sensors"] = eval_scores(
        linear_ns.score(ds.X[test_mask][:, ns_mask]),
        linear_ns.score(shifted.X[:, ns_mask]),
        key="linear_no_sensors",
    )

    # --------------------------------------------------------------------- GBT
    print("[ladder] LambdaMART on features (grouped-CV round selection)")
    t0 = time.time()
    gbt, gbt_cv = fit_lambda_mart_cv(
        ds.X[train_mask],
        ds.y[train_mask],
        ds.groups[train_mask],
        seed=gbt_seed,
    )
    timings["gbt"] = time.time() - t0
    print(
        f"[ladder]   cv_best_round={gbt_cv['cv_best_round']} "
        f"val_regret={gbt_cv['cv_mean_val_regret_at_best']:.4f} "
        f"({timings['gbt']:.1f}s)"
    )
    results["gbt"] = eval_scores(
        gbt.score(ds.X[test_mask]),
        gbt.score(shifted.X),
        gbt.score(ds.X[val_mask]),
        key="gbt",
    )
    results["gbt"]["best_iteration"] = gbt.best_iteration
    results["gbt"]["cv"] = {
        k: v for k, v in gbt_cv.items() if k != "cv_curve"
    }

    # ------------------------------------------------------------- CNN + embed
    embeddings = None
    if not args.skip_cnn:
        from phase0.models.cnn import (
            TinyCNN,
            embed_cnn,
            make_context,
            score_cnn,
            train_cnn,
        )

        context = make_context(ds.av, ds.vv, ds.hr)
        shifted_context = make_context(shifted.av, shifted.vv, shifted.hr)
        cnn_path = out / f"cnn_{out_tag}.pt"
        training_path = out / f"cnn_training_{out_tag}.json"

        if args.reuse_cnn and cnn_path.exists() and training_path.exists():
            print(f"[ladder] reusing CNN {cnn_path}")
            meta = json.loads(training_path.read_text(encoding="utf-8"))
            mc = meta.get("model_config", {})
            width = int(mc.get("width", cfg["benchmark"]["cnn_width"]))
            model = TinyCNN(
                in_channels=ds.signals.shape[1],
                context_dim=context.shape[1],
                width=width,
                input_length=ds.signals.shape[2],
            )
            model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
            model.eval()
        else:
            print("[ladder] training tiny CNN")
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
                seed=args.seed,
                verbose=True,
            )
            model = cnn_meta["model"]
            cnn_meta["model_config"] = {
                "width": int(cfg["benchmark"]["cnn_width"]),
                "in_channels": int(ds.signals.shape[1]),
                "context_dim": int(context.shape[1]),
                "input_length": int(ds.signals.shape[2]),
            }
            torch.save(model.state_dict(), cnn_path)
            training_path.write_text(
                json.dumps(
                    _jsonable({k: v for k, v in cnn_meta.items() if k != "model"}),
                    indent=2,
                ),
                encoding="utf-8",
            )
            timings["cnn"] = float(cnn_meta.get("train_time_s", 0.0))
            results["cnn_meta"] = {
                "best_epoch": int(cnn_meta.get("best_epoch", 0)),
                "n_params": int(cnn_meta.get("n_params", 0)),
                "n_pairs": int(cnn_meta.get("n_pairs", 0)),
            }

        t0 = time.time()
        val_scores = score_cnn(model, ds.signals, context, np.where(val_mask)[0])
        test_scores = score_cnn(model, ds.signals, context, np.where(test_mask)[0])
        shifted_scores = score_cnn(model, shifted.signals, shifted_context)
        results["cnn"] = eval_scores(
            test_scores, shifted_scores, val_scores, key="cnn"
        )

        print("[ladder] CNN embeddings for GBT")
        emb = embed_cnn(model, ds.signals, context)
        emb_shifted = embed_cnn(model, shifted.signals, shifted_context)
        embeddings = (emb, emb_shifted)

        def fit_gbt_on(Z, Z_sh, key):
            t0 = time.time()
            m, cv = fit_lambda_mart_cv(
                Z[train_mask],
                ds.y[train_mask],
                ds.groups[train_mask],
                seed=gbt_seed,
            )
            timings[key] = time.time() - t0
            print(
                f"[ladder] {key}: cv_best_round={cv['cv_best_round']} "
                f"({timings[key]:.1f}s)"
            )
            results[key] = eval_scores(
                m.score(Z[test_mask]), m.score(Z_sh), m.score(Z[val_mask]), key=key
            )
            results[key]["best_iteration"] = m.best_iteration
            results[key]["cv"] = {
                k: v for k, v in cv.items() if k != "cv_curve"
            }
            return m

        fit_gbt_on(emb, emb_shifted, "gbt_cnn_embed")
        hybrid = np.concatenate([ds.X.astype(np.float32), emb], axis=1)
        hybrid_sh = np.concatenate(
            [shifted.X.astype(np.float32), emb_shifted], axis=1
        )
        fit_gbt_on(hybrid, hybrid_sh, "gbt_hybrid")

        # ------------------------------------------ linear ranker + CNN embed
        print("[ladder] linear ranker on features + CNN embedding")
        t0 = time.time()
        linear_embed = PairwiseRanker(**model_kwargs).fit(
            hybrid[train_mask], ds.y[train_mask], ds.groups[train_mask]
        )
        timings["linear_cnn_embed"] = time.time() - t0
        results["linear_cnn_embed"] = eval_scores(
            linear_embed.score(hybrid[test_mask]),
            linear_embed.score(hybrid_sh),
            linear_embed.score(hybrid[val_mask]),
            key="linear_cnn_embed",
        )

        # Rank-average blend with the features-only linear ranker.  The weight
        # is selected on the validation split (never on test) by validation
        # near-optimal fraction, breaking ties on validation regret.
        print("[ladder] linear<->linear+embed rank blend (weight from val)")
        val_lin = linear.score(ds.X[val_mask])
        val_he = linear_embed.score(hybrid[val_mask])
        best = None
        for w in np.linspace(0.0, 1.0, 11):
            blended = w * _rank_within_patient(
                val_lin, ds.groups[val_mask]
            ) + (1.0 - w) * _rank_within_patient(val_he, ds.groups[val_mask])
            met = metrics_from_scores(
                ds.y[val_mask], ds.groups[val_mask], blended,
                ds.av[val_mask], ds.vv[val_mask],
                top_k=top_k, regret_threshold=regret_threshold,
            )
            key = (
                round(met["near_optimal_fraction"], 6),
                -round(met["mean_regret"], 6),
            )
            if best is None or key > best[0]:
                best = (key, float(w), met)
        blend_w = best[1]
        print(f"[ladder]   blend weight w={blend_w:.2f} (val-selected)")

        def rank_blend(scores_lin, scores_he, groups):
            return blend_w * _rank_within_patient(
                scores_lin, groups
            ) + (1.0 - blend_w) * _rank_within_patient(scores_he, groups)

        blend_test = rank_blend(
            linear.score(ds.X[test_mask]),
            linear_embed.score(hybrid[test_mask]),
            ds.groups[test_mask],
        )
        blend_sh = rank_blend(
            linear.score(shifted.X),
            linear_embed.score(hybrid_sh),
            shifted.groups,
        )
        blend_val = rank_blend(val_lin, val_he, ds.groups[val_mask])
        results["blend_linear_embed"] = eval_scores(
            blend_test, blend_sh, blend_val, key="blend_linear_embed"
        )
        results["blend_linear_embed"]["blend_weight"] = blend_w

    # ------------------------------------------------- negative control (GBT)
    print("[ladder] GBT negative control (labels shuffled within patient)")
    rng = np.random.default_rng(args.seed + 777)
    y_shuf = ds.y.copy()
    for g in np.unique(ds.groups[train_mask]):
        m = (ds.groups == g) & train_mask
        y_shuf[m] = rng.permutation(ds.y[m])
    gbt_nc, _ = fit_lambda_mart_cv(
        ds.X[train_mask],
        y_shuf[train_mask],
        ds.groups[train_mask],
        seed=gbt_seed,
    )
    results["gbt_negative_control"] = eval_scores(
        gbt_nc.score(ds.X[test_mask]), gbt_nc.score(shifted.X),
        key="gbt_negative_control",
    )

    # --------------------------------------------- learning curve (GBT regret)
    print("[ladder] GBT learning curve")
    train_ids = np.unique(ds.groups[train_mask])
    rng_lc = np.random.default_rng(args.seed)
    rng_lc.shuffle(train_ids)
    curve = []
    for n in [10, 20, 40, 80, 150, len(train_ids)]:
        n = min(int(n), len(train_ids))
        chosen = train_ids[:n]
        m = np.isin(ds.groups, chosen)
        lc, _ = fit_lambda_mart_cv(
            ds.X[m], ds.y[m], ds.groups[m], n_folds=3, seed=gbt_seed
        )
        met = metrics_from_scores(
            ds.y[test_mask], ds.groups[test_mask], lc.score(ds.X[test_mask]),
            ds.av[test_mask], ds.vv[test_mask],
            top_k=top_k, regret_threshold=regret_threshold,
        )
        met["n_train_patients"] = float(n)
        curve.append(met)
        print(
            f"[ladder]   n={n:3d} near={met['near_optimal_fraction']:.3f} "
            f"regret={met['mean_regret']:.4f}"
        )
    results["gbt_learning_curve"] = curve

    # -------------------------------------------------- paired bootstrap (GBT)
    print("[ladder] paired bootstrap: gbt - linear on test patients")
    lin_scores = linear.score(ds.X[test_mask])
    gbt_scores = gbt.score(ds.X[test_mask])
    lin_sh = linear.score(shifted.X)
    gbt_sh = gbt.score(shifted.X)
    reg_l, near_l = _per_patient_regret(
        ds.y[test_mask], ds.groups[test_mask], lin_scores, None, regret_threshold
    )
    reg_g, near_g = _per_patient_regret(
        ds.y[test_mask], ds.groups[test_mask], gbt_scores, None, regret_threshold
    )
    reg_l_sh, near_l_sh = _per_patient_regret(
        shifted.y, shifted.groups, lin_sh, None, regret_threshold
    )
    reg_g_sh, near_g_sh = _per_patient_regret(
        shifted.y, shifted.groups, gbt_sh, None, regret_threshold
    )
    bootstrap = {
        "source": {
            "near_optimal_diff": _bootstrap_paired_diff(
                near_g, near_l, args.n_boot, args.seed
            ),
            "regret_diff": _bootstrap_paired_diff(
                reg_g, reg_l, args.n_boot, args.seed
            ),
        },
        "shifted": {
            "near_optimal_diff": _bootstrap_paired_diff(
                near_g_sh, near_l_sh, args.n_boot, args.seed
            ),
            "regret_diff": _bootstrap_paired_diff(
                reg_g_sh, reg_l_sh, args.n_boot, args.seed
            ),
        },
    }
    print(
        "[ladder]   source near-opt diff {mean_diff:+.3f} "
        "[{ci_low:+.3f}, {ci_high:+.3f}]".format(
            **bootstrap["source"]["near_optimal_diff"]
        )
    )

    # ------------------------------- paired bootstrap: blend - linear (test)
    blend_bootstrap = None
    if "blend_linear_embed" in results:
        print("[ladder] paired bootstrap: blend - linear on test patients")
        blend_test_scores = rank_blend(
            linear.score(ds.X[test_mask]),
            linear_embed.score(hybrid[test_mask]),
            ds.groups[test_mask],
        )
        blend_sh_scores = rank_blend(
            linear.score(shifted.X),
            linear_embed.score(hybrid_sh),
            shifted.groups,
        )
        reg_b, near_b = _per_patient_regret(
            ds.y[test_mask], ds.groups[test_mask], blend_test_scores, None,
            regret_threshold,
        )
        reg_b_sh, near_b_sh = _per_patient_regret(
            shifted.y, shifted.groups, blend_sh_scores, None, regret_threshold
        )
        blend_bootstrap = {
            "source": {
                "near_optimal_diff": _bootstrap_paired_diff(
                    near_b, near_l, args.n_boot, args.seed
                ),
                "regret_diff": _bootstrap_paired_diff(
                    reg_b, reg_l, args.n_boot, args.seed
                ),
            },
            "shifted": {
                "near_optimal_diff": _bootstrap_paired_diff(
                    near_b_sh, near_l_sh, args.n_boot, args.seed
                ),
                "regret_diff": _bootstrap_paired_diff(
                    reg_b_sh, reg_l_sh, args.n_boot, args.seed
                ),
            },
        }
        print(
            "[ladder]   source near-opt diff {mean_diff:+.3f} "
            "[{ci_low:+.3f}, {ci_high:+.3f}]".format(
                **blend_bootstrap["source"]["near_optimal_diff"]
            )
        )

    # ------------------------------------------------------- feature importance
    imp = gbt.feature_importance()
    imp_frac = imp / max(imp.sum(), 1e-12)
    names = np.array(ds.feature_names)
    order = np.argsort(imp_frac)[::-1][:15]
    importance = [
        {"feature": str(names[i]), "gain_fraction": float(imp_frac[i])}
        for i in order
    ]

    # ------------------------------------------------------------------- save
    report = {
        "dataset_tag": args.dataset_tag,
        "out_tag": out_tag,
        "dataset": {**ds.meta, "n_train_patients": int(train_mask.sum() and len(np.unique(ds.groups[train_mask]))),
                    "n_val_patients": int(len(np.unique(ds.groups[val_mask]))),
                    "n_test_patients": int(len(np.unique(ds.groups[test_mask])))},
        "patient_normalize": pp_prefixes,
        "primary_threshold": regret_threshold,
        "secondary_threshold": secondary_threshold,
        "results": results,
        "bootstrap_gbt_vs_linear": bootstrap,
        "bootstrap_blend_vs_linear": blend_bootstrap,
        "gbt_feature_importance": importance,
        "timings_s": timings,
        "runtime_s": time.time() - t_start,
    }
    (out / f"gbt_report_{out_tag}.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )

    gbt.model.save_model(str(out / f"gbt_{out_tag}.txt"))
    np.savez(
        out / f"linear_{out_tag}.npz",
        w=linear.w, mu=linear.mu, sd=linear.sd,
        feature_names=np.array(ds.feature_names),
    )
    np.savez(out / f"scores_{out_tag}.npz", **score_artifacts)

    # ----------------------------------------------------------------- report
    model_order = [
        ("linear", "Linear"),
        ("linear_no_sensors", "Linear\nno sensors"),
        ("gbt", "LambdaMART\nfeatures"),
        ("cnn", "Tiny CNN\nraw signal"),
        ("gbt_cnn_embed", "LambdaMART\nCNN embed"),
        ("gbt_hybrid", "LambdaMART\nhybrid"),
        ("linear_cnn_embed", "Linear\n+CNN embed"),
        ("blend_linear_embed", "Blend\nlinear+embed"),
    ]
    rows = []
    rows5 = []
    for key, label in model_order:
        if key in results:
            r = results[key]
            rows.append(
                f"| {label.replace(chr(10), ' ')} | "
                f"{r['source_test']['near_optimal_fraction']:.3f} | "
                f"{r['source_test']['mean_regret']:.4f} | "
                f"{r['shifted_test']['near_optimal_fraction']:.3f} | "
                f"{r['shifted_test']['mean_regret']:.4f} |"
            )
            k5_src = f"source_test_{secondary_threshold:g}"
            k5_sh = f"shifted_test_{secondary_threshold:g}"
            rows5.append(
                f"| {label.replace(chr(10), ' ')} | "
                f"{r[k5_src]['near_optimal_fraction']:.3f} | "
                f"{r[k5_sh]['near_optimal_fraction']:.3f} |"
            )
    lines = [
        f"# Model ladder benchmark: {report['dataset']['n_patients']} patients "
        f"(tag {out_tag})",
        "",
        "Same dataset, patient-level split and metrics as the linear baseline.",
        f"Within-patient normalization: {pp_prefixes or 'none'} (primary baseline).",
        "",
        "| Model | In-domain near-optimal | In-domain regret | Shifted near-optimal | Shifted regret |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        f"## Near-optimal fraction at {secondary_threshold:.0%} tolerance",
        "",
        "| Model | In-domain near-optimal | Shifted near-optimal |",
        "|---|---:|---:|",
        *rows5,
        "",
        "## GBT vs linear, paired bootstrap (95% CI, test patients)",
        "",
        f"- source near-optimal: {bootstrap['source']['near_optimal_diff']['mean_diff']:+.3f} "
        f"[{bootstrap['source']['near_optimal_diff']['ci_low']:+.3f}, "
        f"{bootstrap['source']['near_optimal_diff']['ci_high']:+.3f}]",
        f"- source regret: {bootstrap['source']['regret_diff']['mean_diff']:+.4f} "
        f"[{bootstrap['source']['regret_diff']['ci_low']:+.4f}, "
        f"{bootstrap['source']['regret_diff']['ci_high']:+.4f}]",
        f"- shifted near-optimal: {bootstrap['shifted']['near_optimal_diff']['mean_diff']:+.3f} "
        f"[{bootstrap['shifted']['near_optimal_diff']['ci_low']:+.3f}, "
        f"{bootstrap['shifted']['near_optimal_diff']['ci_high']:+.3f}]",
        f"- shifted regret: {bootstrap['shifted']['regret_diff']['mean_diff']:+.4f} "
        f"[{bootstrap['shifted']['regret_diff']['ci_low']:+.4f}, "
        f"{bootstrap['shifted']['regret_diff']['ci_high']:+.4f}]",
        "",
        "## Negative control",
        "",
        f"- GBT with within-patient shuffled labels: near-optimal "
        f"{results['gbt_negative_control']['source_test']['near_optimal_fraction']:.3f}, "
        f"Spearman {results['gbt_negative_control']['source_test']['mean_spearman']:.3f}.",
        "",
        "## GBT learning curve (test-set regret vs train patients)",
        "",
        "| n train patients | near-optimal | regret |",
        "|---:|---:|---:|",
        *[
            f"| {int(c['n_train_patients'])} | {c['near_optimal_fraction']:.3f} | "
            f"{c['mean_regret']:.4f} |"
            for c in curve
        ],
        "",
        "## Top GBT features by gain",
        "",
        "| Feature | gain fraction |",
        "|---|---:|",
        *[
            f"| {fi['feature']} | {fi['gain_fraction']:.3f} |"
            for fi in importance[:10]
        ],
        "",
        "In-silico demonstration only; not clinical validation.",
    ]
    (out / f"gbt_summary_{out_tag}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _plot_summary(results, out / f"gbt_summary_{out_tag}.png", model_order)

    print(json.dumps(_jsonable({
        k: results[k]["source_test"]
        for k in ("linear", "gbt", "cnn", "gbt_cnn_embed", "gbt_hybrid")
        if k in results
    }), indent=2))
    print(f"[ladder] results written to {out} (tag {out_tag})")


if __name__ == "__main__":
    main()
