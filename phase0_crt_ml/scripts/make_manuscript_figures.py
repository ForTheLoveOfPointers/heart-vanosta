#!/usr/bin/env python3
"""Regenera las figuras del manuscrito a partir de los reportes primarios.

Salidas (por defecto en ``manuscript/figures/``):

- ``response_surface.png``: superficie AV/VV de un paciente virtual de test.
- ``model_comparison.png``: casi óptimo y regret por modelo, source vs shifted.
- ``learning_curve.png``: curva de aprendizaje del ranker lineal y la CNN.
- ``ablations.png``: ablación de sensores y control negativo.

No entrena modelos ni importa torch/lightgbm: solo lee los artefactos curados.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase0.benchmark import BenchmarkDataset  # noqa: E402

MODEL_ORDER = [
    ("linear", "Linear"),
    ("linear_no_sensors", "Lineal\nsin onda"),
    ("gbt", "LambdaMART"),
    ("cnn", "CNN cruda"),
    ("gbt_cnn_embed", "LGBM\nCNN"),
    ("gbt_hybrid", "LGBM\nhíbrido"),
    ("linear_cnn_embed", "Lineal +\nCNN"),
    ("blend_linear_embed", "Blend"),
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(block: dict, threshold: str = "source_test") -> dict:
    return block[threshold]


def response_surface(reports: Path, figures: Path) -> Path:
    ds = BenchmarkDataset.load(reports / "benchmark_dataset_bench500_imp.npz")
    split_path = reports / "benchmark_split_bench500_imp.json"
    split = _load_json(split_path)
    patient_id = int(split["test_ids"][0])
    mask = ds.groups == patient_id
    av = np.unique(ds.av[mask])
    vv = np.unique(ds.vv[mask])
    surface = np.full((len(vv), len(av)), np.nan)
    for a, v, y in zip(ds.av[mask], ds.vv[mask], ds.y[mask]):
        i = int(np.where(vv == v)[0][0])
        j = int(np.where(av == a)[0][0])
        surface[i, j] = y

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(
        surface,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[av.min(), av.max(), vv.min(), vv.max()],
    )
    best = np.unravel_index(np.nanargmax(surface), surface.shape)
    ax.plot(av[best[1]], vv[best[0]], "*", color="white", markersize=16,
            markeredgecolor="black", label="óptimo")
    ax.set_xlabel("AV (ms)")
    ax.set_ylabel("VV (ms)")
    ax.set_title(f"Superficie AV/VV de trabajo sistólico — paciente virtual {patient_id}")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="trabajo sistólico (u.a.)")
    fig.tight_layout()
    path = figures / "response_surface.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def model_comparison(reports: Path, figures: Path) -> Path:
    report = _load_json(reports / "gbt_report_ladder500_imp_reconcile.json")
    results = report["results"]
    labels, near_src, near_sh, reg_src, reg_sh = [], [], [], [], []
    for key, label in MODEL_ORDER:
        if key not in results:
            continue
        r = results[key]
        labels.append(label)
        near_src.append(r["source_test"]["near_optimal_fraction"])
        near_sh.append(r["shifted_test"]["near_optimal_fraction"])
        reg_src.append(r["source_test"]["mean_regret"])
        reg_sh.append(r["shifted_test"]["mean_regret"])

    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, src, sh, ylabel, title in (
        (axes[0], near_src, near_sh, "fracción casi óptima @2 %", "Mayor es mejor"),
        (axes[1], reg_src, reg_sh, "regret medio", "Menor es mejor"),
    ):
        ax.bar(x - width / 2, src, width, label="in-domain", color="tab:blue")
        ax.bar(x + width / 2, sh, width, label="shifted", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Comparación de modelos al umbral del 2 %")
    fig.tight_layout()
    path = figures / "model_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def learning_curve(reports: Path, figures: Path) -> Path:
    curve = _load_json(reports / "benchmark_learning_curve_bench500_imp.json")
    ns = [c["n_train_patients"] for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, color, label in (
        ("linear", "tab:blue", "linear"),
        ("cnn", "tab:orange", "CNN"),
    ):
        axes[0].plot(
            ns,
            [c[name]["near_optimal_fraction"] for c in curve],
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(
            ns,
            [c[name]["mean_regret"] for c in curve],
            "o-",
            color=color,
            label=label,
        )
    axes[0].set_xlabel("pacientes de entrenamiento")
    axes[0].set_ylabel("fracción casi óptima @2 %")
    axes[0].set_title("Mayor es mejor")
    axes[1].set_xlabel("pacientes de entrenamiento")
    axes[1].set_ylabel("regret medio")
    axes[1].set_title("Menor es mejor")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Curva de aprendizaje (500 pacientes virtuales)")
    fig.tight_layout()
    path = figures / "learning_curve.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def ablations(reports: Path, figures: Path) -> Path:
    bench = _load_json(reports / "benchmark_report_bench500_imp.json")
    gbt = _load_json(reports / "gbt_report_ladder500_imp_reconcile.json")
    rows = [
        ("Completo", bench["linear"]["source_test"]),
        ("Sin onda", bench["linear_no_sensors"]["source_test"]),
    ]
    neg = gbt.get("results", {}).get("gbt_negative_control")
    if neg is not None:
        rows.append(("Etiquetas\nbarajadas", neg["source_test"]))

    labels = [r[0] for r in rows]
    near = [r[1]["near_optimal_fraction"] for r in rows]
    regret = [r[1]["mean_regret"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = ["tab:blue", "tab:gray", "tab:red"][: len(rows)]
    for ax, values, ylabel, title in (
        (axes[0], near, "fracción casi óptima @2 %", "Mayor es mejor"),
        (axes[1], regret, "regret medio", "Menor es mejor"),
    ):
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablación de sensores y control negativo")
    fig.tight_layout()
    path = figures / "ablations.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default=str(ROOT / "reports"))
    parser.add_argument(
        "--figures", default=str(ROOT.parent / "manuscript" / "figures")
    )
    args = parser.parse_args()
    reports = Path(args.reports)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    for fn in (response_surface, model_comparison, learning_curve, ablations):
        path = fn(reports, figures)
        print(f"[figures] {path}")


if __name__ == "__main__":
    main()
