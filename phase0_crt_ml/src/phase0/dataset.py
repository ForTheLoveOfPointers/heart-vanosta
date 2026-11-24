"""Build the synthetic labelled dataset for Phase 0.

Each row is (patient, AV, VV, device features, ground-truth stroke work).  Labels
come from the simulator; features come from the sensor forward model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .features import extract_features
from .physiology.base import Simulator
from .sensors import observe
from .types import PatientParams, Setting


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    y_asym: np.ndarray
    groups: np.ndarray
    av: np.ndarray
    vv: np.ndarray
    feature_names: List[str]
    meta: Dict[str, object]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            y_asym=self.y_asym,
            groups=self.groups,
            av=self.av,
            vv=self.vv,
            feature_names=np.array(self.feature_names),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        data = np.load(path, allow_pickle=True)
        return cls(
            X=data["X"],
            y=data["y"],
            y_asym=data["y_asym"],
            groups=data["groups"],
            av=data["av"],
            vv=data["vv"],
            feature_names=[str(x) for x in data["feature_names"]],
            meta={},
        )


def sample_patient(patient_id: int, cfg: dict, rng: np.random.Generator) -> PatientParams:
    r = cfg["simulation"]["patient_ranges"]

    def u(key: str) -> float:
        lo, hi = r[key]
        return float(rng.uniform(lo, hi))

    return PatientParams(
        patient_id=patient_id,
        hr_bpm=u("hr_bpm"),
        contractility=u("contractility"),
        lv_viability=u("lv_viability"),
        septal_viability=u("septal_viability"),
        lateral_viability=u("lateral_viability"),
        rv_viability=u("rv_viability") if "rv_viability" in r else 1.0,
        preload=u("preload"),
        afterload=u("afterload"),
        intrinsic_av_ms=u("intrinsic_av_ms"),
        qrs_ms=u("qrs_ms"),
        dyssynchrony_ms=u("dyssynchrony_ms"),
        lead_lv_delay_ms=u("lead_lv_delay_ms"),
        lead_rv_delay_ms=u("lead_rv_delay_ms"),
        av_block=bool(rng.random() < cfg["simulation"]["av_block_prob"]),
        biventricular_pacing=True,
        seed=int(rng.integers(0, 2**31 - 1)),
    )


def build_dataset(
    cfg: dict,
    simulator: Simulator,
    n_patients: int,
    seed: int | None = None,
    verbose: bool = True,
    max_attempts: int = 3,
) -> Dataset:
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    rng = np.random.default_rng(seed)

    av_grid: Sequence[float] = cfg["simulation"]["av_grid_ms"]
    vv_grid: Sequence[float] = cfg["simulation"]["vv_grid_ms"]

    rows: List[dict] = []
    feature_names: List[str] | None = None
    n_skipped = 0

    for pid in range(n_patients):
        patient_rows = None
        for attempt in range(max(1, int(max_attempts))):
            patient_rng_state = (
                rng
                if attempt == 0
                else np.random.default_rng(seed + 1_000_003 * pid + attempt)
            )
            patient = sample_patient(pid, cfg, patient_rng_state)
            patient_rng = np.random.default_rng(patient.seed)
            trial_rows: List[dict] = []
            try:
                for av in av_grid:
                    for vv in vv_grid:
                        setting = Setting(av_ms=float(av), vv_ms=float(vv))
                        beat = simulator.simulate(patient, setting)
                        signals = observe(beat, patient, patient_rng)
                        feats = extract_features(signals, patient, setting)
                        trial_rows.append(
                            {
                                "group": pid,
                                "av": float(av),
                                "vv": float(vv),
                                "y": beat.labels.stroke_work,
                                "y_asym": beat.labels.work_asymmetry,
                                "feats": feats,
                            }
                        )
            except Exception as exc:  # noqa: BLE001 - CircAdapt can fail numerically
                if verbose:
                    print(
                        f"  patient {pid}: attempt {attempt + 1} failed "
                        f"({type(exc).__name__}); retrying"
                    )
                continue
            patient_rows = trial_rows
            break

        if patient_rows is None:
            n_skipped += 1
            if verbose:
                print(f"  patient {pid}: skipped after retries")
            continue

        if feature_names is None:
            feature_names = sorted(patient_rows[0]["feats"].keys())
        rows.extend(patient_rows)

        if verbose and (pid + 1) % 20 == 0:
            print(f"  simulated {pid + 1}/{n_patients} patients")

    if feature_names is None:
        raise RuntimeError("No patient could be simulated.")
    X = np.array([[row["feats"][k] for k in feature_names] for row in rows], dtype=float)
    y = np.array([row["y"] for row in rows], dtype=float)
    y_asym = np.array([row["y_asym"] for row in rows], dtype=float)
    groups = np.array([row["group"] for row in rows], dtype=int)
    av = np.array([row["av"] for row in rows], dtype=float)
    vv = np.array([row["vv"] for row in rows], dtype=float)

    return Dataset(
        X=X,
        y=y,
        y_asym=y_asym,
        groups=groups,
        av=av,
        vv=vv,
        feature_names=feature_names,
        meta={"n_patients": len(np.unique(groups)), "seed": seed, "n_skipped": n_skipped},
    )
