"""Benchmark dataset with raw device signals and domain randomisation.

For each (patient, AV, VV) it stores:

- the 42 impedance-only hand-crafted features (linear model),
- the raw single-channel resampled impedance waveform (CNN),
- the ground-truth stroke work and patient id.

The impedance observation is randomised according to a ``SignalDomain``.  The
physiology and the labels never see the domain parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .dataset import sample_patient
from .features import extract_features
from .physiology.base import Simulator
from .sensors import resample_signals
from .sensors.pipeline import observe
from .types import PatientParams, Setting


@dataclass
class BenchmarkDataset:
    X: np.ndarray
    y: np.ndarray
    y_asym: np.ndarray
    groups: np.ndarray
    av: np.ndarray
    vv: np.ndarray
    hr: np.ndarray
    signals: np.ndarray
    feature_names: List[str]
    meta: Dict[str, object] = field(default_factory=dict)
    effective_patients: List[PatientParams] = field(default_factory=list, repr=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            X=self.X,
            y=self.y,
            y_asym=self.y_asym,
            groups=self.groups,
            av=self.av,
            vv=self.vv,
            hr=self.hr,
            signals=self.signals,
            feature_names=np.array(self.feature_names),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkDataset":
        data = np.load(path, allow_pickle=False)
        return cls(
            X=data["X"],
            y=data["y"],
            y_asym=data["y_asym"],
            groups=data["groups"],
            av=data["av"],
            vv=data["vv"],
            hr=data["hr"],
            signals=data["signals"],
            feature_names=[str(x) for x in data["feature_names"]],
            meta=json.loads(str(data["meta"])),
        )


def sample_patients(cfg: dict, n_patients: int, seed: int = 0) -> List[PatientParams]:
    """Independent reproducible patient samples.

    Each patient has its own RNG stream, so adding/removing patients does not
    change the others and a failed patient can be resampled deterministically.
    """
    patients: List[PatientParams] = []
    for pid in range(n_patients):
        rng = np.random.default_rng(int(seed) + 1000 * pid)
        patients.append(sample_patient(pid, cfg, rng))
    return patients


def _feature_mode(cfg: dict) -> str:
    return str(cfg.get("features", {}).get("mode", "raw")).lower()


def _simulate_patient(
    cfg: dict,
    simulator: Simulator,
    patient: PatientParams,
    domain,
    signal_length: int,
    seed: int,
) -> List[dict]:
    av_grid: Sequence[float] = cfg["simulation"]["av_grid_ms"]
    vv_grid: Sequence[float] = cfg["simulation"]["vv_grid_ms"]
    # Domain draws must depend on the requested dataset seed, not only on the
    # patient.  Otherwise source and shifted datasets reuse the same random
    # quantiles and are not independent stress tests.
    patient_rng = np.random.default_rng(
        int(seed) + 1_000_003 * int(patient.patient_id)
    )

    rows: List[dict] = []
    for av in av_grid:
        for vv in vv_grid:
            setting = Setting(av_ms=float(av), vv_ms=float(vv))
            beat = simulator.simulate(patient, setting)
            signals = observe(beat, patient, patient_rng, domain=domain)
            feats = extract_features(
                signals, patient, setting, mode=_feature_mode(cfg)
            )
            raw = resample_signals(signals, signal_length)
            rows.append(
                {
                    "group": int(patient.patient_id),
                    "av": float(av),
                    "vv": float(vv),
                    "hr": float(patient.hr_bpm),
                    "y": float(beat.labels.stroke_work),
                    "y_asym": float(beat.labels.work_asymmetry),
                    "feats": feats,
                    "raw": raw,
                }
            )
    return rows


def build_benchmark_dataset(
    cfg: dict,
    simulator: Simulator,
    patients: Sequence[PatientParams],
    domain,
    signal_length: int = 512,
    seed: int = 0,
    verbose: bool = True,
    max_attempts: int = 3,
    resample_on_failure: bool = True,
) -> BenchmarkDataset:
    """Build a raw impedance dataset.

    A patient is only added if *all* of its AV/VV settings simulate correctly.
    If CircAdapt crashes, the patient is resampled with a deterministic new RNG
    stream up to ``max_attempts`` times; otherwise it is skipped and counted.
    Sensor nuisance draws are seeded from ``seed`` and the patient id, so
    source and shifted datasets with different seeds are independent.
    """
    rows: List[dict] = []
    feature_names: List[str] | None = None
    n_skipped = 0
    effective_patients: List[PatientParams] = []

    for i, patient in enumerate(patients):
        trial_rows = None
        for attempt in range(max_attempts):
            trial_patient = patient
            if attempt > 0:
                rng = np.random.default_rng(
                    int(seed) + 100_003 * patient.patient_id + attempt
                )
                trial_patient = sample_patient(patient.patient_id, cfg, rng)
            try:
                trial_rows = _simulate_patient(
                    cfg, simulator, trial_patient, domain, signal_length, seed
                )
                break
            except Exception as exc:  # noqa: BLE001 - CircAdapt can fail numerically
                if not resample_on_failure:
                    raise RuntimeError(
                        f"Patient {patient.patient_id} failed and resampling is "
                        "disabled for this dataset."
                    ) from exc
                if verbose:
                    print(
                        f"  patient {patient.patient_id}: attempt {attempt + 1} "
                        f"failed ({type(exc).__name__}); retrying"
                    )

        if trial_rows is None:
            n_skipped += 1
            if verbose:
                print(f"  patient {patient.patient_id}: skipped after retries")
            continue

        if feature_names is None:
            feature_names = sorted(trial_rows[0]["feats"].keys())
        rows.extend(trial_rows)
        effective_patients.append(trial_patient)

        if verbose and (i + 1) % 20 == 0:
            print(f"  simulated {i + 1}/{len(patients)} patients")

    if not rows or feature_names is None:
        raise RuntimeError("No patient could be simulated.")

    X = np.array([[r["feats"][k] for k in feature_names] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.float64)
    y_asym = np.array([r["y_asym"] for r in rows], dtype=np.float64)
    groups = np.array([r["group"] for r in rows], dtype=np.int32)
    av = np.array([r["av"] for r in rows], dtype=np.float32)
    vv = np.array([r["vv"] for r in rows], dtype=np.float32)
    hr = np.array([r["hr"] for r in rows], dtype=np.float32)
    signals = np.stack([r["raw"] for r in rows]).astype(np.float16)

    return BenchmarkDataset(
        X=X,
        y=y,
        y_asym=y_asym,
        groups=groups,
        av=av,
        vv=vv,
        hr=hr,
        signals=signals,
        feature_names=feature_names,
        meta={
            "requested_patients": int(len(patients)),
            "n_patients": int(len(np.unique(groups))),
            "n_rows": int(len(y)),
            "n_skipped": int(n_skipped),
            "signal_length": int(signal_length),
            "domain": getattr(domain, "name", "unknown"),
            "feature_mode": _feature_mode(cfg),
            "seed": int(seed),
        },
        effective_patients=effective_patients,
    )
