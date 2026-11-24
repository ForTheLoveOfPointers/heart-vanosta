"""Hand-crafted features extracted from intracardiac impedance.

These features are the input to the linear ranking model.  They are
intentionally interpretable and cheap.  The project uses a single device sensor:
intracardiac impedance.  Heart rate, intrinsic AV and QRS are device context,
not waveform sensors.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import detrend

from .types import BeatSignals, PatientParams, Setting


def _robust_standardize(sig: np.ndarray) -> np.ndarray:
    """Remove offset and scale using median/MAD; affine-invariant."""
    med = float(np.median(sig))
    mad = float(np.median(np.abs(sig - med)))
    if mad < 1e-12:
        mad = float(np.std(sig))
    if mad < 1e-12:
        mad = 1.0
    return (sig - med) / mad


def normalize_impedance(signals: BeatSignals) -> BeatSignals:
    """Detrend and robustly standardize the impedance waveform.

    This makes the per-beat features invariant to gain and offset, which are
    calibration nuisance parameters in the forward model.
    """
    impedance = _robust_standardize(
        detrend(np.asarray(signals.impedance, dtype=float), type="linear")
    )
    return BeatSignals(t_ms=signals.t_ms, impedance=impedance)


def extract_features(
    signals: BeatSignals,
    patient: PatientParams,
    setting: Setting,
    mode: str = "raw",
) -> Dict[str, float]:
    """Extract the impedance-only feature vector.

    ``mode='normalized'`` applies linear detrending and robust standardisation
    before feature computation.  Feature names and dimensionality are identical,
    so raw and normalized runs are directly comparable.
    """
    if mode not in {"raw", "normalized"}:
        raise ValueError(f"Unknown feature mode: {mode!r}")
    if mode == "normalized":
        signals = normalize_impedance(signals)

    out: Dict[str, float] = {}
    t = signals.t_ms
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    cycle_ms = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    imp = signals.impedance
    out["imp_min"] = float(np.min(imp))
    out["imp_max"] = float(np.max(imp))
    out["imp_stroke"] = float(np.max(imp) - np.min(imp))
    out["imp_argmin"] = float(np.argmin(imp) * dt)
    out["imp_argmax"] = float(np.argmax(imp) * dt)
    out["imp_std"] = float(np.std(imp))
    out["imp_dzdt_max"] = float(np.max(np.abs(np.gradient(imp, dt))))
    ejection = abs(out["imp_argmin"] - out["imp_argmax"])
    if cycle_ms > 0:
        ejection = min(ejection, cycle_ms - ejection)
    out["imp_ejection_time"] = float(ejection)

    out["ctx_hr"] = float(patient.hr_bpm)
    # Device-derived context: intrinsic AV conduction and QRS can be measured
    # by the device (e.g. by temporarily inhibiting pacing).
    out["ctx_intrinsic_av_est"] = float(patient.intrinsic_av_ms)
    out["ctx_qrs_est"] = float(patient.qrs_ms)

    av = float(setting.av_ms)
    vv = float(setting.vv_ms)
    out["set_av"] = av
    out["set_vv"] = vv
    out["set_av_sq"] = av * av
    out["set_vv_sq"] = vv * vv
    out["set_av_x_vv"] = av * vv

    # Setting x context interactions let a linear-in-parameters model represent
    # a patient-specific response surface.
    setting_keys = ["set_av", "set_vv", "set_av_sq", "set_vv_sq", "set_av_x_vv"]
    interaction_keys = [
        "ctx_intrinsic_av_est",
        "ctx_qrs_est",
        "ctx_hr",
        "imp_stroke",
        "imp_ejection_time",
    ]
    for s in setting_keys:
        for c in interaction_keys:
            out[f"ix_{s}_x_{c}"] = out[s] * out[c]

    out["ix_av_minus_ctx_intrinsic_sq"] = (
        out["set_av"] - out["ctx_intrinsic_av_est"]
    ) ** 2

    return out


def patient_normalize(
    X: np.ndarray,
    groups: np.ndarray,
    feature_names,
    prefixes=("imp_",),
    eps: float = 1e-9,
) -> np.ndarray:
    """Standardize selected features within each patient across settings.

    A device sweep evaluates all AV/VV candidates for one patient.  Centering
    and scaling a signal feature across that sweep removes patient/session gain
    and offset while retaining how the feature changes with the setting.  No
    labels are used.  Context and setting columns are left untouched.

    Interaction terms are recomputed after their source columns are
    standardized, so a normalized single-sensor model does not mix standardized
    base features with raw-scale ``ix_*_x_imp_*`` terms.
    """
    X = np.asarray(X, dtype=np.float64).copy()
    groups = np.asarray(groups)
    names = [str(n) for n in feature_names]
    prefixes = tuple(prefixes)
    if not prefixes:
        return X
    cols = np.array([n.startswith(prefixes) for n in names], dtype=bool)
    if not np.any(cols):
        return X
    name_to_idx = {name: i for i, name in enumerate(names)}
    interaction_cols = []
    for i, name in enumerate(names):
        if not name.startswith("ix_") or "_x_" not in name:
            continue
        left, right = name[3:].split("_x_", 1)
        if left in name_to_idx and right in name_to_idx:
            interaction_cols.append((i, name_to_idx[left], name_to_idx[right]))
    for group in np.unique(groups):
        mask = groups == group
        values = X[np.ix_(mask, cols)]
        center = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale > eps, scale, 1.0)
        X[np.ix_(mask, cols)] = (values - center) / scale
        for target, left, right in interaction_cols:
            if cols[left] or cols[right]:
                X[np.ix_(mask, [target])] = (
                    X[np.ix_(mask, [left])] * X[np.ix_(mask, [right])]
                )
    return X
