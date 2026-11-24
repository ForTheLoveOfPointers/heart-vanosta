"""Synthetic intracardiac impedance.

Conductance-catheter logic: conductance is approximately affine in volume,
G(t) = offset + gain * V(t), and impedance is Z(t) = 1 / G(t).  Gain and offset
are patient/session specific and noisy, which is exactly the nuisance the model
has to be robust to.

The optional ``params`` argument lets the benchmark randomise that nuisance.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def synth_impedance(
    t_ms: np.ndarray,
    volume_ml: np.ndarray,
    patient,
    rng: np.random.Generator,
    noise: float = 0.02,
    params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    if params is None:
        gain = 0.020 + 0.004 * rng.normal()
        offset = 0.45 + 0.05 * rng.normal()
        drift = 0.01
        drift_freq = 0.2
    else:
        gain = params["imp_gain"]
        offset = params["imp_offset"]
        noise = params["imp_noise"]
        drift = params["imp_drift"]
        drift_freq = params["imp_drift_freq"]
    gain = max(gain, 1e-4)

    conductance = offset + gain * volume_ml
    conductance = np.maximum(conductance, 0.05)
    impedance = 1.0 / conductance

    scale = float(np.std(impedance))
    impedance += noise * scale * rng.normal(size=t_ms.shape)
    impedance += drift * scale * np.sin(
        2 * np.pi * drift_freq * t_ms / 1000.0 + rng.uniform(0, 2 * np.pi)
    )
    return impedance
