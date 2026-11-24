"""Impedance nuisance-domain definitions.

A ``SignalDomain`` describes the *nuisance* parameters of the intracardiac
impedance observation model: gain, offset, noise, and baseline drift.  The
physiology is unchanged; only what the device would record changes.  A concrete
draw is made for every beat (see ``sample_domain`` and ``observe``), so the
nuisance varies across the AV/VV sweep of a single patient.

The ``SOURCE_DOMAIN`` is used for training and in-domain test.  The
``SHIFTED_DOMAIN`` represents a plausible but different device/session and is
used as a stress test for sim-to-real robustness.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Tuple

import numpy as np

Range = Tuple[float, float]


@dataclass(frozen=True)
class SignalDomain:
    name: str

    imp_gain: Range = (0.016, 0.024)
    imp_offset: Range = (0.40, 0.50)
    imp_noise: Range = (0.01, 0.035)
    imp_drift: Range = (0.005, 0.020)
    imp_drift_freq: Range = (0.10, 0.35)


SOURCE_DOMAIN = SignalDomain(name="source")

SHIFTED_DOMAIN = SignalDomain(
    name="shifted",
    imp_gain=(0.010, 0.014),
    imp_offset=(0.30, 0.40),
    imp_noise=(0.05, 0.10),
    imp_drift=(0.030, 0.070),
    imp_drift_freq=(0.40, 0.80),
)


def sample_domain(domain: SignalDomain, rng: np.random.Generator) -> Dict[str, float]:
    """Draw one concrete nuisance-parameter set from a domain."""
    out: Dict[str, float] = {}
    for f in fields(domain):
        value = getattr(domain, f.name)
        if f.name == "name":
            continue
        if isinstance(value, tuple):
            out[f.name] = float(rng.uniform(value[0], value[1]))
        else:
            out[f.name] = float(value)
    return out
