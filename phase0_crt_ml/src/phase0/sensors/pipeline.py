"""Intracardiac impedance observation model."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import Beat, BeatSignals, PatientParams
from .domain import SignalDomain, sample_domain
from .impedance import synth_impedance

CHANNEL_ORDER = ("impedance",)


def observe(
    beat: Beat,
    patient: PatientParams,
    rng: np.random.Generator,
    domain: Optional[SignalDomain] = None,
) -> BeatSignals:
    """Generate the device impedance recording for one beat.

    Only physiology is used; ``beat.labels`` is never read here.  When
    ``domain`` is given, one nuisance-parameter set is sampled for this beat.
    """
    params = None if domain is None else sample_domain(domain, rng)
    impedance = synth_impedance(
        beat.t_ms, beat.lv_volume, patient, rng, params=params
    )
    return BeatSignals(t_ms=beat.t_ms, impedance=impedance)


def resample_signals(signals: BeatSignals, length: int = 512) -> np.ndarray:
    """Resample one impedance beat to a fixed-length single-channel array.

    Returns an array of shape ``(1, length)``.  Linear interpolation over the
    cardiac cycle keeps timing relationships while making the input independent
    of heart rate.
    """
    t = np.asarray(signals.t_ms, dtype=float)
    grid = np.linspace(float(t[0]), float(t[-1]), int(length))
    impedance = np.interp(grid, t, np.asarray(signals.impedance, dtype=float))
    return np.stack([impedance]).astype(np.float32)
