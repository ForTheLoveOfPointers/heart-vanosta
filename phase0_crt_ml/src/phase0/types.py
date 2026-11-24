"""Shared data structures for the Phase 0 CRT pipeline.

The central design rule is that the *simulator* produces physiology and
ground-truth labels, while the *sensors* module produces what a device would
actually record.  The observation model must never receive the labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PatientParams:
    """Virtual patient parameters.

    Units: times in ms, volumes in mL, pressures in mmHg.
    """

    patient_id: int
    hr_bpm: float = 70.0
    contractility: float = 1.0
    lv_viability: float = 0.90
    septal_viability: float = 0.85
    lateral_viability: float = 0.95
    rv_viability: float = 1.00
    preload: float = 1.00
    afterload: float = 1.00
    intrinsic_av_ms: float = 160.0
    qrs_ms: float = 150.0
    dyssynchrony_ms: float = 60.0
    lead_lv_delay_ms: float = 35.0
    lead_rv_delay_ms: float = 10.0
    av_block: bool = False
    biventricular_pacing: bool = True
    seed: int = 0


@dataclass
class Setting:
    """A candidate device programming.

    vv_ms is relative to the patient's intrinsic LV-RV delay, not an absolute
    LV-before-RV offset.  Positive values pre-activate LV relative to that
    intrinsic delay; because the intrinsic delay can be positive, LV may still
    activate after RV in absolute terms.
    """

    av_ms: float
    vv_ms: float = 0.0

    def label(self) -> str:
        return f"AV{self.av_ms:.0f}_VV{self.vv_ms:+.0f}"


@dataclass
class BeatLabels:
    # Stroke work is reported in mmHg*mL (not joules); ranking is scale-invariant.
    stroke_work: float
    edv: float
    esv: float
    ef: float
    dpdt_max: float
    work_asymmetry: float
    wasted_work: float
    filling_time_ms: float
    fusion_index: float


@dataclass
class Beat:
    """Physiology returned by a simulator."""

    t_ms: np.ndarray
    lv_pressure: np.ndarray
    lv_volume: np.ndarray
    aortic_pressure: np.ndarray
    activation_times: Dict[str, float]
    labels: BeatLabels


@dataclass
class BeatSignals:
    """Device impedance recording for one beat.

    The project deliberately models a single sensor: intracardiac impedance.
    """

    t_ms: np.ndarray
    impedance: np.ndarray
