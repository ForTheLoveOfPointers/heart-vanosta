from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from phase0.evaluate import metrics_from_scores, three_way_split
from phase0.features import extract_features
from phase0.models.pairwise import build_pairs
from phase0.types import BeatSignals, PatientParams, Setting


def _patient(**kwargs):
    base = dict(
        patient_id=0,
        hr_bpm=70.0,
        contractility=1.0,
        lv_viability=0.9,
        septal_viability=0.85,
        lateral_viability=0.95,
        rv_viability=1.0,
        preload=1.0,
        afterload=1.0,
        intrinsic_av_ms=160.0,
        qrs_ms=150.0,
        dyssynchrony_ms=60.0,
        lead_lv_delay_ms=35.0,
        lead_rv_delay_ms=10.0,
    )
    base.update(kwargs)
    return PatientParams(**base)


def test_beat_signals_is_impedance_only():
    names = {f.name for f in fields(BeatSignals)}
    assert names == {"t_ms", "impedance"}


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics_from_scores(
            np.array([1.0, 2.0]),
            np.array([0, 0]),
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
        )


def test_three_way_split_is_disjoint():
    groups = np.repeat(np.arange(20), 3)
    tr, va, te = three_way_split(groups, val_frac=0.2, test_frac=0.2, seed=0)
    assert not np.any(tr & va)
    assert not np.any(tr & te)
    assert not np.any(va & te)
    assert len(np.unique(groups[tr])) == 12
    assert len(np.unique(groups[va])) == 4
    assert len(np.unique(groups[te])) == 4


def test_pairs_are_within_groups():
    y = np.array([3.0, 1.0, 2.0, 5.0, 4.0, 6.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    ii, jj = build_pairs(y, groups, margin_frac=0.0, max_pairs_per_group=100)
    assert len(ii) > 0
    assert np.all(groups[ii] == groups[jj])


def test_impedance_ejection_time_uses_circular_distance():
    t = np.arange(0.0, 101.0)
    imp = np.zeros_like(t)
    imp[10] = 1.0
    imp[90] = -1.0
    signals = BeatSignals(t_ms=t, impedance=imp)
    feats = extract_features(signals, _patient(), Setting(av_ms=140, vv_ms=0))
    assert feats["imp_ejection_time"] == pytest.approx(20.0)


def test_feature_set_is_impedance_only():
    t = np.arange(0.0, 700.0)
    imp = 0.45 + 0.02 * (60.0 + 30.0 * np.sin(2 * np.pi * t / 700.0))
    feats = extract_features(
        BeatSignals(t_ms=t, impedance=imp), _patient(), Setting(av_ms=160, vv_ms=0)
    )
    assert len(feats) == 42


def test_benchmark_seed_changes_sensor_draws(monkeypatch):
    from phase0.benchmark import _simulate_patient
    from phase0.types import Beat, BeatLabels

    class FakeSim:
        def simulate(self, patient, setting):
            t = np.array([0.0, 1.0, 2.0])
            z = np.zeros_like(t)
            return Beat(
                t_ms=t,
                lv_pressure=z.copy(),
                lv_volume=z.copy() + 100.0,
                aortic_pressure=z.copy(),
                activation_times={},
                labels=BeatLabels(
                    stroke_work=1.0,
                    edv=100.0,
                    esv=50.0,
                    ef=0.5,
                    dpdt_max=1.0,
                    work_asymmetry=0.0,
                    wasted_work=0.0,
                    filling_time_ms=1.0,
                    fusion_index=1.0,
                ),
            )

    def fake_observe(beat, patient, rng, domain=None):
        value = rng.random()
        return BeatSignals(t_ms=beat.t_ms, impedance=np.zeros_like(beat.t_ms) + value)

    monkeypatch.setattr("phase0.benchmark.observe", fake_observe)
    cfg = {
        "simulation": {
            "av_grid_ms": [120],
            "vv_grid_ms": [0],
        }
    }
    rows1 = _simulate_patient(cfg, FakeSim(), _patient(), None, 8, seed=1)
    rows2 = _simulate_patient(cfg, FakeSim(), _patient(), None, 8, seed=2)
    assert not np.allclose(rows1[0]["raw"], rows2[0]["raw"])
