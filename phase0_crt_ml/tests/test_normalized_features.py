from __future__ import annotations

import numpy as np

from phase0.features import (
    extract_features,
    normalize_impedance,
    patient_normalize,
)
from phase0.types import BeatSignals, PatientParams, Setting


def _signals() -> BeatSignals:
    t = np.arange(0.0, 700.0, 1.0)
    rng = np.random.default_rng(0)
    imp = 0.45 + 0.02 * (60.0 + 30.0 * np.sin(2 * np.pi * t / 700.0))
    return BeatSignals(
        t_ms=t,
        impedance=imp + rng.normal(0.0, 1e-4, t.shape),
    )


def test_normalized_impedance_features_are_affine_invariant():
    patient = PatientParams(patient_id=0)
    setting = Setting(av_ms=160, vv_ms=0)
    sig = _signals()
    feats = extract_features(sig, patient, setting, mode="normalized")

    scaled = BeatSignals(
        t_ms=sig.t_ms,
        impedance=2.0 * sig.impedance + 0.5,
    )
    scaled_feats = extract_features(scaled, patient, setting, mode="normalized")
    for key in feats:
        assert np.isclose(feats[key], scaled_feats[key], rtol=1e-4, atol=1e-4), key


def test_normalized_impedance_differs_from_raw_when_gain_changes():
    patient = PatientParams(patient_id=0)
    setting = Setting(av_ms=160, vv_ms=0)
    sig = _signals()
    raw = extract_features(sig, patient, setting, mode="raw")
    scaled = BeatSignals(t_ms=sig.t_ms, impedance=2.0 * sig.impedance)
    raw_scaled = extract_features(scaled, patient, setting, mode="raw")
    assert not np.isclose(raw["imp_stroke"], raw_scaled["imp_stroke"])


def test_normalize_impedance_preserves_time_and_shape():
    sig = _signals()
    norm = normalize_impedance(sig)
    assert norm.t_ms is sig.t_ms
    assert norm.impedance.shape == sig.impedance.shape


def test_patient_normalize_is_within_patient_and_ignores_context():
    names = ["imp_stroke", "imp_std", "ctx_hr", "set_av"]
    X = np.array(
        [
            [1.0, 10.0, 70.0, 80.0],
            [2.0, 20.0, 70.0, 140.0],
            [3.0, 30.0, 70.0, 200.0],
            [10.0, 1.0, 80.0, 80.0],
            [20.0, 2.0, 80.0, 140.0],
            [30.0, 3.0, 80.0, 200.0],
        ],
        dtype=float,
    )
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = patient_normalize(X, groups, names, prefixes=("imp_",))
    for g in (0, 1):
        m = groups == g
        block = out[np.ix_(m, [0, 1])]
        assert np.allclose(block.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(block.std(axis=0), 1.0, atol=1e-8)
    assert np.array_equal(out[:, 2:], X[:, 2:])


def test_patient_normalize_recomputes_impedance_interactions():
    names = ["imp_stroke", "set_av", "ctx_qrs_est", "ix_set_av_x_imp_stroke"]
    X = np.array(
        [
            [1.0, 80.0, 130.0, 80.0],
            [2.0, 140.0, 130.0, 280.0],
            [3.0, 200.0, 130.0, 600.0],
            [10.0, 80.0, 130.0, 800.0],
            [20.0, 140.0, 130.0, 2800.0],
            [30.0, 200.0, 130.0, 6000.0],
        ],
        dtype=float,
    )
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = patient_normalize(X, groups, names, prefixes=("imp_",))
    for g in (0, 1):
        m = groups == g
        expected = out[np.ix_(m, [1])] * out[np.ix_(m, [0])]
        assert np.allclose(out[np.ix_(m, [3])], expected)
    assert np.array_equal(out[:, 1:3], X[:, 1:3])


def test_patient_normalize_recomputes_interactions_with_composite_operands():
    """`set_av_x_vv` is itself a base feature, so it must win the longest match."""
    t = np.arange(0.0, 700.0)
    rows = []
    names = None
    for group in range(2):
        for k, av in enumerate([80, 140, 200]):
            imp = (0.4 + 0.05 * group) + (0.02 + 0.003 * group) * (
                60.0 + 30.0 * np.sin(2 * np.pi * t / 700.0 + k)
            )
            feats = extract_features(
                BeatSignals(t_ms=t, impedance=imp),
                PatientParams(patient_id=group),
                Setting(av_ms=av, vv_ms=0),
            )
            names = sorted(feats)
            rows.append(feats)
    X = np.array([[row[n] for n in names] for row in rows], dtype=float)
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = patient_normalize(X, groups, names, prefixes=("imp_",))
    idx = {name: i for i, name in enumerate(names)}
    for target, left, right in (
        ("ix_set_av_x_vv_x_imp_stroke", "set_av_x_vv", "imp_stroke"),
        ("ix_set_av_x_vv_x_imp_ejection_time", "set_av_x_vv", "imp_ejection_time"),
        ("ix_set_av_x_vv_x_ctx_hr", "set_av_x_vv", "ctx_hr"),
    ):
        expected = out[:, idx[left]] * out[:, idx[right]]
        assert np.allclose(out[:, idx[target]], expected), target
