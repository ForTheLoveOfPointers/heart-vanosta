from __future__ import annotations

import numpy as np

from phase0.sensors import SHIFTED_DOMAIN, SOURCE_DOMAIN, sample_domain


def test_sample_domain_is_seeded():
    a = sample_domain(SOURCE_DOMAIN, np.random.default_rng(123))
    b = sample_domain(SOURCE_DOMAIN, np.random.default_rng(123))
    c = sample_domain(SOURCE_DOMAIN, np.random.default_rng(124))
    assert a == b
    assert a != c
    assert set(a) == {
        f.name for f in SOURCE_DOMAIN.__dataclass_fields__.values()
    } - {"name"}


def test_shifted_impedance_domain_is_different():
    rng = np.random.default_rng(0)
    source = sample_domain(SOURCE_DOMAIN, rng)
    shifted = sample_domain(SHIFTED_DOMAIN, rng)
    for key in ("imp_gain", "imp_offset", "imp_noise", "imp_drift", "imp_drift_freq"):
        assert shifted[key] != source[key]
    assert shifted["imp_noise"] > source["imp_noise"]
    assert shifted["imp_gain"] < source["imp_gain"]
