"""Tests for the LambdaMART (LightGBM) ranking model."""

from __future__ import annotations

import numpy as np
import pytest

from phase0.evaluate import metrics_from_scores
from phase0.models.gbdt import LambdaMARTRanker, group_rank_grades

lightgbm = pytest.importorskip("lightgbm")


def _synthetic(n_patients=30, n_settings=8, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups, av, vv = [], [], [], [], []
    for g in range(n_patients):
        opt = rng.integers(0, n_settings)
        s = rng.uniform(0.4, 1.0)
        scores = s * (
            1.0 - 0.6 * (np.arange(n_settings) - opt) ** 2 / n_settings**2
        )
        noise = rng.normal(0, 0.02, n_settings)
        y.extend(scores + noise)
        # informative feature: slightly noisy copy of the label
        X.extend((scores + rng.normal(0, 0.02, n_settings)).tolist())
        groups.extend([g] * n_settings)
        av.extend(np.linspace(80, 300, n_settings))
        vv.extend([0.0] * n_settings)
    return (
        np.array(X).reshape(-1, 1),
        np.array(y),
        np.array(groups),
        np.array(av),
        np.array(vv),
    )


def test_group_rank_grades_span_and_order():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    groups = np.zeros(8, dtype=int)
    grades = group_rank_grades(y, groups, n_grades=4)
    assert grades.min() == 0 and grades.max() == 3
    assert np.all(np.diff(grades) >= 0)
    assert grades[0] == 0 and grades[-1] == 3


def test_group_rank_grades_are_within_patient():
    y = np.array([1.0, 2.0, 100.0, 200.0])
    groups = np.array([0, 0, 1, 1])
    grades = group_rank_grades(y, groups, n_grades=4)
    # patient 1 has much larger labels but the same grade range
    assert grades[0] == grades[2] and grades[1] == grades[3]


def test_lambdamart_learns_informative_feature():
    X, y, groups, av, vv = _synthetic()
    model = LambdaMARTRanker(n_estimators=200, early_stopping_rounds=20).fit(
        X, y, groups
    )
    scores = model.score(X)
    met = metrics_from_scores(
        y, groups, scores, av, vv, top_k=3, regret_threshold=0.02
    )
    # the flat quadratic optimum makes the exact argmax genuinely ambiguous;
    # the meaningful check is that LambdaMART clearly beats the linear ranker
    # on the same data (and far above chance).
    from phase0.models.pairwise import PairwiseRanker

    linear = PairwiseRanker().fit(X, y, groups)
    lin_met = metrics_from_scores(
        y, groups, linear.score(X), av, vv, top_k=3, regret_threshold=0.02
    )
    assert met["near_optimal_fraction"] > 0.6
    assert met["mean_spearman"] > 0.6
    assert (
        met["near_optimal_fraction"] > lin_met["near_optimal_fraction"] + 0.15
    )


def test_lambdamart_rejects_nonfinite():
    X = np.array([[np.nan, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 2.0])
    groups = np.array([0, 0])
    with pytest.raises(ValueError):
        LambdaMARTRanker().fit(X, y, groups)


def test_fit_lambda_mart_cv_selects_rounds_and_learns():
    from phase0.models.gbdt import fit_lambda_mart_cv

    X, y, groups, av, vv = _synthetic(n_patients=24)
    model, diag = fit_lambda_mart_cv(
        X, y, groups, n_folds=3, num_boost_round=60, seed=0
    )
    assert 1 <= diag["cv_best_round"] <= 60
    assert len(diag["cv_curve"]) == 60
    assert model.best_iteration == diag["cv_best_round"]
    met = metrics_from_scores(
        y, groups, model.score(X), av, vv, top_k=3, regret_threshold=0.02
    )
    assert met["near_optimal_fraction"] > 0.5
    assert met["mean_spearman"] > 0.5


def test_rank_within_patient_is_bounded():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_gbt_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_gbt_benchmark", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scores = np.array([1.0, 5.0, 3.0, 2.0, 9.0])
    groups = np.array([0, 0, 0, 1, 1])
    out = mod._rank_within_patient(scores, groups)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[0] == 0.0 and out[1] == 1.0
    assert np.allclose(np.sort(out[:3]), [0.0, 0.5, 1.0])
