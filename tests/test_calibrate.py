"""Known-value tests for the ASHRAE Guideline 14 metrics."""

import math

import numpy as np
import pytest

import calibrate as C


def test_nmbe_zero_for_perfect_model():
    y = [10.0, 12.0, 14.0, 16.0]
    assert C.nmbe(y, y) == pytest.approx(0.0, abs=1e-12)
    assert C.cvrmse(y, y) == pytest.approx(0.0, abs=1e-12)


def test_nmbe_and_cvrmse_for_constant_ten_percent_bias():
    # measured is constant 10, model is constant 9 -> 10 % low everywhere
    measured = [10.0] * 8
    modeled = [9.0] * 8
    assert C.nmbe(measured, modeled) == pytest.approx(10.0)
    assert C.cvrmse(measured, modeled) == pytest.approx(10.0)


def test_nmbe_sign_convention():
    # model above measured -> negative NMBE (model over-predicts)
    measured = [100.0] * 5
    modeled = [110.0] * 5
    assert C.nmbe(measured, modeled) == pytest.approx(-10.0)
    assert C.cvrmse(measured, modeled) == pytest.approx(10.0)


def test_cvrmse_known_value_with_spread():
    measured = [10.0, 20.0, 30.0, 40.0]
    modeled = [11.0, 19.0, 31.0, 39.0]
    # residuals: -1, +1, -1, +1 -> sum of squares = 4, n = 4
    expected_cv = math.sqrt(4.0 / 4.0) / 25.0 * 100.0
    expected_nmbe = 0.0
    assert C.cvrmse(measured, modeled) == pytest.approx(expected_cv)
    assert C.nmbe(measured, modeled) == pytest.approx(expected_nmbe)
    assert C.rmse(measured, modeled) == pytest.approx(1.0)


def test_cvrmse_uses_n_minus_p():
    measured = [10.0, 20.0, 30.0, 40.0]
    modeled = [11.0, 19.0, 31.0, 39.0]
    # p = 1 -> denominator 3 instead of 4
    expected = math.sqrt(4.0 / 3.0) / 25.0 * 100.0
    assert C.cvrmse(measured, modeled, p=1) == pytest.approx(expected)


def test_metrics_block_shape():
    measured = np.array([1.0, 2.0, 3.0, 4.0])
    modeled = np.array([1.0, 2.0, 3.0, 4.0])
    out = C.metrics(measured, modeled)
    assert out["n"] == 4
    assert out["mean_measured"] == pytest.approx(2.5)
    assert out["mean_modeled"] == pytest.approx(2.5)
    assert out["nmbe_pct"] == pytest.approx(0.0)
    assert out["cvrmse_pct"] == pytest.approx(0.0)


def test_monthly_nmbe_splits_by_month():
    measured = [10.0, 10.0, 20.0, 20.0]
    modeled = [9.0, 9.0, 19.0, 19.0]
    months = [1, 1, 7, 7]
    out = C.monthly_nmbe(measured, modeled, months)
    assert set(out) == {1, 7}
    assert out[1] == pytest.approx(10.0)
    assert out[7] == pytest.approx(5.0)


def test_monthly_nmbe_length_mismatch_raises():
    with pytest.raises(ValueError):
        C.monthly_nmbe([1.0, 2.0], [1.0, 2.0], [1])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        C.nmbe([1.0, 2.0], [1.0])


def test_zero_mean_raises():
    with pytest.raises(ValueError):
        C.nmbe([0.0, 0.0], [0.0, 0.0])


# --------------------------------------------------------------------------
# the bounded coordinate search
# --------------------------------------------------------------------------
def test_coordinate_search_finds_the_minimum_with_a_fake_simulator():
    """A quadratic fake model: the search must recover the true parameter."""

    def simulate(params):
        a = params["a"]
        y = (a - 3.0) ** 2
        return f"run-{a}", [y]

    def objective(series):
        return float(series[0])

    grids = {"a": [0.0, 1.0, 2.0, 3.0, 4.0]}
    result = C.coordinate_search(
        grids, start={"a": 0.0}, simulate=simulate, objective=objective, max_sweeps=2
    )
    assert result.best_params["a"] == pytest.approx(3.0)
    assert result.best_objective == pytest.approx(0.0)
    assert all(step.run_id.startswith("run-") for step in result.steps)


def test_coordinate_search_rejects_invalid_candidates():
    """A failed simulation (infinite objective) is never accepted."""

    def simulate(params):
        if params["a"] > 1.5:
            return "bad-run", None
        return f"run-{params['a']}", [abs(params["a"] - 1.0)]

    def objective(series):
        if series is None:
            return float("inf")
        return float(series[0])

    grids = {"a": [0.0, 1.0, 2.0, 3.0]}
    result = C.coordinate_search(
        grids, start={"a": 0.0}, simulate=simulate, objective=objective, max_sweeps=1
    )
    assert result.best_params["a"] == pytest.approx(1.0)
    assert math.isfinite(result.best_objective)


def test_coordinate_search_requires_a_grid_for_every_parameter():
    with pytest.raises(ValueError):
        C.coordinate_search(
            {"a": [1.0]},
            start={"a": 1.0, "b": 1.0},
            simulate=lambda p: ("r", [0.0]),
            objective=lambda s: float(s[0]),
        )
