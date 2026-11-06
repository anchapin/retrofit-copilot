"""ASHRAE Guideline 14 calibration metrics and the bounded calibration search.

Metric definitions (ASHRAE Guideline 14, "Measurement of Energy and Demand
Savings"; the numeric thresholds applied by this POC are quoted in
``REPORT.md``):

    NMBE(%)   = SUM(y_i - yhat_i) / ((n - p) * mean(y)) * 100
    CV(RMSE)  = sqrt( SUM((y_i - yhat_i)^2) / (n - p) ) / mean(y) * 100

with ``y`` the measured series, ``yhat`` the modelled series and ``p`` the
number of fitted parameters.  This POC uses ``p = 0``: the comparison is a
straight time-series comparison of a simulated model against a measured
series, with no regression fitted to the measured data.

The calibration search is a *bounded coordinate search* over at most three
named parameters with fixed, recorded grids.  It is not a black-box
optimiser and it never sees the evaluation period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _as_arrays(measured: Sequence[float], modeled: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(measured, dtype=float)
    yhat = np.asarray(modeled, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: measured {y.shape} vs modeled {yhat.shape}")
    if y.size == 0:
        raise ValueError("empty series")
    return y, yhat


def nmbe(
    measured: Sequence[float], modeled: Sequence[float], p: int = 0
) -> float:
    """Normalised mean bias error [%]."""
    y, yhat = _as_arrays(measured, modeled)
    n = y.size
    denom = n - p
    if denom <= 0:
        raise ValueError("n - p must be positive")
    mean_y = y.mean()
    if mean_y == 0:
        raise ValueError("mean of measured series is zero; NMBE undefined")
    return float(np.sum(y - yhat) / (denom * mean_y) * 100.0)


def cvrmse(measured: Sequence[float], modeled: Sequence[float], p: int = 0) -> float:
    """Coefficient of variation of the root mean square error [%]."""
    y, yhat = _as_arrays(measured, modeled)
    n = y.size
    denom = n - p
    if denom <= 0:
        raise ValueError("n - p must be positive")
    mean_y = y.mean()
    if mean_y == 0:
        raise ValueError("mean of measured series is zero; CV(RMSE) undefined")
    mse = np.sum((y - yhat) ** 2) / denom
    return float(np.sqrt(mse) / mean_y * 100.0)


def rmse(measured: Sequence[float], modeled: Sequence[float], p: int = 0) -> float:
    y, yhat = _as_arrays(measured, modeled)
    return float(np.sqrt(np.sum((y - yhat) ** 2) / (y.size - p)))


def monthly_nmbe(
    measured: Sequence[float],
    modeled: Sequence[float],
    months: Sequence[int],
    p: int = 0,
) -> Dict[int, float]:
    """NMBE [%] per calendar month (months are 1..12)."""
    y, yhat = _as_arrays(measured, modeled)
    if len(months) != y.size:
        raise ValueError("months length must match the series length")
    out: Dict[int, float] = {}
    for m in range(1, 13):
        idx = [i for i, mm in enumerate(months) if mm == m]
        if not idx:
            continue
        out[m] = nmbe(y[idx], yhat[idx], p=p)
    return out


def monthly_cvrmse(
    measured: Sequence[float],
    modeled: Sequence[float],
    months: Sequence[int],
    p: int = 0,
) -> Dict[int, float]:
    y, yhat = _as_arrays(measured, modeled)
    if len(months) != y.size:
        raise ValueError("months length must match the series length")
    out: Dict[int, float] = {}
    for m in range(1, 13):
        idx = [i for i, mm in enumerate(months) if mm == m]
        if not idx:
            continue
        out[m] = cvrmse(y[idx], yhat[idx], p=p)
    return out


def metrics(
    measured: Sequence[float], modeled: Sequence[float], p: int = 0
) -> Dict[str, float]:
    """All headline metrics for one series pair."""
    return {
        "n": int(np.asarray(measured).size),
        "nmbe_pct": nmbe(measured, modeled, p=p),
        "cvrmse_pct": cvrmse(measured, modeled, p=p),
        "rmse": rmse(measured, modeled, p=p),
        "mean_measured": float(np.mean(measured)),
        "mean_modeled": float(np.mean(modeled)),
    }


# ---------------------------------------------------------------------------
# bounded calibration search
# ---------------------------------------------------------------------------


@dataclass
class SearchStep:
    sweep: int
    parameter: str
    value: float
    objective: float
    run_id: str
    accepted: bool


@dataclass
class SearchResult:
    best_params: Dict[str, float]
    best_objective: float
    steps: List[SearchStep] = field(default_factory=list)
    evaluated: List[Dict[str, Any]] = field(default_factory=list)


def coordinate_search(
    grids: Dict[str, Sequence[float]],
    start: Dict[str, float],
    simulate: Callable[[Dict[str, float]], Tuple[str, Sequence[float]]],
    objective: Callable[[Sequence[float]], float],
    *,
    max_sweeps: int = 3,
    parameter_order: Optional[Sequence[str]] = None,
) -> SearchResult:
    """Bounded coordinate (grid) search over named parameters.

    ``simulate(params)`` must return ``(run_id, modelled_series)``; every call
    is a fresh, recorded simulation.  ``objective(series)`` must return a
    scalar to minimise (lower is better).  A candidate is accepted only on a
    *strict* improvement, so the search never drifts away from the best point
    found so far and the procedure is reproducible.
    """
    order = list(parameter_order or sorted(grids))
    unknown = set(order) - set(grids)
    if unknown:
        raise ValueError(f"no grid given for parameters {sorted(unknown)}")
    unknown_start = set(start) - set(grids)
    if unknown_start:
        raise ValueError(f"no grid given for parameters {sorted(unknown_start)}")

    current: Dict[str, float] = {k: float(start[k]) for k in order}
    best_obj = float("inf")
    steps: List[SearchStep] = []
    evaluated: List[Dict[str, Any]] = []

    for sweep in range(1, max_sweeps + 1):
        improved = False
        for param in order:
            local_best_value = current[param]
            local_best_obj = best_obj
            for value in grids[param]:
                candidate = dict(current)
                candidate[param] = float(value)
                run_id, series = simulate(candidate)
                obj = float(objective(series))
                evaluated.append(
                    {
                        "sweep": sweep,
                        "parameter": param,
                        "value": float(value),
                        "objective": obj,
                        "run_id": run_id,
                    }
                )
                accepted = obj < local_best_obj
                steps.append(
                    SearchStep(
                        sweep=sweep,
                        parameter=param,
                        value=float(value),
                        objective=obj,
                        run_id=run_id,
                        accepted=accepted,
                    )
                )
                if accepted:
                    local_best_obj = obj
                    local_best_value = float(value)
            if local_best_obj < best_obj:
                best_obj = local_best_obj
                current[param] = local_best_value
                improved = True
        if not improved:
            break

    return SearchResult(
        best_params=current, best_objective=best_obj, steps=steps, evaluated=evaluated
    )
