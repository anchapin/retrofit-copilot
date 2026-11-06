"""Read EnergyPlus run outputs into plain numpy series.

Only two artefacts of a run are used: ``eplus.csv`` (hourly output variables)
and ``eplusMeter.csv`` (hourly meters).  Both are produced by EnergyPlus'
own ReadVarsESO post-processor, which is invoked by the ``-r`` flag of the
``energyplus`` command line tool.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_DATE_RE = re.compile(r"^\s*(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\s*$")

JOULES_PER_KWH = 3.6e6


@dataclass
class RunSeries:
    """Hourly series extracted from one EnergyPlus run."""

    run_id: str
    months: np.ndarray
    hours: np.ndarray
    columns: Dict[str, np.ndarray]

    def column(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(
                f"column {name!r} not found; available: {sorted(self.columns)}"
            )
        return self.columns[name]

    def has(self, name: str) -> bool:
        return name in self.columns


def _parse_datetime(token: str) -> Optional[Tuple[int, int, int]]:
    match = _DATE_RE.match(token)
    if not match:
        return None
    month, day, hour, _minute, _second = (int(g) for g in match.groups())
    return month, day, hour


def read_energyplus_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Read an EnergyPlus ReadVarsESO CSV into (months, hours, columns)."""
    months: List[int] = []
    hours: List[int] = []
    columns: Dict[str, List[float]] = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        header = [h.strip() for h in header]
        names = header[1:]
        for name in names:
            columns[name] = []
        for row in reader:
            if not row:
                continue
            parsed = _parse_datetime(row[0])
            if parsed is None:
                # design-day / warmup rows are skipped; they have no date
                continue
            month, _day, hour = parsed
            months.append(month)
            hours.append(hour)
            for i, name in enumerate(names):
                raw = row[i + 1] if i + 1 < len(row) else ""
                try:
                    value = float(raw)
                except ValueError:
                    value = float("nan")
                columns[name].append(value)
    return (
        np.asarray(months, dtype=int),
        np.asarray(hours, dtype=int),
        {k: np.asarray(v, dtype=float) for k, v in columns.items()},
    )


def load_run_series(run_dir: Path, run_id: Optional[str] = None) -> RunSeries:
    """Load the hourly series of a recorded run directory."""
    run_dir = Path(run_dir)
    months, hours, columns = read_energyplus_csv(run_dir / "eplus.csv")
    meter_file = run_dir / "eplusMeter.csv"
    if meter_file.is_file():
        m_months, m_hours, m_columns = read_energyplus_csv(meter_file)
        if len(m_months) == len(months) and np.array_equal(m_months, months):
            columns.update(m_columns)
    return RunSeries(
        run_id=run_id or run_dir.name,
        months=months,
        hours=hours,
        columns=columns,
    )


def facility_energy_kwh(series: RunSeries) -> np.ndarray:
    """Total hourly site energy [kWh] = electricity + natural gas fuel."""
    total = np.zeros(len(series.months), dtype=float)
    found = False
    for name in ("Electricity:Facility [J](Hourly)",):
        if series.has(name):
            total = total + series.column(name)
            found = True
    for name in (
        "NaturalGas:Facility [J](Hourly)",
        "DistrictHeating:Facility [J](Hourly)",
        "DistrictCooling:Facility [J](Hourly)",
    ):
        if series.has(name):
            total = total + series.column(name)
            found = True
    if not found:
        raise KeyError(
            "no hourly facility energy meter found in the run outputs; "
            "requested meters are missing"
        )
    return total / JOULES_PER_KWH


def electricity_kwh(series: RunSeries) -> np.ndarray:
    name = "Electricity:Facility [J](Hourly)"
    if not series.has(name):
        raise KeyError(f"{name!r} missing from run outputs")
    return series.column(name) / JOULES_PER_KWH


def zone_mean_temperature(series: RunSeries) -> np.ndarray:
    """Mean air temperature over the five conditioned SPACE zones [degC]."""
    cols = [
        name
        for name in series.columns
        if ":Zone Air Temperature [C](Hourly)" in name and "SPACE" in name
    ]
    if not cols:
        raise KeyError("no zone air temperature columns found in run outputs")
    stacked = np.vstack([series.column(name) for name in sorted(cols)])
    return stacked.mean(axis=0)
