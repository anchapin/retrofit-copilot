"""Tests for reading EnergyPlus CSV outputs into series."""

import numpy as np
import pytest

import results as RES


def _write_csv(path, header, rows):
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")


def test_read_energyplus_csv_parses_dates_and_values(tmp_path):
    csv_path = tmp_path / "eplus.csv"
    _write_csv(
        csv_path,
        ["Date/Time", "SPACE1-1:Zone Air Temperature [C](Hourly)"],
        [
            [" 01/01  01:00:00", 21.0],
            [" 01/01  02:00:00", 22.0],
            [" 12/31  24:00:00", 19.0],
        ],
    )
    months, hours, columns = RES.read_energyplus_csv(csv_path)
    assert list(months) == [1, 1, 12]
    assert list(hours) == [1, 2, 24]
    temp = columns["SPACE1-1:Zone Air Temperature [C](Hourly)"]
    assert temp.tolist() == [21.0, 22.0, 19.0]


def test_non_date_rows_are_skipped(tmp_path):
    csv_path = tmp_path / "eplus.csv"
    _write_csv(
        csv_path,
        ["Date/Time", "SPACE1-1:Zone Air Temperature [C](Hourly)"],
        [[" 01/01  01:00:00", 21.0], ["not a date", 0.0]],
    )
    months, _hours, _columns = RES.read_energyplus_csv(csv_path)
    assert list(months) == [1]


def test_facility_energy_sums_electricity_and_gas(tmp_path):
    csv_path = tmp_path / "eplus.csv"
    _write_csv(
        csv_path,
        ["Date/Time", "SPACE1-1:Zone Air Temperature [C](Hourly)"],
        [[" 01/01  01:00:00", 21.0]],
    )
    meter_path = tmp_path / "eplusMeter.csv"
    _write_csv(
        meter_path,
        ["Date/Time", "Electricity:Facility [J](Hourly)", "NaturalGas:Facility [J](Hourly)"],
        [[" 01/01  01:00:00", 3.6e6, 1.8e6]],
    )
    series = RES.load_run_series(tmp_path, "unit")
    energy = RES.facility_energy_kwh(series)
    assert energy.tolist() == pytest.approx([1.5])
    assert RES.electricity_kwh(series).tolist() == pytest.approx([1.0])


def test_missing_meter_raises(tmp_path):
    csv_path = tmp_path / "eplus.csv"
    _write_csv(
        csv_path,
        ["Date/Time", "SPACE1-1:Zone Air Temperature [C](Hourly)"],
        [[" 01/01  01:00:00", 21.0]],
    )
    series = RES.load_run_series(tmp_path, "unit")
    with pytest.raises(KeyError):
        RES.facility_energy_kwh(series)


def test_zone_mean_temperature_averages_space_zones(tmp_path):
    csv_path = tmp_path / "eplus.csv"
    _write_csv(
        csv_path,
        [
            "Date/Time",
            "SPACE1-1:Zone Air Temperature [C](Hourly)",
            "SPACE2-1:Zone Air Temperature [C](Hourly)",
            "PLENUM-1:Zone Air Temperature [C](Hourly)",
        ],
        [[" 01/01  01:00:00", 20.0, 22.0, 30.0]],
    )
    series = RES.load_run_series(tmp_path, "unit")
    # only the conditioned SPACE zones are averaged, the plenum is not
    assert RES.zone_mean_temperature(series).tolist() == pytest.approx([21.0])
