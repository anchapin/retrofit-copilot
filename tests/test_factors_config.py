"""Tests for the config-file energy-price / emission factors (issue #3).

The four factors (electricity price, gas price, electricity emission factor,
gas emission factor) are loaded from a JSON factors file instead of being
hard-coded: the shipped default is ``config/factors.placeholder.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import ecms as E
import results as RES
import run_ecms as RE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _series(run_id: str, electricity_kwh: float, gas_kwh: float, temp: float = 22.0) -> RES.RunSeries:
    hours = 8760
    months = np.array([((i // 730) % 12) + 1 for i in range(hours)])
    hours_of_day = np.array([i % 24 for i in range(hours)])
    return RES.RunSeries(
        run_id=run_id,
        months=months,
        hours=hours_of_day,
        columns={
            "Electricity:Facility [J](Hourly)": np.full(hours, electricity_kwh * 3.6e6 / hours),
            "NaturalGas:Facility [J](Hourly)": np.full(hours, gas_kwh * 3.6e6 / hours),
            "SPACE1-1:Zone Air Temperature [C](Hourly)": np.full(hours, temp),
            "SPACE1-1:Zone Air System Sensible Heating Rate [W](Hourly)": np.zeros(hours),
            "SPACE1-1:Zone Air System Sensible Cooling Rate [W](Hourly)": np.zeros(hours),
        },
    )


def _metrics(factors=None):
    """Metrics of a known ECM: 10 000 kWh electricity and 1 000 kWh gas saved."""
    baseline = _series("base", 40_000.0, 10_000.0)
    ecm = _series("ecm", 30_000.0, 9_000.0)
    return RE.ecm_metrics(baseline, ecm, placeholder_cost_usd=10_000.0, factors=factors)


def _fake_metrics(cost_usd: float, co2_tco2: float) -> dict:
    """One synthetic model of metrics, shaped like the real pipeline output."""
    out = {}
    for model_name in ("calibrated", "true"):
        per_ecm = {}
        for ecm_id in E.ECM_IDS:
            per_ecm[ecm_id] = {
                "energy_saved_kwh": 4790.3,
                "energy_saved_pct": 7.58,
                "cost_saved_usd": cost_usd,
                "co2_saved_tco2": co2_tco2,
                "simple_payback_years": 127.7,
                "run_id": f"run-{ecm_id}-{model_name}",
                "comfort": {
                    "ecm": {"zone_mean_air_temp_c": 21.54, "zone_hours_outside_comfort_band": 12169},
                    "delta_zone_mean_air_temp_c": 0.11,
                },
            }
        out[model_name] = per_ecm
    return out


def _table(factors) -> str:
    metrics = _fake_metrics(346.54, 1.023)
    lines = RE.headline_table_lines(
        version="EnergyPlus, Version 26.2.0",
        agent_ranking=list(E.ECM_IDS),
        truth_ranking=list(E.ECM_IDS),
        prior_ranking=list(E.ECM_IDS),
        model_metrics=metrics,
        runs={f"{ecm}_{model}": f"run-{ecm}-{model}" for model in ("calibrated", "true") for ecm in E.ECM_IDS},
        factors=factors,
    )
    return "\n".join(lines)


def _detail_header(table: str) -> str:
    for line in table.splitlines():
        if line.startswith("| ECM | model |"):
            return line
    raise AssertionError("no detail table header found")


# ---------------------------------------------------------------------------
# the shipped placeholder factors file
# ---------------------------------------------------------------------------
def test_the_shipped_placeholder_factors_file_has_the_expected_schema():
    path = E.PLACEHOLDER_FACTORS_PATH
    assert path.is_file()
    data = json.loads(path.read_text())
    assert set(data) == set(E.FACTOR_KEYS)
    assert data == {
        "electricity_price_usd_per_kwh": 0.13,
        "natural_gas_price_usd_per_kwh": 0.06,
        "electricity_co2_kg_per_kwh": 0.37,
        "natural_gas_co2_kg_per_kwh": 0.18,
    }


def test_the_factor_constants_come_from_the_config_file():
    """The values are not hard-coded in src/ecms.py any more."""
    data = json.loads(E.PLACEHOLDER_FACTORS_PATH.read_text())
    assert E.ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER == data["electricity_price_usd_per_kwh"]
    assert E.NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER == data["natural_gas_price_usd_per_kwh"]
    assert E.ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER == data["electricity_co2_kg_per_kwh"]
    assert E.NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER == data["natural_gas_co2_kg_per_kwh"]
    # the PLACEHOLDER marker check still passes
    E.check_placeholder_labels()


def test_the_default_factors_are_the_shipped_placeholder_file():
    factors = E.default_factors()
    assert factors.source == str(E.PLACEHOLDER_FACTORS_PATH)
    assert factors.is_placeholder
    assert factors.label == E.PLACEHOLDER
    assert factors.to_dict() == {
        "electricity_price_usd_per_kwh": 0.13,
        "natural_gas_price_usd_per_kwh": 0.06,
        "electricity_co2_kg_per_kwh": 0.37,
        "natural_gas_co2_kg_per_kwh": 0.18,
    }


def test_loading_the_shipped_file_yields_the_placeholder_factors():
    factors = E.load_factors(E.PLACEHOLDER_FACTORS_PATH)
    assert factors.is_placeholder
    assert factors.to_dict() == E.default_factors().to_dict()


# ---------------------------------------------------------------------------
# validation of a factors file
# ---------------------------------------------------------------------------
def test_a_missing_key_is_rejected_with_a_clear_error(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(json.dumps({"electricity_price_usd_per_kwh": 0.2}))
    with pytest.raises(ValueError) as excinfo:
        E.load_factors(path)
    message = str(excinfo.value)
    assert "missing the required key" in message
    assert "natural_gas_price_usd_per_kwh" in message


def test_a_non_numeric_factor_is_rejected(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(json.dumps({key: 0.1 for key in E.FACTOR_KEYS} | {"electricity_co2_kg_per_kwh": "x"}))
    with pytest.raises(ValueError) as excinfo:
        E.load_factors(path)
    assert "must be a number" in str(excinfo.value)


def test_a_negative_factor_is_rejected(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(json.dumps({key: 0.1 for key in E.FACTOR_KEYS} | {"electricity_price_usd_per_kwh": -1.0}))
    with pytest.raises(ValueError) as excinfo:
        E.load_factors(path)
    assert "non-negative" in str(excinfo.value)


def test_an_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(json.dumps({key: 0.1 for key in E.FACTOR_KEYS} | {"electricity_price": 0.2}))
    with pytest.raises(ValueError) as excinfo:
        E.load_factors(path)
    assert "unknown key" in str(excinfo.value)


def test_a_missing_factors_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.load_factors(tmp_path / "does-not-exist.json")


def test_the_cli_accepts_a_factors_flag_and_rejects_a_bad_file(tmp_path):
    """--factors is parsed and the file is validated before any simulation."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"electricity_price_usd_per_kwh": 0.2}))
    with pytest.raises(SystemExit) as excinfo:
        RE.main(["--all", "--factors", str(bad)])
    assert "invalid factors file" in str(excinfo.value)


def test_the_cli_help_documents_the_factors_schema(capsys):
    with pytest.raises(SystemExit):
        RE.main(["--help"])
    out = capsys.readouterr().out
    assert "--factors" in out
    for key in E.FACTOR_KEYS:
        assert key in out


# ---------------------------------------------------------------------------
# the computed $/tCO2 follow the factor file
# ---------------------------------------------------------------------------
def test_the_default_path_keeps_the_placeholder_numbers_and_labels():
    metrics = _metrics()
    assert metrics["cost_saved_usd"] == pytest.approx(
        10_000.0 * E.ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER
        + 1_000.0 * E.NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER
    )
    assert metrics["co2_saved_tco2"] == pytest.approx(
        (
            10_000.0 * E.ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER
            + 1_000.0 * E.NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER
        )
        / 1000.0
    )
    # the legacy PLACEHOLDER-labelled keys are still there for the default path
    assert metrics["cost_saved_usd_placeholder"] == metrics["cost_saved_usd"]
    assert metrics["co2_saved_tco2_placeholder"] == metrics["co2_saved_tco2"]
    assert metrics["factors_label"] == E.PLACEHOLDER


def test_an_override_file_changes_the_computed_cost_and_co2(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(
        json.dumps(
            {
                "electricity_price_usd_per_kwh": 0.20,
                "natural_gas_price_usd_per_kwh": 0.05,
                "electricity_co2_kg_per_kwh": 0.50,
                "natural_gas_co2_kg_per_kwh": 0.10,
            }
        )
    )
    factors = E.load_factors(path)
    assert not factors.is_placeholder
    metrics = _metrics(factors)
    assert metrics["cost_saved_usd"] == pytest.approx(10_000.0 * 0.20 + 1_000.0 * 0.05)
    assert metrics["co2_saved_tco2"] == pytest.approx((10_000.0 * 0.50 + 1_000.0 * 0.10) / 1000.0)
    # ... and they differ from the placeholder numbers
    default = _metrics()
    assert metrics["cost_saved_usd"] != default["cost_saved_usd"]
    assert metrics["co2_saved_tco2"] != default["co2_saved_tco2"]


def test_an_override_drops_the_placeholder_labels_of_the_factor_columns(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(
        json.dumps(
            {
                "electricity_price_usd_per_kwh": 0.20,
                "natural_gas_price_usd_per_kwh": 0.05,
                "electricity_co2_kg_per_kwh": 0.50,
                "natural_gas_co2_kg_per_kwh": 0.10,
            }
        )
    )
    factors = E.load_factors(path)
    metrics = _metrics(factors)
    assert "cost_saved_usd_placeholder" not in metrics
    assert "co2_saved_tco2_placeholder" not in metrics
    header = _detail_header(_table(factors))
    assert "USD saved (PLACEHOLDER)" not in header
    assert "tCO2 saved (PLACEHOLDER)" not in header
    assert "USD saved" in header
    assert "tCO2 saved" in header


def test_the_default_path_keeps_the_placeholder_labels_of_the_factor_columns():
    header = _detail_header(_table(E.default_factors()))
    assert "USD saved (PLACEHOLDER)" in header
    assert "tCO2 saved (PLACEHOLDER)" in header


# ---------------------------------------------------------------------------
# the default factors reproduce the numbers published in REPORT.md
# ---------------------------------------------------------------------------
def test_the_report_table_is_reproduced_by_the_default_factors():
    """REPORT.md's $ and tCO2 columns follow from the default (placeholder) factors.

    Each published row gives the kWh saved, the $ saved and the tCO2 saved.
    The electricity/gas split is solved from the $ column (the factors differ
    between the two fuels) and the tCO2 column is then predicted from it; the
    two agree to the rounding of the published table.
    """
    report = Path(__file__).resolve().parents[1] / "REPORT.md"
    rows = []
    for line in report.read_text().splitlines():
        if re.match(r"^\| E\d \| (calibrated|true) \|", line):
            fields = [f.strip() for f in line.strip().strip("|").split("|")]
            rows.append(fields)
    assert len(rows) == 10, "the published per-ECM table has 10 rows (5 ECMs x 2 models)"

    elec_price = E.ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER
    gas_price = E.NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER
    elec_co2 = E.ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER
    gas_co2 = E.NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER
    for fields in rows:
        kwh_saved = float(fields[2])
        usd_saved = float(fields[4])
        tco2_saved = float(fields[5])
        elec_saved = (usd_saved - gas_price * kwh_saved) / (elec_price - gas_price)
        gas_saved = kwh_saved - elec_saved
        predicted_tco2 = (elec_co2 * elec_saved + gas_co2 * gas_saved) / 1000.0
        assert predicted_tco2 == pytest.approx(tco2_saved, abs=0.001), fields


# ---------------------------------------------------------------------------
# the source path is recorded
# ---------------------------------------------------------------------------
def test_the_default_table_records_the_placeholder_factors_source():
    table = _table(E.default_factors())
    assert str(E.PLACEHOLDER_FACTORS_PATH) in table
    assert "All costs and prices are PLACEHOLDER values" in table


def test_an_override_table_records_the_factors_source_path(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(
        json.dumps(
            {
                "electricity_price_usd_per_kwh": 0.20,
                "natural_gas_price_usd_per_kwh": 0.05,
                "electricity_co2_kg_per_kwh": 0.50,
                "natural_gas_co2_kg_per_kwh": 0.10,
            }
        )
    )
    factors = E.load_factors(path)
    table = _table(factors)
    assert str(path) in table
    assert "label: CUSTOM" in table


def test_the_results_json_records_the_factors_source(tmp_path):
    path = tmp_path / "factors.json"
    path.write_text(
        json.dumps(
            {
                "electricity_price_usd_per_kwh": 0.20,
                "natural_gas_price_usd_per_kwh": 0.05,
                "electricity_co2_kg_per_kwh": 0.50,
                "natural_gas_co2_kg_per_kwh": 0.10,
            }
        )
    )
    factors = E.load_factors(path)
    results = {"factors": factors.record()}
    out = tmp_path / "ecm_headline_table.md"
    results_path = RE.write_outputs(out, _table(factors).splitlines(), results)
    written = json.loads(results_path.read_text())
    assert written["factors"]["source"] == str(path)
    assert written["factors"]["electricity_price_usd_per_kwh"] == 0.20
    assert written["factors"]["label"] == "CUSTOM"
    assert written["factors"]["is_placeholder"] is False
    assert "placeholders" not in written


def test_the_results_json_keeps_the_placeholder_block_for_the_default_path(tmp_path):
    factors = E.default_factors()
    results = {
        "factors": factors.record(),
        "placeholders": {
            "electricity_price_usd_per_kwh": factors.electricity_price_usd_per_kwh,
            "natural_gas_price_usd_per_kwh": factors.natural_gas_price_usd_per_kwh,
            "electricity_co2_kg_per_kwh": factors.electricity_co2_kg_per_kwh,
            "natural_gas_co2_kg_per_kwh": factors.natural_gas_co2_kg_per_kwh,
            "label": E.PLACEHOLDER,
            "source": factors.source,
        },
    }
    results_path = RE.write_outputs(tmp_path / "ecm_headline_table.md", [], results)
    written = json.loads(results_path.read_text())
    assert written["factors"]["source"] == str(E.PLACEHOLDER_FACTORS_PATH)
    assert written["factors"]["is_placeholder"] is True
    assert written["placeholders"]["label"] == E.PLACEHOLDER
