"""Tests for the five ECMs and for the slice-2 ranking / cost maths.

The tests use a small self-contained IDF fixture (so they do not depend on an
EnergyPlus installation) plus the shipped example when it is available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

import ecms as E
import idf_params as P
import results as RES
import run_ecms as RE

FIXTURE_IDF = """\
Version,26.2;

Material,
  WD01, MediumSmooth, 0.0191, 0.115, 513.0, 1381.0, 0.9, 0.78, 0.78;

Material,
  RG01, Rough, 0.0127, 1.442, 881.0, 1674.0, 0.9, 0.65, 0.65;

Material,
  GP02, MediumSmooth, 0.0159, 0.16, 801.0, 837.0, 0.9, 0.75, 0.75;

Material:NoMass,
  MAT-CLNG-1, Rough, 0.65225929, 0.65, 0.65, 0.65;

Construction,
  ROOF-1, RG01, WD01;

Construction,
  WALL-1, WD01, GP02;

Construction,
  CLNG-1, MAT-CLNG-1;

Construction,
  INT-WALL-1, GP02;

WindowMaterial:Glazing,
  CLEAR 3MM, SpectralAverage, , 0.003, 0.837, 0.075, 0.075, 0.898, 0.081, 0.081,
  0.0, 0.84, 0.84, 0.9;

WindowMaterial:Gas,
  AIR 13MM, Air, 0.0127;

Construction,
  Dbl Clr 3mm/13mm Air, CLEAR 3MM, AIR 13MM, CLEAR 3MM;

BuildingSurface:Detailed,
  TOP-1, ROOF, ROOF-1, ZONE1, , Outdoors, , SunExposed, WindExposed, 0.5, 4,
  0.0, 0.0, 3.0, 10.0, 0.0, 3.0, 10.0, 10.0, 3.0, 0.0, 10.0, 3.0;

BuildingSurface:Detailed,
  WALL-1PF, WALL, WALL-1, ZONE1, , Outdoors, , SunExposed, WindExposed, 0.5, 4,
  0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 10.0, 3.0, 0.0, 10.0, 0.0;

BuildingSurface:Detailed,
  CEIL-1, CEILING, CLNG-1, ZONE1, , Surface, , , , 0.5, 4,
  0.0, 0.0, 3.0, 10.0, 0.0, 3.0, 10.0, 10.0, 3.0, 0.0, 10.0, 3.0;

BuildingSurface:Detailed,
  IWALL-1, WALL, INT-WALL-1, ZONE1, , Surface, , , , 0.5, 4,
  0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 10.0, 3.0, 0.0, 10.0, 0.0;

FenestrationSurface:Detailed,
  WF-1, WINDOW, Dbl Clr 3mm/13mm Air, WALL-1PF, , 0.5, , 1, 4,
  0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 1.0;

Boiler:HotWater,
  Central Boiler, NaturalGas, autosize, 0.8, LeavingBoiler, BoilerEfficiency,
  autosize, 0.0, 1.2, 1.0, Boiler In Node, Boiler Out Node, 100.,
  LeavingSetpointModulated;

Curve:Quadratic,
  BoilerEfficiency, 1.0, 0.0, 0.0, 0, 1;

Chiller:Electric,
  Central Chiller, AirCooled, autosize, 3.2, Chiller In Node, Chiller Out Node,
  Cond In Node, Cond Out Node, 0.0, 1.0, 0.65, 35.0, 2.778, 6.67, autosize,
  autosize, 0.9949, -0.045954, -0.0013543, 2.333, -1.975, 0.6121, 0.03303,
  0.6852, 0.2818, 5, LeavingSetpointModulated;

Schedule:Compact,
  Clg-SetP-Sch, Temperature,
  Through: 12/31,
  For: SummerDesignDay,
  Until: 24:00,23.9,
  For: WinterDesignDay,
  Until: 24:00,29.4,
  For: WeekDays,
  Until: 6:00,29.4,
  Until: 20:00,23.9,
  Until: 24:00,29.4,
  For: WeekEnds Holiday,
  Until: 24:00,29.4,
  For: AllOtherDays,
  Until: 24:00,29.4;

Schedule:Compact,
  Htg-SetP-Sch, Temperature,
  Through: 12/31,
  For: SummerDesignDay,
  Until: 24:00,16.7,
  For: WinterDesignDay,
  Until: 24:00,22.2,
  For: WeekDays,
  Until: 6:00,16.7,
  Until: 20:00,22.2,
  Until: 24:00,16.7,
  For: WeekEnds Holiday,
  Until: 24:00,16.7,
  For: AllOtherDays,
  Until: 24:00,16.7;

Schedule:Compact,
  PlenumClg-SetP-Sch, Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,40.0;

Schedule:Compact,
  PlenumHtg-SetP-Sch, Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,12.8;
"""


@pytest.fixture()
def fixture_idf() -> str:
    return FIXTURE_IDF


def objects_of(text: str) -> Dict[str, List[P.IdfObject]]:
    out: Dict[str, List[P.IdfObject]] = {}
    for obj in P.parse_idf(text):
        out.setdefault(obj.otype.lower(), []).append(obj)
    return out


def find(objects, otype: str, name: str) -> P.IdfObject:
    if isinstance(objects, dict):
        objects = objects.get(otype.lower(), [])
    for obj in objects:
        if obj.otype.lower() == otype.lower() and obj.fields[0].strip() == name:
            return obj
    raise KeyError(f"{otype} {name} not found")


# ---------------------------------------------------------------------------
# E1 - envelope insulation
# ---------------------------------------------------------------------------
def test_e1_touches_only_the_outdoor_envelope_constructions(fixture_idf):
    new_text, changes = E.get_ecm("E1").apply(fixture_idf)
    construction_changes = [c for c in changes if c["object"] == "Construction"]
    assert {c["name"] for c in construction_changes} == {"ROOF-1", "WALL-1"}
    for change in construction_changes:
        assert change["field"] == "Outside Layer"
        assert change["old"] != change["new"]
        assert change["new"] == E.E1_INSULATION_NAME

    objects = objects_of(new_text)
    roof = find(objects, "Construction", "ROOF-1")
    wall = find(objects, "Construction", "WALL-1")
    assert roof.fields[1] == E.E1_INSULATION_NAME
    assert wall.fields[1] == E.E1_INSULATION_NAME
    # the original layers are still there, in order
    assert roof.fields[2:] == ["RG01", "WD01"]
    assert wall.fields[2:] == ["WD01", "GP02"]


def test_e1_leaves_interior_materials_and_constructions_alone(fixture_idf):
    new_text, _changes = E.get_ecm("E1").apply(fixture_idf)
    before = objects_of(fixture_idf)
    after = objects_of(new_text)
    for name in ("GP02", "MAT-CLNG-1"):
        old = find(before, "Material", name) if name == "GP02" else find(before, "Material:NoMass", name)
        new = find(after, "Material", name) if name == "GP02" else find(after, "Material:NoMass", name)
        assert old.fields == new.fields
    for name in ("CLNG-1", "INT-WALL-1"):
        assert find(before, "Construction", name).fields == find(after, "Construction", name).fields


def test_e1_adds_exactly_one_material_with_the_stated_properties(fixture_idf):
    new_text, changes = E.get_ecm("E1").apply(fixture_idf)
    materials = objects_of(new_text)["material"]
    added = [m for m in materials if m.fields[0].strip() == E.E1_INSULATION_NAME]
    assert len(added) == 1
    assert float(added[0].fields[2]) == pytest.approx(0.05)
    assert float(added[0].fields[3]) == pytest.approx(0.04)
    assert sum(1 for c in changes if c["object"] == "Material") == 1


def test_e1_cost_is_the_placeholder_area_times_the_placeholder_rate(fixture_idf):
    cost = E.get_ecm("E1").cost(fixture_idf)
    objects = P.parse_idf(fixture_idf)
    area, by_construction = E.opaque_envelope_area(objects)
    assert set(by_construction) == {"ROOF-1", "WALL-1"}
    assert area == pytest.approx(100.0 + 30.0)
    assert cost["amount_usd"] == pytest.approx(area * E.E1_INSULATION_UNIT_COST_USD_PER_M2_PLACEHOLDER)
    assert cost["label"] == E.PLACEHOLDER


# ---------------------------------------------------------------------------
# E2 - glazing
# ---------------------------------------------------------------------------
def test_e2_replaces_only_the_window_constructions(fixture_idf):
    new_text, changes = E.get_ecm("E2").apply(fixture_idf)
    construction_changes = [c for c in changes if c["object"] == "Construction"]
    assert {c["name"] for c in construction_changes} == {"Dbl Clr 3mm/13mm Air"}
    objects = objects_of(new_text)
    window = find(objects, "Construction", "Dbl Clr 3mm/13mm Air")
    assert window.fields == ["Dbl Clr 3mm/13mm Air", E.E2_GLAZING_NAME]
    # the shipped glazing material is untouched
    glazing = find(objects, "WindowMaterial:Glazing", "CLEAR 3MM")
    assert float(glazing.fields[4]) == pytest.approx(0.837)
    # the fenestration surfaces still reference the same construction
    fen = find(objects, "FenestrationSurface:Detailed", "WF-1")
    assert fen.fields[2].strip() == "Dbl Clr 3mm/13mm Air"


def test_e2_states_its_u_factor_and_shgc(fixture_idf):
    new_text, _changes = E.get_ecm("E2").apply(fixture_idf)
    objects = objects_of(new_text)
    glazing = find(objects, "WindowMaterial:SimpleGlazingSystem", E.E2_GLAZING_NAME)
    assert float(glazing.fields[1]) == pytest.approx(1.60)
    assert float(glazing.fields[2]) == pytest.approx(0.30)
    assert float(glazing.fields[3]) == pytest.approx(0.55)


def test_e2_cost_is_the_placeholder_glazed_area_times_the_placeholder_rate(fixture_idf):
    cost = E.get_ecm("E2").cost(fixture_idf)
    area, by_construction = E.glazed_area(P.parse_idf(fixture_idf))
    assert set(by_construction) == {"Dbl Clr 3mm/13mm Air"}
    assert area == pytest.approx(1.0)
    assert cost["amount_usd"] == pytest.approx(area * E.E2_GLAZING_UNIT_COST_USD_PER_M2_PLACEHOLDER)
    assert cost["label"] == E.PLACEHOLDER


# ---------------------------------------------------------------------------
# E3 - plant efficiency
# ---------------------------------------------------------------------------
def test_e3_changes_exactly_the_two_plant_efficiencies(fixture_idf):
    new_text, changes = E.get_ecm("E3").apply(fixture_idf)
    assert len(changes) == 2
    by_field = {c["field"]: c for c in changes}
    assert by_field["Nominal COP {W/W}"]["old"] == pytest.approx(3.2)
    assert by_field["Nominal COP {W/W}"]["new"] == pytest.approx(4.5)
    assert by_field["Nominal Thermal Efficiency"]["old"] == pytest.approx(0.8)
    assert by_field["Nominal Thermal Efficiency"]["new"] == pytest.approx(0.92)
    objects = objects_of(new_text)
    chiller = find(objects, "Chiller:Electric", "Central Chiller")
    assert float(chiller.fields[3]) == pytest.approx(4.5)
    boiler = find(objects, "Boiler:HotWater", "Central Boiler")
    assert float(boiler.fields[3]) == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# E4 - electrification
# ---------------------------------------------------------------------------
def test_e4_switches_the_boiler_fuel_to_electricity(fixture_idf):
    new_text, changes = E.get_ecm("E4").apply(fixture_idf)
    fields = {c["field"]: c for c in changes}
    assert fields["Fuel Type"]["old"] == "NaturalGas"
    assert fields["Fuel Type"]["new"] == "Electricity"
    assert fields["Nominal Thermal Efficiency"]["old"] == pytest.approx(0.8)
    assert fields["Nominal Thermal Efficiency"]["new"] == pytest.approx(1.0)
    objects = objects_of(new_text)
    boiler = find(objects, "Boiler:HotWater", "Central Boiler")
    assert boiler.fields[1].strip() == "Electricity"
    assert float(boiler.fields[3]) == pytest.approx(1.0)
    assert boiler.fields[5].strip() == ""  # constant-efficiency boiler


def test_e4_leaves_the_cooling_plant_alone(fixture_idf):
    new_text, _changes = E.get_ecm("E4").apply(fixture_idf)
    before = objects_of(fixture_idf)
    after = objects_of(new_text)
    assert find(before, "Chiller:Electric", "Central Chiller").fields == find(
        after, "Chiller:Electric", "Central Chiller"
    ).fields


# ---------------------------------------------------------------------------
# E5 - controls
# ---------------------------------------------------------------------------
def test_e5_widens_the_deadband_in_the_occupied_blocks_only(fixture_idf):
    new_text, changes = E.get_ecm("E5").apply(fixture_idf)
    assert len(changes) == 2
    values = {(c["name"], c["old"], c["new"]) for c in changes}
    assert ("Clg-SetP-Sch", 23.9, 24.9) in values
    assert ("Htg-SetP-Sch", 22.2, 21.2) in values
    objects = objects_of(new_text)
    clg = find(objects, "Schedule:Compact", "Clg-SetP-Sch")
    htg = find(objects, "Schedule:Compact", "Htg-SetP-Sch")
    # night setback values are untouched
    assert "Until: 6:00,29.4" in clg.fields
    assert "Until: 24:00,29.4" in clg.fields
    assert "Until: 6:00,16.7" in htg.fields
    # the plenum setpoint schedules are untouched
    plenum = find(objects, "Schedule:Compact", "PlenumClg-SetP-Sch")
    assert "Until: 24:00,40.0" in plenum.fields


def test_e5_composes_with_the_slice1_cooling_setpoint_offset(fixture_idf):
    shifted, _ = P.apply_params(
        fixture_idf, {"cooling_setpoint_offset_k": 1.0}, schedule_name="Clg-SetP-Sch"
    )
    new_text, changes = E.get_ecm("E5").apply(shifted)
    cooling = [c for c in changes if c["name"] == "Clg-SetP-Sch"]
    assert len(cooling) == 1
    assert cooling[0]["old"] == pytest.approx(24.9)
    assert cooling[0]["new"] == pytest.approx(25.9)
    objects = objects_of(new_text)
    clg = find(objects, "Schedule:Compact", "Clg-SetP-Sch")
    assert "Until: 20:00,25.9" in clg.fields


# ---------------------------------------------------------------------------
# placeholder labelling
# ---------------------------------------------------------------------------
def test_every_cost_constant_carries_the_placeholder_marker():
    E.check_placeholder_labels()  # raises when a marker is missing
    for name in E.COST_CONSTANTS:
        assert E.PLACEHOLDER in name


def test_every_ecm_cost_is_labelled_as_a_placeholder(fixture_idf):
    for ecm in E.ECMS:
        cost = ecm.cost(fixture_idf)
        assert cost["label"] == E.PLACEHOLDER
        assert E.PLACEHOLDER in cost["basis"]
        assert E.PLACEHOLDER in ecm.cost_basis
        assert cost["amount_usd"] > 0.0
        described = ecm.to_dict(fixture_idf)
        assert described["cost_label"] == E.PLACEHOLDER
        assert described["placeholder_cost_usd"] == cost["amount_usd"]


def test_the_five_ecms_are_defined_with_the_expected_ids():
    assert E.ECM_IDS == ("E1", "E2", "E3", "E4", "E5")
    for ecm in E.ECMS:
        assert ecm.definition.strip()
        assert ecm.mechanism.strip()
        assert ecm.assumptions.strip()
        assert ecm.grid_flexibility.strip()


# ---------------------------------------------------------------------------
# ranking / cost / payback maths (known values)
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


def test_savings_cost_co2_and_payback_math():
    baseline = _series("base", 40000.0, 10000.0)
    ecm = _series("ecm", 30000.0, 9000.0)
    metrics = RE.ecm_metrics(baseline, ecm, placeholder_cost_usd=10_000.0)
    assert metrics["energy_saved_kwh"] == pytest.approx(11_000.0)
    assert metrics["energy_saved_pct"] == pytest.approx(22.0)
    assert metrics["cost_saved_usd_placeholder"] == pytest.approx(
        10_000.0 * E.ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER
        + 1_000.0 * E.NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER
    )
    assert metrics["co2_saved_tco2_placeholder"] == pytest.approx(
        (
            10_000.0 * E.ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER
            + 1_000.0 * E.NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER
        )
        / 1000.0
    )
    assert metrics["simple_payback_years_placeholder"] == pytest.approx(
        10_000.0 / metrics["cost_saved_usd_placeholder"]
    )


def test_no_payback_when_the_ecm_costs_money():
    baseline = _series("base", 10000.0, 0.0)
    ecm = _series("ecm", 12000.0, 0.0)
    metrics = RE.ecm_metrics(baseline, ecm, placeholder_cost_usd=5_000.0)
    assert metrics["energy_saved_kwh"] == pytest.approx(-2000.0)
    assert metrics["cost_saved_usd_placeholder"] < 0
    assert metrics["simple_payback_years_placeholder"] is None


def test_ranking_is_by_annual_kwh_saved_descending():
    metrics = {
        "E1": {"energy_saved_kwh": 1000.0},
        "E2": {"energy_saved_kwh": 3000.0},
        "E3": {"energy_saved_kwh": 2000.0},
        "E4": {"energy_saved_kwh": 500.0},
        "E5": {"energy_saved_kwh": 3000.0},
    }
    order = RE.ranking_from_metrics(metrics)
    assert order[:2] == ["E2", "E5"]  # tie broken by ECM id
    assert order[-1] == "E4"


def test_comfort_metrics_report_band_hours():
    series = _series("s", 1.0, 1.0, temp=30.0)
    comfort = RE.comfort_metrics(series)
    assert comfort["zone_mean_air_temp_c"] == pytest.approx(30.0)
    assert comfort["zone_hours_outside_comfort_band"] == 8760
    assert comfort["zone_hours_outside_comfort_band_band_c"] == [20.0, 26.0]
