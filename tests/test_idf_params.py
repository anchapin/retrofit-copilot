"""Tests for the IDF parameter edits (the three bounded calibration parameters)."""

import pytest

import idf_params as P

MINI_IDF = """Version,26.2;

Building,MyBldg,North Axis,0,0,0,0,0,0;

ZoneInfiltration:DesignFlowRate,
    SPACE1-1 Infil 1,        !- Name
    SPACE1-1,                !- Zone or ZoneList or Space or SpaceList Name
    INFIL-SCH,               !- Schedule Name
    flow/zone,               !- Design Flow Rate Calculation Method
    0.032,                   !- Design Flow Rate {m3/s}
    ,                        !- Flow Rate per Floor Area {m3/s-m2}
    ,                        !- Flow Rate per Exterior Surface Area {m3/s-m2}
    ,                        !- Air Changes per Hour {1/hr}
    0,                       !- Constant Term Coefficient
    0,                       !- Temperature Term Coefficient
    0.2237,                  !- Velocity Term Coefficient
    0;                       !- Velocity Squared Term Coefficient

Material,
    WD10,                    !- Name
    MediumSmooth,            !- Roughness
    0.667,                   !- Thickness {m}
    0.115,                   !- Conductivity {W/m-K}
    513,                     !- Density {kg/m3}
    1381,                    !- Specific Heat {J/kg-K}
    0.9,                     !- Thermal Absorptance
    0.78,                    !- Solar Absorptance
    0.78;                    !- Visible Absorptance

Schedule:Compact,
    Clg-SetP-Sch,            !- Name
    Temperature,             !- Schedule Type Limits Name
    Through: 12/31,          !- Field 1
    For: WeekDays,           !- Field 2
    Until: 6:00,29.4,        !- Field 3
    Until: 20:00,23.9,       !- Field 5
    Until: 24:00,29.4;       !- Field 7
"""


def test_default_params_leave_the_physics_untouched():
    text, changes = P.apply_params(MINI_IDF, dict(P.DEFAULT_PARAMS))
    physics = [c for c in changes if c["parameter"] != "reporting"]
    assert physics == []
    # only the reporting meters were added
    assert {c["parameter"] for c in changes} == {"reporting"}


def test_infiltration_scale_doubles_the_design_flow_rate():
    text, changes = P.apply_params(MINI_IDF, {"infiltration_flow_scale": 2.0})
    infil = [c for c in changes if c["parameter"] == "infiltration_flow_scale"]
    assert len(infil) == 1
    assert infil[0]["old"] == pytest.approx(0.032)
    assert infil[0]["new"] == pytest.approx(0.064)
    assert "0.064" in text


def test_envelope_scale_changes_material_conductivity():
    text, changes = P.apply_params(MINI_IDF, {"envelope_conductivity_scale": 1.3})
    mat = [c for c in changes if c["parameter"] == "envelope_conductivity_scale"]
    assert len(mat) == 1
    assert mat[0]["old"] == pytest.approx(0.115)
    assert mat[0]["new"] == pytest.approx(0.1495)
    assert "0.1495" in text


def test_cooling_setpoint_shift_keeps_the_until_fields_intact():
    text, changes = P.apply_params(MINI_IDF, {"cooling_setpoint_offset_k": 1.0})
    sched = [c for c in changes if c["parameter"] == "cooling_setpoint_offset_k"]
    assert sorted(c["old"] for c in sched) == [23.9, 29.4, 29.4]
    assert sorted(c["new"] for c in sched) == [24.9, 30.4, 30.4]
    # the schedule must stay a valid Schedule:Compact
    assert "Until: 6:00,30.4," in text
    assert "Until: 20:00,24.9," in text


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError):
        P.apply_params(MINI_IDF, {"not_a_parameter": 1.0})


# --------------------------------------------------------------------------
# envelope_conductivity_scale_scope (issue #5)
# --------------------------------------------------------------------------
ENVELOPE_SCOPE_IDF = """\
Version,26.2;

Material,
    EXT-MAT,                 !- Name
    Rough,                   !- Roughness
    0.2,                     !- Thickness {m}
    0.5,                     !- Conductivity {W/m-K}
    900,                     !- Density {kg/m3}
    800,                     !- Specific Heat {J/kg-K}
    0.9, 0.7, 0.7;           !- Absorptance / reflectance

Material,
    INT-MAT,                 !- Name
    Rough,                   !- Roughness
    0.01,                    !- Thickness {m}
    0.16,                    !- Conductivity {W/m-K}
    800,                     !- Density {kg/m3}
    1000,                    !- Specific Heat {J/kg-K}
    0.9, 0.7, 0.7;           !- Absorptance / reflectance

Construction,
    EXT-CON,                 !- Name
    EXT-MAT;                 !- Outside Layer

Construction,
    INT-CON,                 !- Name
    INT-MAT;                 !- Outside Layer

BuildingSurface:Detailed,
    W1,                      !- Name
    WALL,                    !- Surface Type
    EXT-CON,                 !- Construction Name
    ZONE1,                   !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    0.5,                     !- View Factor to Ground
    4,                       !- Number of Vertices
    0.0, 0.0, 3.0, 10.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 3.0, 0.0, 10.0,
    0.0, 3.0, 10.0, 10.0;

BuildingSurface:Detailed,
    C1,                      !- Name
    CEILING,                 !- Surface Type
    INT-CON,                 !- Construction Name
    ZONE1,                   !- Zone Name
    ,                        !- Space Name
    Surface,                 !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    ,                        !- Sun Exposure
    ,                        !- Wind Exposure
    0.5,                     !- View Factor to Ground
    4,                       !- Number of Vertices
    0.0, 0.0, 3.0, 10.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 3.0, 0.0, 10.0,
    0.0, 3.0, 10.0, 10.0;
"""


def _conductivity(idf_text, material_name):
    for obj in P.parse_idf(idf_text):
        if obj.otype.lower() == "material" and obj.name.strip() == material_name:
            return float(obj.fields[3])
    raise AssertionError(f"material {material_name} not found")


def test_scoped_conductivity_leaves_an_interior_only_material_alone():
    """Known values: the interior carpet keeps 0.16 W/m-K."""
    text, changes = P.apply_params(
        ENVELOPE_SCOPE_IDF,
        {
            "envelope_conductivity_scale": 2.0,
            "envelope_conductivity_scale_scope": "envelope",
        },
    )
    assert _conductivity(text, "INT-MAT") == pytest.approx(0.16)
    scaled = {c["name"]: (c["old"], c["new"]) for c in changes
              if c["parameter"] == "envelope_conductivity_scale"}
    assert scaled == {"EXT-MAT": (pytest.approx(0.5), pytest.approx(1.0))}


def test_scoped_conductivity_scales_an_envelope_material():
    """Known values: the exterior wall material goes 0.5 -> 1.0 W/m-K."""
    text, changes = P.apply_params(
        ENVELOPE_SCOPE_IDF,
        {"envelope_conductivity_scale": 2.0},
        envelope_conductivity_scale_scope="envelope",
    )
    assert _conductivity(text, "EXT-MAT") == pytest.approx(1.0)
    assert _conductivity(text, "INT-MAT") == pytest.approx(0.16)


def test_blanket_conductivity_scales_both_materials():
    """The current (slice-1) behaviour: every Material conductivity scales."""
    text, changes = P.apply_params(
        ENVELOPE_SCOPE_IDF, {"envelope_conductivity_scale": 2.0}
    )
    assert _conductivity(text, "EXT-MAT") == pytest.approx(1.0)
    assert _conductivity(text, "INT-MAT") == pytest.approx(0.32)
    scaled = {c["name"] for c in changes
              if c["parameter"] == "envelope_conductivity_scale"}
    assert scaled == {"EXT-MAT", "INT-MAT"}


def test_blanket_conductivity_scope_is_the_default():
    """No scope given => blanket (bit-identical to the recorded behaviour)."""
    default_text, _ = P.apply_params(
        ENVELOPE_SCOPE_IDF, {"envelope_conductivity_scale": 2.0}
    )
    blanket_text, _ = P.apply_params(
        ENVELOPE_SCOPE_IDF,
        {
            "envelope_conductivity_scale": 2.0,
            "envelope_conductivity_scale_scope": "blanket",
        },
    )
    assert default_text == blanket_text
    assert _conductivity(default_text, "INT-MAT") == pytest.approx(0.32)


def test_an_unknown_conductivity_scope_is_rejected():
    with pytest.raises(ValueError):
        P.apply_params(
            ENVELOPE_SCOPE_IDF,
            {
                "envelope_conductivity_scale": 2.0,
                "envelope_conductivity_scale_scope": "roof-only",
            },
        )


def test_scoped_conductivity_without_any_outdoor_surface_is_an_error():
    with pytest.raises(ValueError):
        P.apply_params(
            MINI_IDF,
            {
                "envelope_conductivity_scale": 2.0,
                "envelope_conductivity_scale_scope": "envelope",
            },
        )


def test_the_scoped_conductivity_selection_is_e1s_construction_selection():
    """The scoped mode reuses E1's selection, not a parallel implementation."""
    import ecms as E

    objects = P.parse_idf(ENVELOPE_SCOPE_IDF)
    _, by_construction = E.opaque_envelope_area(objects)
    assert P.envelope_construction_names(objects) == sorted(by_construction)
    # the materials are the layers of exactly those constructions
    assert P.envelope_material_names(objects) == ["EXT-MAT"]


def test_parsing_keeps_every_object():
    objects = P.parse_idf(MINI_IDF)
    types = [o.otype for o in objects]
    assert "ZoneInfiltration:DesignFlowRate" in types
    assert "Material" in types
    assert "Schedule:Compact" in types
    infil = [o for o in objects if o.otype == "ZoneInfiltration:DesignFlowRate"][0]
    assert infil.fields[4] == "0.032"
    sched = [o for o in objects if o.otype == "Schedule:Compact"][0]
    assert sched.fields[0] == "Clg-SetP-Sch"
    assert "Until: 6:00,29.4" in sched.fields
