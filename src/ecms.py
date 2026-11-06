"""The five pre-defined energy conservation measures (ECMs) of the POC.

Each ECM is a *deterministic* transformation of an IDF text.  The shipped
EnergyPlus example file is never edited in place: an ECM parses the IDF,
changes named objects on the parsed copy, and returns

    (new_idf_text, change_record)

where ``change_record`` is a list of one entry per changed value:

    {"ecm": "E1", "object": "Construction", "name": "ROOF-1",
     "field": "Outside Layer", "old": "RG01", "new": "E1-INSULATION-50MM"}

Every ECM is applied on top of a *model* (the shipped IDF plus the slice-1
calibration parameters), so the same ECM code runs on the calibrated model
and on the true model.

Costs
-----
**Every cost in this module is a PLACEHOLDER.**  No figure here is a real
price, a quotation, or a vendor price.  The placeholders are order-of-magnitude
numbers with a stated basis (RSMeans-style unit rate x area, or a lump-sum
equipment+install placeholder) so that the *ranking* and the *simple payback*
of the ECMs can be exercised end to end.  They must not be used for
procurement, business case, or any external claim.

The placeholder labels are checked mechanically: :func:`check_placeholder_labels`
raises when a cost constant does not carry the marker, and the test suite runs
it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import idf_params as P

#: Marker that every cost constant and every cost output must carry.
PLACEHOLDER = "PLACEHOLDER"

# ---------------------------------------------------------------------------
# placeholder cost constants (RSMeans-style placeholder unit rates / lump sums)
# ---------------------------------------------------------------------------
#: PLACEHOLDER RSMeans-style unit rate for added exterior insulation
#: (material + labour), applied to the opaque envelope area.
E1_INSULATION_UNIT_COST_USD_PER_M2_PLACEHOLDER = 60.0

#: PLACEHOLDER RSMeans-style unit rate for a low-e glazing replacement
#: (glazing + labour), applied to the glazed area.
E2_GLAZING_UNIT_COST_USD_PER_M2_PLACEHOLDER = 450.0

#: PLACEHOLDER lump sum for a chiller + boiler plant efficiency replacement.
E3_PLANT_UPGRADE_COST_USD_PLACEHOLDER = 45_000.0

#: PLACEHOLDER lump sum for a fuel switch to electric resistance heating,
#: including a nominal service upgrade.
E4_ELECTRIFICATION_COST_USD_PLACEHOLDER = 25_000.0

#: PLACEHOLDER lump sum for thermostat reprogramming / replacement.
E5_CONTROLS_COST_USD_PLACEHOLDER = 2_500.0

#: PLACEHOLDER energy prices used for the cost ranking (NOT tariffs).
ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER = 0.13
NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER = 0.06

#: PLACEHOLDER grid emission factors (NOT measured factors).
ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER = 0.37
NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER = 0.18

COST_CONSTANTS: Dict[str, float] = {
    "E1_INSULATION_UNIT_COST_USD_PER_M2_PLACEHOLDER": E1_INSULATION_UNIT_COST_USD_PER_M2_PLACEHOLDER,
    "E2_GLAZING_UNIT_COST_USD_PER_M2_PLACEHOLDER": E2_GLAZING_UNIT_COST_USD_PER_M2_PLACEHOLDER,
    "E3_PLANT_UPGRADE_COST_USD_PLACEHOLDER": E3_PLANT_UPGRADE_COST_USD_PLACEHOLDER,
    "E4_ELECTRIFICATION_COST_USD_PLACEHOLDER": E4_ELECTRIFICATION_COST_USD_PLACEHOLDER,
    "E5_CONTROLS_COST_USD_PLACEHOLDER": E5_CONTROLS_COST_USD_PLACEHOLDER,
    "ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER": ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER,
    "NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER": NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER,
    "ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER": ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER,
    "NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER": NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER,
}


def check_placeholder_labels() -> None:
    """Raise if any cost constant does not carry the PLACEHOLDER marker."""
    missing = sorted(name for name in COST_CONSTANTS if PLACEHOLDER not in name)
    if missing:
        raise ValueError(f"cost constants without the PLACEHOLDER marker: {missing}")


def _fmt(value: float) -> str:
    return f"{value:.6g}"


# ---------------------------------------------------------------------------
# geometry helpers (areas are read from the IDF, never hard-coded)
# ---------------------------------------------------------------------------
def _surface_area(vertex_fields: Sequence[str]) -> float:
    """Area of an EnergyPlus planar surface given its vertex fields."""
    pts: List[Tuple[float, float, float]] = []
    for i in range(0, len(vertex_fields) - 2, 3):
        pts.append(
            (
                float(vertex_fields[i]),
                float(vertex_fields[i + 1]),
                float(vertex_fields[i + 2]),
            )
        )
    if len(pts) < 3:
        return 0.0
    cx = cy = cz = 0.0
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        cx += p[1] * q[2] - q[1] * p[2]
        cy += p[2] * q[0] - q[2] * p[0]
        cz += p[0] * q[1] - q[0] * p[1]
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def opaque_envelope_area(objects: Sequence[P.IdfObject]) -> Tuple[float, Dict[str, float]]:
    """Opaque envelope area [m2] that faces the outdoors.

    Only ``BuildingSurface:Detailed`` objects whose *Outside Boundary
    Condition* is ``Outdoors`` and whose surface type is a wall or a roof are
    counted, and only the constructions that those surfaces reference are
    returned.  Interior constructions (ceilings, interior walls, slab) are
    therefore never touched by the envelope ECM.
    """
    by_construction: Dict[str, float] = {}
    for obj in objects:
        if obj.otype.lower() != "buildingsurface:detailed":
            continue
        fields = obj.fields
        if len(fields) < 11:
            continue
        surface_type = fields[1].strip().lower()
        construction = fields[2].strip()
        obc = fields[5].strip().lower()
        if surface_type not in ("wall", "roof"):
            continue
        if obc != "outdoors":
            continue
        n_vertices = int(float(fields[10]))
        vertices = fields[11 : 11 + 3 * n_vertices]
        by_construction[construction] = by_construction.get(construction, 0.0) + _surface_area(vertices)
    total = sum(by_construction.values())
    return total, by_construction


def glazed_area(objects: Sequence[P.IdfObject]) -> Tuple[float, Dict[str, float]]:
    """Glazed area [m2] per fenestration construction."""
    by_construction: Dict[str, float] = {}
    for obj in objects:
        if obj.otype.lower() != "fenestrationsurface:detailed":
            continue
        fields = obj.fields
        if len(fields) < 9:
            continue
        construction = fields[2].strip()
        n_vertices = int(float(fields[8]))
        vertices = fields[9 : 9 + 3 * n_vertices]
        by_construction[construction] = by_construction.get(construction, 0.0) + _surface_area(vertices)
    return sum(by_construction.values()), by_construction


# ---------------------------------------------------------------------------
# ECM definition
# ---------------------------------------------------------------------------
@dataclass
class Ecm:
    """One energy conservation measure."""

    ecm_id: str
    name: str
    definition: str
    mechanism: str
    cost_basis: str
    assumptions: str
    grid_flexibility: str
    apply_fn: Callable[[str], Tuple[str, List[Dict[str, Any]]]]
    cost_fn: Callable[[str], Dict[str, Any]]

    # -- public API --------------------------------------------------------
    def apply(self, idf_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply the ECM to a copy of ``idf_text``; return (new text, changes)."""
        return self.apply_fn(idf_text)

    def cost(self, idf_text: str) -> Dict[str, Any]:
        """Placeholder installed cost of the ECM for the geometry in ``idf_text``."""
        return self.cost_fn(idf_text)

    def to_dict(self, idf_text: str) -> Dict[str, Any]:
        cost = self.cost(idf_text)
        return {
            "ecm_id": self.ecm_id,
            "name": self.name,
            "definition": self.definition,
            "mechanism": self.mechanism,
            "cost_basis": self.cost_basis,
            "assumptions": self.assumptions,
            "grid_flexibility": self.grid_flexibility,
            "placeholder_cost_usd": cost["amount_usd"],
            "cost_label": cost["label"],
        }


# ---------------------------------------------------------------------------
# E1 - opaque envelope insulation
# ---------------------------------------------------------------------------
E1_INSULATION_NAME = "E1-INSULATION-50MM"
#: PLACEHOLDER material properties of the added insulation layer.
E1_INSULATION_THICKNESS_M = 0.05
E1_INSULATION_CONDUCTIVITY_W_PER_MK = 0.04
E1_INSULATION_DENSITY_KG_PER_M3 = 30.0
E1_INSULATION_SPECIFIC_HEAT_J_PER_KG_K = 840.0


def _apply_e1(idf_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Add a 50 mm insulation layer to the outside of the opaque envelope.

    Mechanism: one new ``Material`` object is appended to the IDF and inserted
    as the *outside* layer of every construction that is used by an
    outdoor-facing wall or roof surface.  Interior constructions (ceiling,
    interior wall, floor slab) are never touched, which is the difference
    from slice 1's blanket ``envelope_conductivity_scale``.
    """
    objects = P.parse_idf(idf_text)
    changes: List[Dict[str, Any]] = []

    _, by_construction = opaque_envelope_area(objects)
    if not by_construction:
        raise ValueError("E1: no outdoor-facing opaque envelope found in the IDF")

    # the new material (added once)
    material = P.IdfObject(
        otype="Material",
        fields=[
            E1_INSULATION_NAME,
            "MediumRough",
            _fmt(E1_INSULATION_THICKNESS_M),
            _fmt(E1_INSULATION_CONDUCTIVITY_W_PER_MK),
            _fmt(E1_INSULATION_DENSITY_KG_PER_M3),
            _fmt(E1_INSULATION_SPECIFIC_HEAT_J_PER_KG_K),
            "0.9",
            "0.6",
            "0.6",
        ],
        raw="",
    )
    material._dirty = True  # type: ignore[attr-defined]
    objects.append(material)
    changes.append(
        {
            "ecm": "E1",
            "object": "Material",
            "name": E1_INSULATION_NAME,
            "field": "new object",
            "old": None,
            "new": f"thickness {_fmt(E1_INSULATION_THICKNESS_M)} m, "
            f"conductivity {_fmt(E1_INSULATION_CONDUCTIVITY_W_PER_MK)} W/m-K",
        }
    )

    for obj in objects:
        if obj.otype.lower() != "construction":
            continue
        name = obj.fields[0].strip()
        if name not in by_construction:
            continue
        if len(obj.fields) < 2:
            continue
        old_layer = obj.fields[1]
        obj.fields = [obj.fields[0], E1_INSULATION_NAME] + list(obj.fields[1:])
        obj._dirty = True  # type: ignore[attr-defined]
        changes.append(
            {
                "ecm": "E1",
                "object": "Construction",
                "name": name,
                "field": "Outside Layer",
                "old": old_layer,
                "new": E1_INSULATION_NAME,
            }
        )
    return P.render_idf(objects), changes


def _cost_e1(idf_text: str) -> Dict[str, Any]:
    objects = P.parse_idf(idf_text)
    area, by_construction = opaque_envelope_area(objects)
    return {
        "amount_usd": area * E1_INSULATION_UNIT_COST_USD_PER_M2_PLACEHOLDER,
        "label": PLACEHOLDER,
        "basis": (
            f"{area:.1f} m2 outdoor-facing opaque envelope "
            f"({', '.join(f'{k}: {v:.1f} m2' for k, v in sorted(by_construction.items()))}) "
            f"x PLACEHOLDER RSMeans-style unit rate "
            f"{E1_INSULATION_UNIT_COST_USD_PER_M2_PLACEHOLDER:.0f} USD/m2 "
            "(added insulation material + labour)"
        ),
    }


# ---------------------------------------------------------------------------
# E2 - glazing upgrade
# ---------------------------------------------------------------------------
E2_GLAZING_NAME = "E2-LOW-E"
#: PLACEHOLDER / nominal product targets for the replacement glazing.
E2_GLAZING_U_FACTOR_W_PER_M2K = 1.60
E2_GLAZING_SHGC = 0.30
E2_GLAZING_VISIBLE_TRANSMITTANCE = 0.55


def _apply_e2(idf_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace the window constructions with a low-e glazing system.

    Mechanism: one new ``WindowMaterial:SimpleGlazingSystem`` object (stated
    U-factor and SHGC) is appended and every window construction is reduced
    to that single layer.  The two window constructions of the shipped example
    are ``Dbl Clr 3mm/13mm Air`` (double clear / air) and ``Sgl Grey 3mm``
    (single grey); both are replaced by the same low-e system, so the edit is
    symmetric across the glazing of the building.
    """
    objects = P.parse_idf(idf_text)
    changes: List[Dict[str, Any]] = []

    total, by_construction = glazed_area(objects)
    if not by_construction:
        raise ValueError("E2: no fenestration found in the IDF")

    glazing = P.IdfObject(
        otype="WindowMaterial:SimpleGlazingSystem",
        fields=[
            E2_GLAZING_NAME,
            _fmt(E2_GLAZING_U_FACTOR_W_PER_M2K),
            _fmt(E2_GLAZING_SHGC),
            _fmt(E2_GLAZING_VISIBLE_TRANSMITTANCE),
        ],
        raw="",
    )
    glazing._dirty = True  # type: ignore[attr-defined]
    objects.append(glazing)
    changes.append(
        {
            "ecm": "E2",
            "object": "WindowMaterial:SimpleGlazingSystem",
            "name": E2_GLAZING_NAME,
            "field": "new object",
            "old": None,
            "new": f"U-factor {_fmt(E2_GLAZING_U_FACTOR_W_PER_M2K)} W/m2-K, "
            f"SHGC {_fmt(E2_GLAZING_SHGC)}, VT {_fmt(E2_GLAZING_VISIBLE_TRANSMITTANCE)}",
        }
    )

    for obj in objects:
        if obj.otype.lower() != "construction":
            continue
        name = obj.fields[0].strip()
        if name not in by_construction:
            continue
        old_layers = list(obj.fields[1:])
        obj.fields = [obj.fields[0], E2_GLAZING_NAME]
        obj._dirty = True  # type: ignore[attr-defined]
        changes.append(
            {
                "ecm": "E2",
                "object": "Construction",
                "name": name,
                "field": "layers",
                "old": ", ".join(old_layers),
                "new": E2_GLAZING_NAME,
            }
        )
    return P.render_idf(objects), changes


def _cost_e2(idf_text: str) -> Dict[str, Any]:
    objects = P.parse_idf(idf_text)
    area, by_construction = glazed_area(objects)
    return {
        "amount_usd": area * E2_GLAZING_UNIT_COST_USD_PER_M2_PLACEHOLDER,
        "label": PLACEHOLDER,
        "basis": (
            f"{area:.1f} m2 glazed area "
            f"({', '.join(f'{k}: {v:.1f} m2' for k, v in sorted(by_construction.items()))}) "
            f"x PLACEHOLDER RSMeans-style unit rate "
            f"{E2_GLAZING_UNIT_COST_USD_PER_M2_PLACEHOLDER:.0f} USD/m2 "
            "(low-e glazing + labour)"
        ),
    }


# ---------------------------------------------------------------------------
# E3 - plant efficiency
# ---------------------------------------------------------------------------
E3_CHILLER_NAME = "Central Chiller"
E3_BOILER_NAME = "Central Boiler"
E3_CHILLER_COP_BEFORE = 3.2
E3_CHILLER_COP_AFTER = 4.5
E3_BOILER_EFFICIENCY_BEFORE = 0.8
E3_BOILER_EFFICIENCY_AFTER = 0.92


def _apply_e3(idf_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Upgrade the cooling and heating plant efficiency.

    Mechanism: two scalar field edits - the ``Nominal COP`` of the electric
    chiller (3.2 -> 4.5) and the ``Nominal Thermal Efficiency`` of the gas
    boiler (0.80 -> 0.92).  Nothing else in the plant is changed (same
    equipment, same autosize behaviour, same schedules).
    """
    objects = P.parse_idf(idf_text)
    changes: List[Dict[str, Any]] = []

    for obj in objects:
        if obj.otype.lower() == "chiller:electric" and obj.fields[0].strip() == E3_CHILLER_NAME:
            old = float(obj.fields[3])
            obj.fields[3] = _fmt(E3_CHILLER_COP_AFTER)
            obj._dirty = True  # type: ignore[attr-defined]
            changes.append(
                {
                    "ecm": "E3",
                    "object": obj.otype,
                    "name": obj.fields[0].strip(),
                    "field": "Nominal COP {W/W}",
                    "old": old,
                    "new": E3_CHILLER_COP_AFTER,
                }
            )
        if obj.otype.lower() == "boiler:hotwater" and obj.fields[0].strip() == E3_BOILER_NAME:
            old = float(obj.fields[3])
            obj.fields[3] = _fmt(E3_BOILER_EFFICIENCY_AFTER)
            obj._dirty = True  # type: ignore[attr-defined]
            changes.append(
                {
                    "ecm": "E3",
                    "object": obj.otype,
                    "name": obj.fields[0].strip(),
                    "field": "Nominal Thermal Efficiency",
                    "old": old,
                    "new": E3_BOILER_EFFICIENCY_AFTER,
                }
            )
    if len(changes) != 2:
        raise ValueError(f"E3: expected 2 plant edits, got {len(changes)}")
    return P.render_idf(objects), changes


def _cost_e3(idf_text: str) -> Dict[str, Any]:
    return {
        "amount_usd": E3_PLANT_UPGRADE_COST_USD_PLACEHOLDER,
        "label": PLACEHOLDER,
        "basis": (
            "PLACEHOLDER lump sum (chiller + boiler equipment replacement, "
            "installed) for the single central plant of the example building"
        ),
    }


# ---------------------------------------------------------------------------
# E4 - electrification of heating
# ---------------------------------------------------------------------------
E4_BOILER_NAME = "Central Boiler"
E4_BOILER_EFFICIENCY_AFTER = 1.0


def _apply_e4(idf_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Switch the central boiler from natural gas to electric resistance heat.

    Mechanism: the ``Boiler:HotWater`` object keeps its name, capacity and
    nodes; its ``Fuel Type`` changes from ``NaturalGas`` to ``Electricity``
    and its ``Nominal Thermal Efficiency`` from 0.80 to 1.00.  The normalised
    boiler efficiency curve is blanked (EnergyPlus models a constant
    efficiency boiler when the curve name is empty), so the boiler is a pure
    resistance element.

    SUBSTITUTION (recorded): a full air-source heat pump swap is **not**
    implementable as an IDF edit in this slice - the shipped example has a
    hydronic plant loop whose heating is produced by a ``Boiler:HotWater``,
    and adding an air-source heat-pump plant (coil + loop + branch +
    condenser water loop) would be a different building model, not an
    energy-conservation-measure edit of this one.  The honest nearest variant
    is therefore the fuel switch to electric resistance heating.
    """
    objects = P.parse_idf(idf_text)
    changes: List[Dict[str, Any]] = []
    for obj in objects:
        if obj.otype.lower() != "boiler:hotwater":
            continue
        if obj.fields[0].strip() != E4_BOILER_NAME:
            continue
        old_fuel = obj.fields[1].strip()
        obj.fields[1] = "Electricity"
        changes.append(
            {
                "ecm": "E4",
                "object": obj.otype,
                "name": obj.fields[0].strip(),
                "field": "Fuel Type",
                "old": old_fuel,
                "new": "Electricity",
            }
        )
        old_eff = float(obj.fields[3])
        obj.fields[3] = _fmt(E4_BOILER_EFFICIENCY_AFTER)
        changes.append(
            {
                "ecm": "E4",
                "object": obj.otype,
                "name": obj.fields[0].strip(),
                "field": "Nominal Thermal Efficiency",
                "old": old_eff,
                "new": E4_BOILER_EFFICIENCY_AFTER,
            }
        )
        # constant efficiency: blank the efficiency curve reference
        if len(obj.fields) > 5 and obj.fields[5].strip():
            old_curve = obj.fields[5]
            obj.fields[4] = ""
            obj.fields[5] = ""
            changes.append(
                {
                    "ecm": "E4",
                    "object": obj.otype,
                    "name": obj.fields[0].strip(),
                    "field": "Normalized Boiler Efficiency Curve Name",
                    "old": old_curve,
                    "new": None,
                }
            )
        obj._dirty = True  # type: ignore[attr-defined]
    if not changes:
        raise ValueError("E4: central boiler not found in the IDF")
    return P.render_idf(objects), changes


def _cost_e4(idf_text: str) -> Dict[str, Any]:
    return {
        "amount_usd": E4_ELECTRIFICATION_COST_USD_PLACEHOLDER,
        "label": PLACEHOLDER,
        "basis": (
            "PLACEHOLDER lump sum (electric resistance boiler equipment + "
            "install + nominal service upgrade) for the central heating plant"
        ),
    }


# ---------------------------------------------------------------------------
# E5 - thermostat deadband widening / night setback
# ---------------------------------------------------------------------------
E5_COOLING_SCHEDULE = "Clg-SetP-Sch"
E5_HEATING_SCHEDULE = "Htg-SetP-Sch"
E5_COOLING_OFFSET_K = 1.0
E5_HEATING_OFFSET_K = -1.0


def _schedule_items(fields: Sequence[str]) -> List[Dict[str, Any]]:
    """Split a ``Schedule:Compact`` field list into renderable items.

    Each item is a dict with ``kind`` (``"for"``, ``"until"`` or ``"other"``),
    the original ``raw`` text and - for ``"until"`` items - the parsed
    ``head`` (``"Until: 6:00"``) and the numeric ``values``.
    """
    items: List[Dict[str, Any]] = []
    for raw in fields[2:]:
        value = raw.strip()
        if value.lower().startswith("for:"):
            items.append({"kind": "for", "raw": raw})
            continue
        match = P._UNTIL_RE.match(value)
        if match:
            head, tail = match.group(1), match.group(2)
            numbers: List[float] = []
            for part in tail.split():
                try:
                    numbers.append(float(part))
                except ValueError:
                    continue
            items.append({"kind": "until", "raw": raw, "head": head, "values": numbers})
            continue
        items.append({"kind": "other", "raw": raw})
    return items


def _render_schedule_items(items: Sequence[Dict[str, Any]]) -> List[str]:
    """Render schedule items back into ``Schedule:Compact`` field strings."""
    out: List[str] = []
    for item in items:
        if item["kind"] == "until":
            values = " ".join(_fmt(v) for v in item["values"])
            out.append(f"{item['head']},{values}")
        else:
            out.append(item["raw"])
    return out


def _apply_e5(idf_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Widen the thermostat deadband by 1 K on each side.

    Mechanism: in the *occupied* day-type blocks of the cooling setpoint
    schedule the occupied cooling setpoint is raised by 1.0 K (23.9 -> 24.9 K
    in the shipped schedule) and in the occupied blocks of the heating
    setpoint schedule the occupied heating setpoint is lowered by 1.0 K
    (22.2 -> 21.2 K).  The night setback values (29.4 K cooling / 16.7 K
    heating) and the plenum setpoint schedules are left untouched.

    The "occupied" value of a block is identified structurally: a block that
    contains two distinct setpoint values is an occupied block, and the
    occupied value is the *lower* one for cooling (23.9 K vs 29.4 K) and the
    *higher* one for heating (22.2 K vs 16.7 K).  This identification is
    independent of a uniform offset on the whole schedule, so the ECM
    composes with slice 1's ``cooling_setpoint_offset_k`` parameter.
    """
    objects = P.parse_idf(idf_text)
    changes: List[Dict[str, Any]] = []

    for obj in objects:
        if obj.otype.lower() != "schedule:compact":
            continue
        name = obj.fields[0].strip() if obj.fields else ""
        if name == E5_COOLING_SCHEDULE:
            offset, occupied_pick = E5_COOLING_OFFSET_K, min
        elif name == E5_HEATING_SCHEDULE:
            offset, occupied_pick = E5_HEATING_OFFSET_K, max
        else:
            continue

        items = _schedule_items(obj.fields)
        # group the "until" items per day-type block
        block_items: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for item in items:
            if item["kind"] == "for":
                current = {"day_type": item["raw"].strip(), "segments": []}
                block_items.append(current)
            elif item["kind"] == "until" and current is not None:
                current["segments"].append(item)

        for block in block_items:
            values = [v for item in block["segments"] for v in item["values"]]
            distinct = sorted(set(values))
            if len(distinct) < 2:
                # single-valued block (design day, or a night-only block):
                # there is no occupied/unoccupied pair to widen
                continue
            occupied = occupied_pick(distinct)
            for item in block["segments"]:
                new_values = []
                for num in item["values"]:
                    if abs(num - occupied) < 1e-9:
                        new_values.append(num + offset)
                        changes.append(
                            {
                                "ecm": "E5",
                                "object": obj.otype,
                                "name": name,
                                "field": f"{block['day_type']} / {item['head']}",
                                "old": num,
                                "new": num + offset,
                            }
                        )
                    else:
                        new_values.append(num)
                item["values"] = new_values

        if not any(c["name"] == name for c in changes):
            continue
        obj.fields = list(obj.fields[:2]) + _render_schedule_items(items)
        obj._dirty = True  # type: ignore[attr-defined]

    if not changes:
        raise ValueError("E5: no occupied setpoint segment found to widen")
    return P.render_idf(objects), changes


def _cost_e5(idf_text: str) -> Dict[str, Any]:
    return {
        "amount_usd": E5_CONTROLS_COST_USD_PLACEHOLDER,
        "label": PLACEHOLDER,
        "basis": (
            "PLACEHOLDER lump sum (thermostat replacement / reprogramming of "
            "the five zone thermostats of the example building)"
        ),
    }


# ---------------------------------------------------------------------------
# the five ECMs
# ---------------------------------------------------------------------------
ECMS: Tuple[Ecm, ...] = (
    Ecm(
        ecm_id="E1",
        name="Opaque envelope insulation",
        definition=(
            "Add a 50 mm insulation layer (k = 0.04 W/m-K) to the outside of "
            "every outdoor-facing opaque envelope surface of the building - the "
            "roof and the exterior walls.  The envelope area (737.8 m2 in the "
            "shipped 5ZoneAirCooled example) is taken from the IDF geometry, "
            "not hard-coded.  The layer is modelled as a new ``Material`` "
            "object inserted as the outside layer of the two envelope "
            "constructions (ROOF-1, WALL-1); the interior constructions "
            "(ceiling, interior walls, floor slab) and every interior "
            "material are left exactly as shipped.  This is deliberately a "
            "different mechanism from slice 1's blanket "
            "'envelope_conductivity_scale', which scaled every ``Material`` "
            "conductivity including carpet and ceiling tile."
        ),
        mechanism=(
            "Add Material 'E1-INSULATION-50MM' (0.05 m, 0.04 W/m-K) and "
            "insert it as the Outside Layer of every construction used by an "
            "outdoor-facing wall or roof surface."
        ),
        cost_basis=(
            "PLACEHOLDER RSMeans-style unit rate ("
            f"{E1_INSULATION_UNIT_COST_USD_PER_M2_PLACEHOLDER:.0f} USD/m2) "
            "x outdoor-facing opaque envelope area read from the IDF."
        ),
        assumptions=(
            "Adds exterior insulation only; no air-sealing, no window/air "
            "leakage change, no geometry change.  The added layer is treated "
            "as a pure conduction-resistance addition; its solar absorptance "
            "(0.6) slightly darkens the roof and wall surfaces, which is "
            "reflected in the simulation rather than corrected for.  Cost is "
            "a PLACEHOLDER and excludes scaffolding, roof penetrations, "
            "interior finish repair and any incentives."
        ),
        grid_flexibility=(
            "Low: the envelope is a passive measure with no dispatchable "
            "component; it reduces baseload but cannot be shifted in time."
        ),
        apply_fn=_apply_e1,
        cost_fn=_cost_e1,
    ),
    Ecm(
        ecm_id="E2",
        name="Glazing upgrade (low-e)",
        definition=(
            "Replace the window and glass-door glazing of the building with a "
            "low-emissivity system.  The two glazed constructions of the "
            "shipped example ('Dbl Clr 3mm/13mm Air', double clear glazing "
            "with a 13 mm air gap, and 'Sgl Grey 3mm', single grey glazing) "
            "are each reduced to a single "
            "``WindowMaterial:SimpleGlazingSystem`` layer with an explicit "
            "U-factor of 1.60 W/m2-K, an SHGC of 0.30 and a visible "
            "transmittance of 0.55.  The before U-factor of the shipped "
            "glazing is the one EnergyPlus reports for the baseline model; "
            "the after U-factor is 1.60 W/m2-K by construction of the edit. "
            "The 60.9 m2 of glazed area (51.2 m2 of windows plus 9.7 m2 of "
            "glass doors) is read from the IDF geometry."
        ),
        mechanism=(
            "Add WindowMaterial:SimpleGlazingSystem 'E2-LOW-E' (U 1.60 W/m2-K, "
            "SHGC 0.30, VT 0.55) and replace the layer list of every window "
            "construction with that single layer."
        ),
        cost_basis=(
            "PLACEHOLDER RSMeans-style unit rate ("
            f"{E2_GLAZING_UNIT_COST_USD_PER_M2_PLACEHOLDER:.0f} USD/m2) x "
            "glazed area read from the IDF."
        ),
        assumptions=(
            "Glazing only: frames, spacers, shutters and skylights are not "
            "modelled, and the simple glazing system represents the glazing "
            "as a single layer without a frame, so the resulting U-factor is "
            "a nominal centre-of-glass value.  Daylight-driven lighting "
            "response is not modelled (the example has no daylighting "
            "controls).  Cost is a PLACEHOLDER."
        ),
        grid_flexibility=(
            "Low: passive reduction of solar and conductive gains; it lowers "
            "the cooling peak but offers no controllable load-shift channel."
        ),
        apply_fn=_apply_e2,
        cost_fn=_cost_e2,
    ),
    Ecm(
        ecm_id="E3",
        name="HVAC plant efficiency upgrade",
        definition=(
            "Upgrade the central heating and cooling plant: the electric "
            "chiller nominal COP goes from 3.2 to 4.5 W/W and the gas boiler "
            "nominal thermal efficiency from 0.80 to 0.92.  Equipment, "
            "autosize behaviour, schedules, air loops and controls are "
            "untouched; only the two efficiency scalars change.  This is the "
            "cheapest-savings-per-dollar ECM of the set in most retrofit "
            "practice, which is exactly the prior the LLM control is expected "
            "to have."
        ),
        mechanism=(
            "Two scalar field edits: Chiller:Electric 'Central Chiller' "
            "Nominal COP 3.2 -> 4.5; Boiler:HotWater 'Central Boiler' "
            "Nominal Thermal Efficiency 0.80 -> 0.92."
        ),
        cost_basis=(
            "PLACEHOLDER lump sum for replacing the single chiller and the "
            "single boiler (equipment + installation)."
        ),
        assumptions=(
            "Constant-efficiency equipment: the upgraded chiller keeps the "
            "same part-load curves, so the COP gain is applied across the "
            "whole part-load range; the boiler keeps its normalised "
            "efficiency curve.  No changes to pumping, controls, condenser "
            "water or plant sizing.  Cost is a PLACEHOLDER."
        ),
        grid_flexibility=(
            "Low to medium: better plant efficiency lowers the energy needed "
            "per unit of cooling, but the plant still follows the same "
            "dispatch and offers no explicit load-shift control."
        ),
        apply_fn=_apply_e3,
        cost_fn=_cost_e3,
    ),
    Ecm(
        ecm_id="E4",
        name="Heating electrification (fuel switch)",
        definition=(
            "Electrify the central heating plant by switching the "
            "``Boiler:HotWater`` object from natural gas to electricity and "
            "from 80% to 100% nominal thermal efficiency, i.e. an electric "
            "resistance boiler.  The recorded SUBSTITUTION: a full "
            "air-source heat-pump swap is *not* implementable as an IDF edit "
            "in this slice, because the shipped example heats the zones "
            "through a hydronic plant loop whose heat is produced by a "
            "``Boiler:HotWater``; adding an air-source heat pump would mean "
            "adding a coil, a loop, a branch and a condenser-water loop - a "
            "different building model rather than an ECM edit of this one. "
            "The honest nearest variant is therefore the fuel switch, and it "
            "is recorded as such.  Note the consequence for the metrics: "
            "site energy in kWh falls (gas is replaced by electricity at a "
            "better efficiency) while the cost of that energy usually rises, "
            "so this ECM is a good test of whether the ranking metric is "
            "honest."
        ),
        mechanism=(
            "Boiler:HotWater 'Central Boiler': Fuel Type NaturalGas -> "
            "Electricity; Nominal Thermal Efficiency 0.80 -> 1.00; the "
            "normalised efficiency curve reference is blanked (constant "
            "efficiency boiler)."
        ),
        cost_basis=(
            "PLACEHOLDER lump sum for an electric resistance boiler plus "
            "installation plus a nominal electrical service upgrade."
        ),
        assumptions=(
            "No heat pump, no change to the chilled-water plant, no change to "
            "emission factors beyond the stated placeholders, and no "
            "electrical tariff optimisation (no time-of-use shifting is "
            "modelled).  The boiler is autosized as before.  Cost is a "
            "PLACEHOLDER."
        ),
        grid_flexibility=(
            "Medium: resistance heating is fully dispatchable and can be "
            "curtailed or shifted, but it also adds winter peak load to the "
            "electricity system rather than removing it."
        ),
        apply_fn=_apply_e4,
        cost_fn=_cost_e4,
    ),
    Ecm(
        ecm_id="E5",
        name="Thermostat deadband widening (setback)",
        definition=(
            "Widen the thermostat deadband by 1.0 K on each side: the "
            "occupied cooling setpoint is raised by 1.0 K (23.9 -> 24.9 K in "
            "the shipped schedule) and the occupied heating setpoint is "
            "lowered by 1.0 K (22.2 -> 21.2 K), so the occupied deadband goes "
            "from 1.7 K to 3.7 K.  Only the occupied day-type blocks are "
            "edited: the night setback values (29.4 K cooling / 16.7 K "
            "heating), the design days and the plenum reheat setpoint "
            "schedules ('PlenumClg-SetP-Sch' 40.0 K, 'PlenumHtg-SetP-Sch' "
            "12.8 K) are untouched.  The occupied value of a block is "
            "identified structurally (the block that carries both an "
            "occupied and an unoccupied value), so the edit composes with "
            "slice 1's calibration parameter ``cooling_setpoint_offset_k``."
        ),
        mechanism=(
            "Edit the values of the 'Clg-SetP-Sch' (+1.0 K) and "
            "'Htg-SetP-Sch' (-1.0 K) Schedule:Compact objects in their "
            "occupied blocks only."
        ),
        cost_basis=(
            "PLACEHOLDER lump sum for replacing/reprogramming the five zone "
            "thermostats of the example building."
        ),
        assumptions=(
            "Setpoint-only: no change to the thermostat hardware, airflow, "
            "reheat coil control or ventilation schedules; occupants are "
            "assumed to accept the widened deadband.  The zone temperature is "
            "reported as a comfort check for every ECM run, separately from "
            "the energy result.  Cost is a PLACEHOLDER."
        ),
        grid_flexibility=(
            "High: a widened deadband with setback is the cheapest dispatch "
            "channel of the set - it can be pre-empted, curtailed or shifted "
            "by a controller or a grid signal."
        ),
        apply_fn=_apply_e5,
        cost_fn=_cost_e5,
    ),
)

ECM_IDS: Tuple[str, ...] = tuple(ecm.ecm_id for ecm in ECMS)


def get_ecm(ecm_id: str) -> Ecm:
    for ecm in ECMS:
        if ecm.ecm_id == ecm_id:
            return ecm
    raise KeyError(f"unknown ECM id {ecm_id!r}; known: {list(ECM_IDS)}")
