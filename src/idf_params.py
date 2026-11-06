"""Minimal IDF object model plus the three bounded calibration parameters.

The example IDF shipped with EnergyPlus is the *only* model used in this POC.
This module never edits the shipped file in place: it parses it, applies a
named parameter change to a copy, and returns the new text together with a
record of every value that changed (so each run can be audited).

The three named parameters are:

``infiltration_flow_scale``
    Multiplies the *Design Flow Rate* (m3/s) of every
    ``ZoneInfiltration:DesignFlowRate`` object.  1.0 = as shipped.
``envelope_conductivity_scale``
    Multiplies the *Conductivity* (W/m-K) of every ``Material`` object, i.e.
    scales the opaque envelope U-value by the same factor.  1.0 = as shipped.
``cooling_setpoint_offset_k``
    Shifts every temperature value of the ``Clg-SetP-Sch`` schedule by the
    given number of kelvin.  0.0 = as shipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PARAM_NAMES = (
    "infiltration_flow_scale",
    "envelope_conductivity_scale",
    "cooling_setpoint_offset_k",
)

DEFAULT_PARAMS: Dict[str, float] = {
    "infiltration_flow_scale": 1.0,
    "envelope_conductivity_scale": 1.0,
    "cooling_setpoint_offset_k": 0.0,
}


@dataclass
class IdfObject:
    otype: str
    fields: List[str]
    raw: str

    @property
    def name(self) -> str:
        return self.fields[0] if self.fields else ""


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("!")[0] for line in text.splitlines())


_UNTIL_TOKEN_RE = re.compile(r"Until:\s*\d{1,2}:\d{2}\s*,")


def split_fields(body: str) -> List[str]:
    """Split an IDF object body on commas.

    ``Schedule:Compact`` fields legitimately contain a comma
    (``Until: 24:00,23.9``), so those are kept together.
    """
    parts: List[str] = []
    buf = ""
    i = 0
    while i < len(body):
        match = _UNTIL_TOKEN_RE.match(body, i)
        if match:
            buf += match.group(0)
            i = match.end()
            continue
        if body[i] == ",":
            parts.append(buf.strip())
            buf = ""
            i += 1
            continue
        buf += body[i]
        i += 1
    parts.append(buf.strip())
    return parts


def parse_idf(text: str) -> List[IdfObject]:
    """Split an IDF into objects.  Comments outside objects are preserved in
    the original text of unmodified objects only."""
    objects: List[IdfObject] = []
    block: List[str] = []
    for line in text.splitlines():
        block.append(line)
        if line.split("!")[0].strip().endswith(";"):
            raw = "\n".join(block)
            body = _strip_comments(raw).strip()
            if body:
                parts = split_fields(body.rstrip(";").strip())
                otype = parts[0]
                fields = parts[1:]
                objects.append(IdfObject(otype=otype, fields=fields, raw=raw))
            block = []
    return objects


def render_object(obj: IdfObject) -> str:
    lines = [f"  {obj.otype},"]
    for i, val in enumerate(obj.fields):
        terminator = ";" if i == len(obj.fields) - 1 else ","
        lines.append(f"    {val}{terminator}")
    return "\n".join(lines)


def render_idf(objects: List[IdfObject]) -> str:
    """Re-emit the IDF, keeping the original raw text of untouched objects."""
    chunks: List[str] = []
    for obj in objects:
        chunks.append(obj.raw if not getattr(obj, "_dirty", False) else render_object(obj))
    return "\n".join(chunks)


def _to_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return None


def _fmt(value: float) -> str:
    return f"{value:.6g}"


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def scale_infiltration(
    objects: List[IdfObject], factor: float
) -> Tuple[List[IdfObject], List[Dict[str, Any]]]:
    changes: List[Dict[str, Any]] = []
    for obj in objects:
        if obj.otype.lower() != "zoneinfiltration:designflowrate":
            continue
        if len(obj.fields) < 5:
            continue
        method = obj.fields[3].strip().lower()
        if method != "flow/zone":
            continue
        old = _to_float(obj.fields[4])
        if old is None:
            continue
        new = old * factor
        obj.fields[4] = _fmt(new)
        obj._dirty = True  # type: ignore[attr-defined]
        changes.append(
            {
                "parameter": "infiltration_flow_scale",
                "object": obj.otype,
                "name": obj.name,
                "field": "Design Flow Rate {m3/s}",
                "old": old,
                "new": new,
            }
        )
    return objects, changes


def scale_material_conductivity(
    objects: List[IdfObject], factor: float
) -> Tuple[List[IdfObject], List[Dict[str, Any]]]:
    changes: List[Dict[str, Any]] = []
    for obj in objects:
        if obj.otype.lower() != "material":
            continue
        if len(obj.fields) < 4:
            continue
        old = _to_float(obj.fields[3])
        if old is None:
            continue
        new = old * factor
        obj.fields[3] = _fmt(new)
        obj._dirty = True  # type: ignore[attr-defined]
        changes.append(
            {
                "parameter": "envelope_conductivity_scale",
                "object": obj.otype,
                "name": obj.name,
                "field": "Conductivity {W/m-K}",
                "old": old,
                "new": new,
            }
        )
    return objects, changes


_UNTIL_RE = re.compile(r"^(Until:\s*\d{1,2}:\d{2})\s*,\s*(.+)$")


def shift_schedule_values(
    objects: List[IdfObject], schedule_name: str, offset: float
) -> Tuple[List[IdfObject], List[Dict[str, Any]]]:
    changes: List[Dict[str, Any]] = []
    for obj in objects:
        if obj.otype.lower() != "schedule:compact":
            continue
        if not obj.fields or obj.fields[0].strip() != schedule_name:
            continue
        for i, val in enumerate(obj.fields):
            match = _UNTIL_RE.match(val.strip())
            if not match:
                continue
            head, tail = match.group(1), match.group(2)
            parts = tail.split()
            new_parts: List[str] = []
            touched = False
            for part in parts:
                num = _to_float(part)
                if num is None:
                    new_parts.append(part)
                    continue
                new_val = num + offset
                new_parts.append(_fmt(new_val))
                touched = True
                changes.append(
                    {
                        "parameter": "cooling_setpoint_offset_k",
                        "object": obj.otype,
                        "name": obj.name,
                        "field": f"field {i} ({head})",
                        "old": num,
                        "new": new_val,
                    }
                )
            if touched:
                obj.fields[i] = f"{head},{' '.join(new_parts)}"
                obj._dirty = True  # type: ignore[attr-defined]
    return objects, changes


def ensure_output_meters(
    objects: List[IdfObject], meters: Tuple[str, ...] = (
        "Electricity:Facility",
        "DistrictHeating:Facility",
        "DistrictCooling:Facility",
        "NaturalGas:Facility",
    )
) -> Tuple[List[IdfObject], List[Dict[str, Any]]]:
    """Add hourly ``Output:Meter`` objects if they are not already present."""
    existing = set()
    for obj in objects:
        if obj.otype.lower() == "output:meter":
            existing.add(obj.fields[0].strip().lower())
    changes: List[Dict[str, Any]] = []
    for meter in meters:
        if meter.lower() in existing:
            continue
        obj = IdfObject(
            otype="Output:Meter",
            fields=[meter, "Hourly"],
            raw="",
        )
        obj._dirty = True  # type: ignore[attr-defined]
        objects.append(obj)
        changes.append(
            {
                "parameter": "reporting",
                "object": "Output:Meter",
                "name": meter,
                "field": "Frequency",
                "old": None,
                "new": "Hourly",
            }
        )
    return objects, changes


def apply_params(
    idf_text: str,
    params: Dict[str, float],
    schedule_name: str = "Clg-SetP-Sch",
) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply the named parameters to an IDF text.

    Returns the new IDF text and the list of individual value changes.
    """
    unknown = set(params) - set(PARAM_NAMES)
    if unknown:
        raise ValueError(f"unknown calibration parameters: {sorted(unknown)}")

    objects = parse_idf(idf_text)
    changes: List[Dict[str, Any]] = []

    objects, ch = ensure_output_meters(objects)
    changes.extend(ch)

    factor = float(params.get("infiltration_flow_scale", 1.0))
    if factor != 1.0:
        objects, ch = scale_infiltration(objects, factor)
        changes.extend(ch)

    factor = float(params.get("envelope_conductivity_scale", 1.0))
    if factor != 1.0:
        objects, ch = scale_material_conductivity(objects, factor)
        changes.extend(ch)

    offset = float(params.get("cooling_setpoint_offset_k", 0.0))
    if offset != 0.0:
        objects, ch = shift_schedule_values(objects, schedule_name, offset)
        changes.extend(ch)

    return render_idf(objects), changes
