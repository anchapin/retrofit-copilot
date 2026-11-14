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

Scope of the conductivity scaling
----------------------------------
``envelope_conductivity_scale_scope`` is an *opt-in* sibling key that selects
how widely ``envelope_conductivity_scale`` is applied:

``"blanket"`` (the default)
    Every ``Material`` conductivity is scaled -- the slice-1 behaviour.  This
    stays the default so that every recorded slice-1/slice-2 result still
    reproduces bit-identically.
``"envelope"``
    Only the ``Material`` objects that are layers of the constructions used by
    outdoor-facing opaque surfaces are scaled.  The construction selection is
    E1's (see :func:`ecms.opaque_envelope_area`): outdoor-facing walls and
    roofs only, so interior constructions (ceiling, interior wall, floor
    slab) and the materials they use are left untouched.

The scope key is *not* part of :data:`PARAM_NAMES`: it is a string-valued
switch, not a calibrated numeric parameter, so it never enters a calibration
grid and is never written to (or read from) a recorded results file.
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

#: Name of the opt-in scope switch for ``envelope_conductivity_scale``.
#:
#: It is deliberately *not* part of :data:`PARAM_NAMES`: it is a
#: string-valued switch, not a calibrated numeric parameter, so it never
#: enters a calibration grid and is never read from (or written to) a
#: recorded results file.
SCOPE_PARAM_NAME = "envelope_conductivity_scale_scope"

#: Scale the conductivity of *every* ``Material`` (the slice-1 behaviour).
SCOPE_BLANKET = "blanket"
#: Scale only the materials of the constructions used by outdoor-facing
#: opaque surfaces (E1's construction selection).
SCOPE_ENVELOPE = "envelope"

#: The valid scope values.
CONDUCTIVITY_SCOPES = (SCOPE_BLANKET, SCOPE_ENVELOPE)
#: The default scope: blanket, so the recorded slice-1/slice-2 results
#: reproduce bit-identically.
DEFAULT_CONDUCTIVITY_SCOPE = SCOPE_BLANKET


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
    objects: List[IdfObject],
    factor: float,
    material_names: Optional[List[str]] = None,
) -> Tuple[List[IdfObject], List[Dict[str, Any]]]:
    """Scale the conductivity of the ``Material`` objects.

    With ``material_names`` given (``None`` = blanket), only the ``Material``
    objects whose name is listed are scaled; every other ``Material`` (an
    interior carpet, a ceiling tile, ...) is left exactly as shipped.
    """
    wanted = None if material_names is None else set(material_names)
    changes: List[Dict[str, Any]] = []
    for obj in objects:
        if obj.otype.lower() != "material":
            continue
        if len(obj.fields) < 4:
            continue
        if wanted is not None and obj.name.strip() not in wanted:
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


def envelope_construction_names(objects: List[IdfObject]) -> List[str]:
    """Names of the constructions used by the outdoor-facing opaque envelope.

    This is E1's construction selection, reused as-is: the constructions
    referenced by outdoor-facing wall and roof surfaces (see
    :func:`ecms.opaque_envelope_area`).  Interior constructions (ceiling,
    interior wall, floor slab) are therefore never selected.
    """
    # Imported lazily: ``ecms`` imports this module, so a module-level
    # import here would be circular.
    from ecms import opaque_envelope_area

    _, by_construction = opaque_envelope_area(objects)
    return sorted(by_construction)


def envelope_material_names(objects: List[IdfObject]) -> List[str]:
    """Names of the ``Material`` objects used by the outdoor-facing envelope.

    The materials are the layers of the constructions selected by
    :func:`envelope_construction_names` (E1's selection).  A material that is
    also used by an interior construction is still returned: it *is* part of
    the envelope.
    """
    constructions = set(envelope_construction_names(objects))
    names: List[str] = []
    for obj in objects:
        if obj.otype.lower() != "construction":
            continue
        if not obj.fields:
            continue
        if obj.fields[0].strip() not in constructions:
            continue
        for layer in obj.fields[1:]:
            layer = layer.strip()
            if layer and layer not in names:
                names.append(layer)
    return names


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
    params: Dict[str, Any],
    schedule_name: str = "Clg-SetP-Sch",
    envelope_conductivity_scale_scope: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply the named parameters to an IDF text.

    Returns the new IDF text and the list of individual value changes.

    The conductivity scale is *blanket* (every ``Material``) unless the
    opt-in scope is set to ``"envelope"``.  The scope may be given either as
    the ``envelope_conductivity_scale_scope`` key of ``params`` or as the
    ``envelope_conductivity_scale_scope`` argument (the argument wins).
    """
    unknown = set(params) - set(PARAM_NAMES) - {SCOPE_PARAM_NAME}
    if unknown:
        raise ValueError(f"unknown calibration parameters: {sorted(unknown)}")

    scope = envelope_conductivity_scale_scope
    if scope is None:
        scope = params.get(SCOPE_PARAM_NAME, DEFAULT_CONDUCTIVITY_SCOPE)
    scope = str(scope).strip().lower()
    if scope not in CONDUCTIVITY_SCOPES:
        raise ValueError(
            f"unknown {SCOPE_PARAM_NAME}: {scope!r} "
            f"(known scopes: {', '.join(CONDUCTIVITY_SCOPES)})"
        )

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
        if scope == SCOPE_ENVELOPE:
            names = envelope_material_names(objects)
            if not names:
                raise ValueError(
                    "envelope_conductivity_scale_scope='envelope': no outdoor-facing "
                    "opaque envelope materials found in the IDF"
                )
            objects, ch = scale_material_conductivity(objects, factor, names)
        else:
            objects, ch = scale_material_conductivity(objects, factor)
        changes.extend(ch)

    offset = float(params.get("cooling_setpoint_offset_k", 0.0))
    if offset != 0.0:
        objects, ch = shift_schedule_values(objects, schedule_name, offset)
        changes.extend(ch)

    return render_idf(objects), changes
