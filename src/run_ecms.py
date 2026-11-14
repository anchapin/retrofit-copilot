"""Slice-2 driver: run the 6 ECMs (E1-E5 plus the E4HP heat-pump variant of
E4) on the calibrated model AND the true model,
rank them, and compare the simulation-grounded ranking with the LLM prior.

    python src/run_ecms.py --all

The energy prices and emission factors used for the cost and carbon columns
come from a JSON *factors file*; the default is the shipped
``config/factors.placeholder.json`` (the PLACEHOLDER factor set) and any
other file may be passed with ``--factors <path.json>``:

    python src/run_ecms.py --all --factors my_factors.json

The file must be a JSON object with exactly these four keys (each a finite,
non-negative number)::

    {
      "electricity_price_usd_per_kwh": 0.13,
      "natural_gas_price_usd_per_kwh": 0.06,
      "electricity_co2_kg_per_kwh": 0.37,
      "natural_gas_co2_kg_per_kwh": 0.18
    }

The source path of the factor set is recorded in the headline table and in
``ecm_results.json``; the factor-derived columns only carry the PLACEHOLDER
marker when the shipped placeholder factor set is used.

The headline ranking is by annual facility kWh saved by default; another
metric can be chosen with ``--rank-by {kwh,cost,carbon}`` (annual USD
saved or annual tCO2 saved - both already computed by the pipeline).  The
chosen metric is named in the ranking headings and ECMs with negative
savings sort last:

    python src/run_ecms.py --all --rank-by cost

Fourteen EnergyPlus runs are recorded (2 model baselines + 6 ECMs x 2 models);
every run gets a run id, a run directory and a manifest under
``runs/``.  The LLM-prior control
(``runs/llm_prior_ranking.json``) must exist before the
first simulation of this script - the script refuses to run without it and
records its hash and its mtime next to the run ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ecms as E  # noqa: E402
import idf_params as P  # noqa: E402
import results as RES  # noqa: E402
import run_energyplus as R  # noqa: E402
import run_poc as POC  # noqa: E402

POC_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = POC_ROOT / "runs"
IDF_NAME = "5ZoneAirCooled.idf"
EPW_NAME = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
PRIOR_FILE = RUNS_DIR / "llm_prior_ranking.json"

#: The canonical file names used in the pre-flight error messages (the
#: messages always name the file, even when a caller points the runner at a
#: file under a different name).
PRIOR_FILE_NAME = "runs/llm_prior_ranking.json"
RESULTS_FILE_NAME = "results.json"
RESULTS_FILE_LABEL = "runs/results.json"


class PreflightError(SystemExit, ValueError):
    """A pre-flight input problem, reported as a clean one-line exit.

    It derives from :class:`SystemExit` so the command line scripts stop with
    the message and no traceback, and from :class:`ValueError` so library
    callers that use the validation helpers directly still see an ordinary
    exception.
    """


#: The calibrated (agent) model parameters recovered by slice 1.  They are
#: re-derived from ``runs/results.json`` when that file exists (the slice-1
#: pipeline writes it); the constant below is the recorded slice-1 result.
CALIBRATED_PARAMS: Dict[str, float] = {
    "infiltration_flow_scale": 2.2,
    "envelope_conductivity_scale": 1.2,
    "cooling_setpoint_offset_k": 1.0,
}

#: The true (ground truth) model = slice 1's synthetic "measured" parameters.
TRUE_PARAMS: Dict[str, float] = dict(POC.PERTURBED_PARAMS)

#: Comfort band used for the comfort check (degC).  This is a *comfort band
#: proxy*, not a setpoint-based unmet-load metric: EnergyPlus' zone air
#: temperature output is compared against this band.
COMFORT_BAND_C = (20.0, 26.0)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_install_dir(binary: Path) -> Path:
    cand = binary.parent
    if (cand / "ExampleFiles").is_dir():
        return cand
    raise FileNotFoundError(f"no ExampleFiles/ next to the EnergyPlus binary at {binary}")


def resolve_input_paths(
    install: Path, idf_name: str, epw_name: str
) -> Tuple[Path, Path]:
    """Resolve the building and weather file names against the install.

    The names are resolved exactly like the module constants have always
    been resolved: the IDF against ``install/ExampleFiles`` and the EPW
    against ``install/WeatherData``.  An absolute path passed as a name is
    used as given.
    """
    return (
        install / "ExampleFiles" / idf_name,
        install / "WeatherData" / epw_name,
    )


def parse_prior_ranking(prior_text: str, prior_path: Any = None) -> List[str]:
    """Parse and validate the LLM-prior control file.

    Returns the list of ECM ids in the prior ranking.  Every problem is
    reported as a one-line :class:`PreflightError` (a ``SystemExit``) that
    names the file and the problem, instead of a traceback.

    The validated content is: a JSON object with a ``ranking`` list whose
    entries are objects carrying an ``ecm_id`` naming one of the known ECMs.
    """
    where = f" ({prior_path})" if prior_path is not None else ""
    try:
        prior = json.loads(prior_text)
    except json.JSONDecodeError as exc:
        raise PreflightError(
            f"invalid LLM prior file {PRIOR_FILE_NAME}{where}: "
            f"not valid JSON ({exc})"
        ) from None
    if not isinstance(prior, dict):
        raise PreflightError(
            f"invalid LLM prior file {PRIOR_FILE_NAME}{where}: "
            "the top-level value is not a JSON object"
        )
    if "ranking" not in prior:
        raise PreflightError(
            f"invalid LLM prior file {PRIOR_FILE_NAME}{where}: "
            "missing the 'ranking' key"
        )
    ranking = prior["ranking"]
    if not isinstance(ranking, list):
        raise PreflightError(
            f"invalid LLM prior file {PRIOR_FILE_NAME}{where}: "
            "'ranking' is not a list"
        )
    prior_ranking: List[str] = []
    for index, entry in enumerate(ranking):
        if not isinstance(entry, dict):
            raise PreflightError(
                f"invalid LLM prior file {PRIOR_FILE_NAME}{where}: "
                f"ranking entry {index} is not a JSON object"
            )
        if "ecm_id" not in entry:
            raise PreflightError(
                f"invalid LLM prior file {PRIOR_FILE_NAME}{where}: "
                f"ranking entry {index} has no 'ecm_id' key"
            )
        ecm_id = entry["ecm_id"]
        if ecm_id not in E.ECM_IDS:
            raise PreflightError(
                f"invalid LLM prior file {PRIOR_FILE_NAME}{where}: "
                f"unknown ECM id {ecm_id!r} in ranking entry {index} "
                f"(known ids: {', '.join(E.ECM_IDS)})"
            )
        prior_ranking.append(ecm_id)
    return prior_ranking


def load_calibrated_params(results_path: Optional[Path] = None) -> Dict[str, float]:
    """Use the slice-1 results file when present, else the recorded values.

    A present results file is validated: it must be a JSON object with a
    ``calibrated_params`` object whose keys are known parameter names
    (:data:`idf_params.PARAM_NAMES`) and whose values are numbers.  Problems
    are reported as a one-line :class:`PreflightError` (a ``SystemExit``)
    naming ``runs/results.json`` and the offending key.  An absent results
    file is not an error: the recorded ``CALIBRATED_PARAMS`` are used.
    """
    path = Path(results_path) if results_path is not None else RUNS_DIR / RESULTS_FILE_NAME
    if not path.is_file():
        return dict(CALIBRATED_PARAMS)
    where = f" ({path})"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PreflightError(
            f"invalid results file {RESULTS_FILE_LABEL}{where}: "
            f"not valid JSON ({exc})"
        ) from None
    if not isinstance(data, dict):
        raise PreflightError(
            f"invalid results file {RESULTS_FILE_LABEL}{where}: "
            "the top-level value is not a JSON object"
        )
    params = data.get("calibrated_params")
    if not params:
        return dict(CALIBRATED_PARAMS)
    if not isinstance(params, dict):
        raise PreflightError(
            f"invalid results file {RESULTS_FILE_LABEL}{where}: "
            "'calibrated_params' is not a JSON object"
        )
    unknown = [k for k in params if k not in P.PARAM_NAMES]
    if unknown:
        raise PreflightError(
            f"invalid results file {RESULTS_FILE_LABEL}{where}: "
            f"unknown parameter name(s) {', '.join(repr(k) for k in unknown)} "
            f"(known names: {', '.join(P.PARAM_NAMES)})"
        )
    out: Dict[str, float] = {}
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PreflightError(
                f"invalid results file {RESULTS_FILE_LABEL}{where}: "
                f"parameter {key!r} is not a number (got {value!r})"
            )
        out[key] = float(value)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def fuel_kwh(series: RES.RunSeries) -> Dict[str, float]:
    """Annual energy per fuel [kWh] from the recorded meters."""
    out = {"electricity": 0.0, "natural_gas": 0.0}
    found = False
    if series.has("Electricity:Facility [J](Hourly)"):
        out["electricity"] = float(series.column("Electricity:Facility [J](Hourly)").sum()) / RES.JOULES_PER_KWH
        found = True
    if series.has("NaturalGas:Facility [J](Hourly)"):
        out["natural_gas"] = float(series.column("NaturalGas:Facility [J](Hourly)").sum()) / RES.JOULES_PER_KWH
        found = True
    if not found:
        raise KeyError(
            "no hourly facility fuel meter found in the run outputs; "
            "requested meters are missing"
        )
    out["total"] = out["electricity"] + out["natural_gas"]
    return out


def comfort_metrics(series: RES.RunSeries) -> Dict[str, Any]:
    """Comfort check of a run: zone temperatures and load-activity hours."""
    zone_cols = sorted(
        name
        for name in series.columns
        if ":Zone Air Temperature [C](Hourly)" in name and "SPACE" in name
    )
    if not zone_cols:
        raise KeyError("no zone air temperature columns found in the run outputs")
    stacked = [series.column(name) for name in zone_cols]
    mean_temp = sum(stacked) / len(stacked)
    low, high = COMFORT_BAND_C
    outside = 0
    for col in stacked:
        outside += int(((col < low) | (col > high)).sum())
    heating_hours = 0
    cooling_hours = 0
    heat_cols = [n for n in series.columns if "Zone Air System Sensible Heating Rate [W](Hourly)" in n]
    cool_cols = [n for n in series.columns if "Zone Air System Sensible Cooling Rate [W](Hourly)" in n]
    if heat_cols:
        heating_hours = int(sum((series.column(n) > 0).sum() for n in heat_cols))
    if cool_cols:
        cooling_hours = int(sum((series.column(n) > 0).sum() for n in cool_cols))
    return {
        "zone_mean_air_temp_c": float(mean_temp.mean()),
        "zone_min_air_temp_c": float(min(float(c.min()) for c in stacked)),
        "zone_max_air_temp_c": float(max(float(c.max()) for c in stacked)),
        "zone_hours_outside_comfort_band": outside,
        "zone_hours_outside_comfort_band_band_c": [low, high],
        "zone_heating_active_hours": heating_hours,
        "zone_cooling_active_hours": cooling_hours,
        "zones": len(zone_cols),
    }


def ecm_metrics(
    baseline: RES.RunSeries,
    ecm_series: RES.RunSeries,
    placeholder_cost_usd: float,
    factors: Optional[E.Factors] = None,
) -> Dict[str, Any]:
    """Savings, cost, CO2, payback and comfort of one ECM against a baseline.

    ``factors`` carries the energy prices and emission factors used for the
    cost and carbon columns; it defaults to the shipped PLACEHOLDER factor
    set (``config/factors.placeholder.json``).  The metric keys are neutral
    (``cost_saved_usd`` / ``co2_saved_tco2``); the legacy ``*_placeholder``
    keys are only emitted when the factors really are the PLACEHOLDER set.
    """
    if factors is None:
        factors = E.default_factors()
    base_fuel = fuel_kwh(baseline)
    ecm_fuel = fuel_kwh(ecm_series)
    d_elec = base_fuel["electricity"] - ecm_fuel["electricity"]
    d_gas = base_fuel["natural_gas"] - ecm_fuel["natural_gas"]
    saved_kwh = base_fuel["total"] - ecm_fuel["total"]
    saved_pct = 100.0 * saved_kwh / base_fuel["total"] if base_fuel["total"] else 0.0
    cost_saved = (
        d_elec * factors.electricity_price_usd_per_kwh
        + d_gas * factors.natural_gas_price_usd_per_kwh
    )
    co2_kg = (
        d_elec * factors.electricity_co2_kg_per_kwh
        + d_gas * factors.natural_gas_co2_kg_per_kwh
    )
    payback = placeholder_cost_usd / cost_saved if cost_saved > 0 else None
    base_comfort = comfort_metrics(baseline)
    ecm_comfort = comfort_metrics(ecm_series)
    metrics: Dict[str, Any] = {
        "baseline_annual_kwh": base_fuel["total"],
        "baseline_electricity_kwh": base_fuel["electricity"],
        "baseline_natural_gas_kwh": base_fuel["natural_gas"],
        "ecm_annual_kwh": ecm_fuel["total"],
        "ecm_electricity_kwh": ecm_fuel["electricity"],
        "ecm_natural_gas_kwh": ecm_fuel["natural_gas"],
        "energy_saved_kwh": saved_kwh,
        "energy_saved_pct": saved_pct,
        "electricity_saved_kwh": d_elec,
        "natural_gas_saved_kwh": d_gas,
        "cost_saved_usd": cost_saved,
        "co2_saved_tco2": co2_kg / 1000.0,
        "simple_payback_years": payback,
        "factors_source": factors.source,
        "factors_label": factors.label,
        "comfort": {
            "baseline": base_comfort,
            "ecm": ecm_comfort,
            "delta_zone_mean_air_temp_c": ecm_comfort["zone_mean_air_temp_c"]
            - base_comfort["zone_mean_air_temp_c"],
            "delta_zone_hours_outside_comfort_band": ecm_comfort["zone_hours_outside_comfort_band"]
            - base_comfort["zone_hours_outside_comfort_band"],
            "delta_zone_heating_active_hours": ecm_comfort["zone_heating_active_hours"]
            - base_comfort["zone_heating_active_hours"],
            "delta_zone_cooling_active_hours": ecm_comfort["zone_cooling_active_hours"]
            - base_comfort["zone_cooling_active_hours"],
        },
    }
    if factors.is_placeholder:
        # the PLACEHOLDER marker may only be carried by PLACEHOLDER numbers
        metrics["cost_saved_usd_placeholder"] = cost_saved
        metrics["co2_saved_tco2_placeholder"] = co2_kg / 1000.0
    # the payback numerator is a PLACEHOLDER installed cost, so that column
    # always keeps the marker
    metrics["simple_payback_years_placeholder"] = payback
    return metrics


#: The metrics the headline table can be ranked by.  Each entry maps the
#: CLI name to the per-ECM metric key, the heading phrase and the number
#: format used for the ranking column.
RANK_METRICS: Dict[str, Dict[str, str]] = {
    "kwh": {
        "key": "energy_saved_kwh",
        "heading": "annual facility kWh saved",
        "column": "kWh saved",
        "format": "{:.1f}",
    },
    "cost": {
        "key": "cost_saved_usd",
        "heading": "annual USD saved",
        "column": "USD saved",
        "format": "{:.2f}",
    },
    "carbon": {
        "key": "co2_saved_tco2",
        "heading": "annual tCO2 saved",
        "column": "tCO2 saved",
        "format": "{:.3f}",
    },
}

#: The accepted values of ``--rank-by``.
RANK_BY_CHOICES: Tuple[str, ...] = tuple(RANK_METRICS)


def rank_metric(rank_by: str = "kwh") -> Dict[str, str]:
    """The metric description of a ``--rank-by`` choice."""
    try:
        return RANK_METRICS[rank_by]
    except KeyError:
        raise ValueError(
            f"unknown ranking metric {rank_by!r}; "
            f"expected one of {', '.join(RANK_BY_CHOICES)}"
        ) from None


def ranking_from_metrics(
    metrics: Dict[str, Dict[str, Any]],
    rank_by: str = "kwh",
) -> List[str]:
    """Rank ECM ids by the chosen metric (descending).

    The default metric is the annual facility kWh saved (the historic
    behaviour).  ``rank_by="cost"`` ranks by the annual USD saved and
    ``rank_by="carbon"`` by the annual tCO2 saved, both of which the
    pipeline already computes.  Savings are sorted descending, so ECMs that
    *cost* money (negative savings) sort last; ties are broken by ECM id.
    """
    metric = rank_metric(rank_by)
    key = metric["key"]
    return [
        ecm_id
        for ecm_id, _m in sorted(
            metrics.items(),
            key=lambda kv: (-float(kv[1][key]), kv[0]),
        )
    ]


# ---------------------------------------------------------------------------
# the headline table
# ---------------------------------------------------------------------------
def _detail_ecm_ids(metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    """The ECM ids listed in the per-ECM detail table, in registry order.

    The detail table lists the ECMs that are actually *present* in the
    metrics it is given, so a results set recorded before an ECM was added
    (e.g. the E4HP heat-pump variant) still renders, while a fresh run
    lists every registered ECM.  The order is the registry order, so the
    recorded rendering is unchanged.
    """
    known = [ecm_id for ecm_id in E.ECM_IDS if ecm_id in metrics]
    extra = sorted(ecm_id for ecm_id in metrics if ecm_id not in E.ECM_IDS)
    return known + extra


def headline_table_lines(
    version: str,
    agent_ranking: List[str],
    truth_ranking: List[str],
    prior_ranking: List[str],
    model_metrics: Dict[str, Dict[str, Any]],
    runs: Dict[str, str],
    factors: E.Factors,
    rank_by: str = "kwh",
) -> List[str]:
    """The markdown headline table of the slice-2 ECM run.

    The factor-derived columns (USD saved, tCO2 saved) carry the PLACEHOLDER
    marker only when ``factors`` is the shipped PLACEHOLDER factor set; the
    source path of the factor set is always recorded.

    ``rank_by`` selects the metric the two ranking tables are sorted by
    (``kwh`` - the default and the historic behaviour - ``cost`` or
    ``carbon``); the metric is named in the table headings and ECMs with
    negative savings sort last.
    """
    suffix = " (PLACEHOLDER)" if factors.is_placeholder else ""
    metric = rank_metric(rank_by)
    metric_key = metric["key"]
    metric_format = metric["format"]
    # the ranked metric gets its own column (the kWh ranking table keeps the
    # historic columns only, so the default output stays byte-identical)
    extra_column: Optional[str] = None
    if rank_by != "kwh":
        extra_column = f"{metric['column']}{suffix}"
    lines: List[str] = []
    lines.append("# Retrofit Copilot - slice 2 ECM headline table")
    lines.append("")
    lines.append(f"EnergyPlus: `{version}`")
    lines.append("")
    lines.append(
        f"Factors source: `{factors.source}` "
        f"(electricity ${factors.electricity_price_usd_per_kwh:g}/kWh, "
        f"gas ${factors.natural_gas_price_usd_per_kwh:g}/kWh, "
        f"electricity {factors.electricity_co2_kg_per_kwh:g} kgCO2/kWh, "
        f"gas {factors.natural_gas_co2_kg_per_kwh:g} kgCO2/kWh; "
        f"label: {factors.label})"
    )
    lines.append("")
    if factors.is_placeholder:
        lines.append("All costs and prices are PLACEHOLDER values (see `src/ecms.py`).")
    else:
        lines.append(
            "Energy prices and emission factors come from the factors file above; "
            "the ECM installed costs are still PLACEHOLDER values (see `src/ecms.py`)."
        )
    lines.append("")

    def ranking_section(title: str, ranking: List[str], model_name: str) -> List[str]:
        """The markdown of one ranking table, sorted by the chosen metric."""
        out: List[str] = []
        out.append(f"## {title}, by {metric['heading']}")
        out.append("")
        if extra_column is None:
            out.append("| rank | ECM | kWh saved | % saved | run id |")
            out.append("|---|---|---|---|---|")
        else:
            out.append(f"| rank | ECM | {extra_column} | kWh saved | % saved | run id |")
            out.append("|---|---|---|---|---|---|")
        for i, ecm_id in enumerate(ranking, start=1):
            m = model_metrics[model_name][ecm_id]
            value = metric_format.format(m[metric_key])
            if extra_column is None:
                out.append(
                    f"| {i} | {ecm_id} {E.get_ecm(ecm_id).name} | {m['energy_saved_kwh']:.1f} | "
                    f"{m['energy_saved_pct']:.2f} | `{m['run_id']}` |"
                )
            else:
                out.append(
                    f"| {i} | {ecm_id} {E.get_ecm(ecm_id).name} | {value} | "
                    f"{m['energy_saved_kwh']:.1f} | {m['energy_saved_pct']:.2f} | `{m['run_id']}` |"
                )
        out.append("")
        return out

    lines.extend(
        ranking_section("Agent ranking (calibrated model)", agent_ranking, "calibrated")
    )
    lines.extend(
        ranking_section("Ground-truth ranking (true model)", truth_ranking, "true")
    )
    lines.append("## LLM-prior ranking (written before the simulations)")
    lines.append("")
    lines.append("| rank | ECM |")
    lines.append("|---|---|")
    for i, ecm_id in enumerate(prior_ranking, start=1):
        lines.append(f"| {i} | {ecm_id} {E.get_ecm(ecm_id).name} |")
    lines.append("")
    lines.append("## Full per-ECM detail (agent model / true model)")
    lines.append("")
    lines.append(
        f"| ECM | model | kWh saved | % saved | USD saved{suffix} | tCO2 saved{suffix} "
        "| payback yr (PLACEHOLDER) | mean zone T [C] | d(zone T) [K] | zone-hrs outside 20-26 C | run id |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model_name in ("calibrated", "true"):
        for ecm_id in _detail_ecm_ids(model_metrics[model_name]):
            m = model_metrics[model_name][ecm_id]
            payback = m["simple_payback_years"]
            lines.append(
                f"| {ecm_id} | {model_name} | {m['energy_saved_kwh']:.1f} | {m['energy_saved_pct']:.2f} | "
                f"{m['cost_saved_usd']:.2f} | {m['co2_saved_tco2']:.3f} | "
                f"{'n/a' if payback is None else f'{payback:.1f}'} | "
                f"{m['comfort']['ecm']['zone_mean_air_temp_c']:.2f} | "
                f"{m['comfort']['delta_zone_mean_air_temp_c']:+.2f} | "
                f"{m['comfort']['ecm']['zone_hours_outside_comfort_band']} | `{m['run_id']}` |"
            )
    lines.append("")
    lines.append("## Run audit trail")
    lines.append("")
    lines.append("| role | run id |")
    lines.append("|---|---|")
    for key, rid in runs.items():
        lines.append(f"| {key} | `{rid}` |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------
def run_one(
    idf: Path,
    epw: Path,
    label: str,
    idf_text: str,
    notes: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = R.run_energyplus(
        idf,
        epw,
        label,
        idf_text=idf_text,
        notes=notes,
        params=dict(params),
    )
    if manifest["returncode"] != 0:
        raise RuntimeError(
            f"EnergyPlus run {manifest['run_id']} failed: {manifest.get('failure')}"
        )
    return manifest


def write_outputs(out_path: Path, lines: List[str], results: Dict[str, Any]) -> Path:
    """Write the headline table and ``ecm_results.json``; return the JSON path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    results_path = out_path.with_name("ecm_results.json")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results_path


def build_parser() -> argparse.ArgumentParser:
    """The command line of the slice-2 driver."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run the full slice-2 ECM pipeline")
    ap.add_argument("--out", type=Path, default=RUNS_DIR / "ecm_headline_table.md")
    ap.add_argument(
        "--rank-by",
        choices=list(RANK_BY_CHOICES),
        default="kwh",
        help=(
            "metric the headline table is ranked by: 'kwh' (annual facility "
            "kWh saved, the default), 'cost' (annual USD saved) or 'carbon' "
            "(annual tCO2 saved)"
        ),
    )
    ap.add_argument(
        "--idf",
        default=IDF_NAME,
        help=(
            "building IDF file name, resolved against the EnergyPlus "
            f"install's ExampleFiles directory (default: {IDF_NAME})"
        ),
    )
    ap.add_argument(
        "--epw",
        default=EPW_NAME,
        help=(
            "weather EPW file name, resolved against the EnergyPlus "
            f"install's WeatherData directory (default: {EPW_NAME})"
        ),
    )
    ap.add_argument(
        "--factors",
        type=Path,
        default=None,
        help=(
            "JSON factors file with the four energy-price / emission factors "
            "(electricity_price_usd_per_kwh, natural_gas_price_usd_per_kwh, "
            "electricity_co2_kg_per_kwh, natural_gas_co2_kg_per_kwh); each value "
            "must be a finite, non-negative number.  Default: the shipped "
            "config/factors.placeholder.json (the PLACEHOLDER factor set)."
        ),
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.all:
        ap.error("nothing to do: pass --all")

    # -- the factor set (validated before any simulation) --------------------
    try:
        factors = E.load_factors(args.factors) if args.factors is not None else E.default_factors()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"invalid factors file: {exc}")
    print(f"factors source: {factors.source} (label: {factors.label})")

    # -- the LLM prior control must exist BEFORE any simulation --------------
    if not PRIOR_FILE.is_file():
        raise SystemExit(
            f"the LLM-prior control {PRIOR_FILE} does not exist; it must be "
            "written before any ECM simulation (step 2 of the slice brief)"
        )
    prior_text = PRIOR_FILE.read_text()
    prior_ranking = parse_prior_ranking(prior_text, PRIOR_FILE)
    prior_mtime = datetime.fromtimestamp(PRIOR_FILE.stat().st_mtime, timezone.utc)

    # -- the slice-1 results file (validated before any simulation) ----------
    calibrated_params = load_calibrated_params()

    try:
        binary = R.find_energyplus()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    version = R.energyplus_version(binary)
    install = find_install_dir(binary)
    idf, epw = resolve_input_paths(install, args.idf, args.epw)
    for path in (idf, epw):
        if not path.is_file():
            raise FileNotFoundError(path)

    models = {
        "calibrated": calibrated_params,
        "true": dict(TRUE_PARAMS),
    }

    print(f"EnergyPlus: {version}")
    print(f"prior ranking file: {PRIOR_FILE} (sha256 {sha256_text(prior_text)[:12]})")
    print(f"prior ranking: {' > '.join(prior_ranking)}")

    runs: Dict[str, Any] = {}
    model_metrics: Dict[str, Dict[str, Any]] = {}
    change_records: Dict[str, List[Dict[str, Any]]] = {}

    for model_name, params in models.items():
        model_idf, model_changes = P.apply_params(idf.read_text(), params)
        base_manifest = run_one(
            idf,
            epw,
            f"ecmbase-{model_name}",
            model_idf,
            f"slice-2 baseline for the {model_name} model: "
            + json.dumps(params, sort_keys=True),
            params,
        )
        runs[f"baseline_{model_name}"] = base_manifest["run_id"]
        print(f"baseline [{model_name}] run id: {base_manifest['run_id']}")
        base_series = RES.load_run_series(Path(base_manifest["cwd"]), base_manifest["run_id"])

        metrics: Dict[str, Any] = {}
        for ecm in E.ECMS:
            ecm_idf, changes = ecm.apply(model_idf)
            manifest = run_one(
                idf,
                epw,
                f"ecm{ecm.ecm_id}-{model_name}",
                ecm_idf,
                f"slice-2 ECM {ecm.ecm_id} ({ecm.name}) on the {model_name} model; "
                + json.dumps(params, sort_keys=True),
                {**params, "ecm": ecm.ecm_id},
            )
            runs[f"{ecm.ecm_id}_{model_name}"] = manifest["run_id"]
            change_records[f"{ecm.ecm_id}_{model_name}"] = changes
            ecm_series = RES.load_run_series(Path(manifest["cwd"]), manifest["run_id"])
            cost = ecm.cost(idf.read_text())
            metrics[ecm.ecm_id] = ecm_metrics(
                base_series, ecm_series, cost["amount_usd"], factors=factors
            )
            metrics[ecm.ecm_id]["run_id"] = manifest["run_id"]
            metrics[ecm.ecm_id]["baseline_run_id"] = base_manifest["run_id"]
            metrics[ecm.ecm_id]["placeholder_cost_usd"] = cost["amount_usd"]
            metrics[ecm.ecm_id]["cost_label"] = cost["label"]
            metrics[ecm.ecm_id]["cost_basis"] = cost["basis"]
            metrics[ecm.ecm_id]["n_changes"] = len(changes)
            print(
                f"  {ecm.ecm_id} [{model_name}] run id: {manifest['run_id']} "
                f"saved {metrics[ecm.ecm_id]['energy_saved_kwh']:.1f} kWh "
                f"({metrics[ecm.ecm_id]['energy_saved_pct']:.2f} %)"
            )
        model_metrics[model_name] = metrics

    agent_ranking = ranking_from_metrics(model_metrics["calibrated"], rank_by=args.rank_by)
    truth_ranking = ranking_from_metrics(model_metrics["true"], rank_by=args.rank_by)

    def top3(order: List[str]) -> List[str]:
        return order[:3]

    top3_match = top3(agent_ranking) == top3(truth_ranking)
    prior_top3_match = top3(prior_ranking) == top3(truth_ranking)
    # "the prior beats simulation" = the prior's top-3 matches the ground truth
    # better than the agent's top-3 does.
    def agreement(order: List[str]) -> int:
        return len(set(top3(order)) & set(top3(truth_ranking)))

    prior_beats_agent = agreement(prior_ranking) > agreement(agent_ranking)

    results = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "energyplus_version": version,
        "idf": str(idf),
        "epw": str(epw),
        "models": {
            "calibrated": {"params": calibrated_params, "role": "agent model"},
            "true": {"params": TRUE_PARAMS, "role": "ground truth (synthetic measured)"},
        },
        "llm_prior": {
            "file": str(PRIOR_FILE),
            "sha256": sha256_text(prior_text),
            "file_mtime_utc": prior_mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ranking": prior_ranking,
            "detail": prior["ranking"],
        },
        "runs": runs,
        "change_records": change_records,
        "metrics": model_metrics,
        # the default ranking metric is not recorded, so that the default
        # results file stays byte-identical to the recorded slice-2 one
        **({} if args.rank_by == "kwh" else {"rank_by": args.rank_by}),
        "rankings": {
            "agent_calibrated": agent_ranking,
            "ground_truth_true": truth_ranking,
            "llm_prior": prior_ranking,
        },
        "comparison": {
            "agent_top3": top3(agent_ranking),
            "ground_truth_top3": top3(truth_ranking),
            "prior_top3": top3(prior_ranking),
            "agent_top3_matches_truth_in_order": top3_match,
            "prior_top3_matches_truth_in_order": prior_top3_match,
            "prior_top3_overlap_with_truth": agreement(prior_ranking),
            "agent_top3_overlap_with_truth": agreement(agent_ranking),
            "prior_beats_agent": prior_beats_agent,
        },
        "factors": factors.record(),
    }
    if factors.is_placeholder:
        # the legacy PLACEHOLDER block: only valid for the placeholder factors
        results["placeholders"] = {
            "electricity_price_usd_per_kwh": factors.electricity_price_usd_per_kwh,
            "natural_gas_price_usd_per_kwh": factors.natural_gas_price_usd_per_kwh,
            "electricity_co2_kg_per_kwh": factors.electricity_co2_kg_per_kwh,
            "natural_gas_co2_kg_per_kwh": factors.natural_gas_co2_kg_per_kwh,
            "label": E.PLACEHOLDER,
            "source": factors.source,
        }

    # -- markdown headline table -------------------------------------------
    lines: List[str] = headline_table_lines(
        version=version,
        agent_ranking=agent_ranking,
        truth_ranking=truth_ranking,
        prior_ranking=prior_ranking,
        model_metrics=model_metrics,
        runs=runs,
        factors=factors,
        rank_by=args.rank_by,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results_path = write_outputs(args.out, lines, results)

    print()
    print("\n".join(lines))
    print(f"written: {args.out}")
    print(f"written: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
