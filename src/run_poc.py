"""Slice-1 driver: baseline -> synthetic measured data -> bounded calibration.

Runs the whole POC slice end to end and writes the headline table:

    python src/run_poc.py --all

Everything is recorded: every EnergyPlus execution gets a run id, a run
directory under ``runs/`` and a manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate as C  # noqa: E402
import idf_params as P  # noqa: E402
import results as RES  # noqa: E402
import run_energyplus as R  # noqa: E402

POC_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = POC_ROOT / "runs"
IDF_NAME = "5ZoneAirCooled.idf"
EPW_NAME = "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"

# ---------------------------------------------------------------------------
# the (known) truth and the bounded calibration setup
# ---------------------------------------------------------------------------

#: The synthetic "measured" building: the shipped example with a recorded
#: perturbation.  These are the GROUND TRUTH values of the synthetic data.
PERTURBED_PARAMS: Dict[str, float] = {
    "infiltration_flow_scale": 2.0,       # design flow rate x2
    "envelope_conductivity_scale": 1.30,  # opaque envelope U-value x1.3
    "cooling_setpoint_offset_k": 1.0,     # cooling setpoint schedule +1.0 K
}

#: The bounded calibration grids (fixed, recorded, never widened silently).
GRIDS: Dict[str, List[float]] = {
    "infiltration_flow_scale": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
    "envelope_conductivity_scale": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    "cooling_setpoint_offset_k": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
}

PARAM_ORDER = [
    "infiltration_flow_scale",
    "envelope_conductivity_scale",
    "cooling_setpoint_offset_k",
]

#: Calibration period (the search never looks at the rest of the year).
CALIBRATION_MONTHS = tuple(range(1, 7))
EVALUATION_MONTHS = tuple(range(7, 13))

#: Thresholds applied (ASHRAE Guideline 14, as quoted in the POC brief).
THRESHOLDS = {
    "hourly_cvrmse_pct": 30.0,
    "monthly_nmbe_pct": 5.0,
    "hourly_nmbe_pct": 10.0,
}

SEARCH_PRUNE_GLOBS = ("eplus.eso", "eplus.bnd", "eplus.eio", "eplus.wrl")


# ---------------------------------------------------------------------------
def find_install_dir(binary: Path) -> Path:
    """The EnergyPlus install directory that ships ExampleFiles/ and WeatherData/."""
    cand = binary.parent
    if (cand / "ExampleFiles").is_dir():
        return cand
    raise FileNotFoundError(
        f"no ExampleFiles/ next to the EnergyPlus binary at {binary}"
    )


def months_mask(months, wanted: Sequence[int]) -> List[int]:
    wanted_set = set(wanted)
    return [i for i, m in enumerate(months) if m in wanted_set]


def series_for(run_manifest: Dict[str, Any]) -> RES.RunSeries:
    run_dir = Path(manifest_dir(run_manifest))
    return RES.load_run_series(run_dir, run_manifest["run_id"])


def manifest_dir(manifest: Dict[str, Any]) -> str:
    return str(Path(manifest["cwd"]))


def run_one(
    idf: Path,
    epw: Path,
    label: str,
    params: Dict[str, float],
    notes: str,
    prune: bool = False,
) -> Dict[str, Any]:
    idf_text, changes = P.apply_params(idf.read_text(), params)
    manifest = R.run_energyplus(
        idf,
        epw,
        label,
        idf_text=idf_text,
        notes=notes,
        params=dict(params),
        prune_globs=SEARCH_PRUNE_GLOBS if prune else (),
    )
    manifest["parameter_changes"] = changes
    # re-write the manifest so the parameter changes are part of the record
    with open(R.manifest_path(manifest["run_id"]), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def metric_block(
    measured: RES.RunSeries,
    measured_energy: Any,
    modeled: RES.RunSeries,
    modeled_energy: Any,
    months: Sequence[int],
) -> Dict[str, Any]:
    idx = months_mask(measured.months, months)
    m_energy = measured_energy[idx]
    d_energy = modeled_energy[idx]
    m_temp = RES.zone_mean_temperature(measured)[idx]
    d_temp = RES.zone_mean_temperature(modeled)[idx]
    return {
        "energy": C.metrics(m_energy, d_energy),
        "zone_temperature": C.metrics(m_temp, d_temp),
        "monthly_nmbe_energy": C.monthly_nmbe(m_energy, d_energy, list(measured.months[idx])),
        "monthly_cvrmse_energy": C.monthly_cvrmse(
            m_energy, d_energy, list(measured.months[idx])
        ),
    }


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run the full slice-1 pipeline")
    ap.add_argument("--max-sweeps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=RUNS_DIR / "headline_table.md")
    args = ap.parse_args(argv)

    binary = R.find_energyplus()
    version = R.energyplus_version(binary)
    install = find_install_dir(binary)
    idf = install / "ExampleFiles" / IDF_NAME
    epw = install / "WeatherData" / EPW_NAME
    for path in (idf, epw):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"EnergyPlus: {version}")
    print(f"Building   : {IDF_NAME}")
    print(f"Weather    : {EPW_NAME}")

    # ---------------- baseline (as shipped + reporting-only meter objects)
    baseline_params = dict(P.DEFAULT_PARAMS)
    baseline = run_one(
        idf,
        epw,
        "baseline",
        baseline_params,
        "baseline: shipped 5ZoneAirCooled.idf, reporting-only Output:Meter patch",
    )
    print(f"baseline run id: {baseline['run_id']}")
    if baseline["returncode"] != 0:
        raise RuntimeError(f"baseline EnergyPlus run failed: {baseline.get('failure')}")

    # ---------------- synthetic measured data (perturbed parameters)
    measured_run = run_one(
        idf,
        epw,
        "measured",
        dict(PERTURBED_PARAMS),
        "synthetic measured data: perturbed parameters "
        + json.dumps(PERTURBED_PARAMS, sort_keys=True),
    )
    print(f"measured run id: {measured_run['run_id']}")
    if measured_run["returncode"] != 0:
        raise RuntimeError(
            f"synthetic measured run failed: {measured_run.get('failure')}"
        )

    base_series = RES.load_run_series(Path(baseline["cwd"]), baseline["run_id"])
    meas_series = RES.load_run_series(Path(measured_run["cwd"]), measured_run["run_id"])
    base_energy = RES.facility_energy_kwh(base_series)
    meas_energy = RES.facility_energy_kwh(meas_series)

    # ---------------- bounded calibration (Jan-Jun objective only)
    cal_idx = months_mask(meas_series.months, CALIBRATION_MONTHS)

    def simulate(params: Dict[str, float]) -> Tuple[str, Any]:
        man = run_one(
            idf,
            epw,
            "calib",
            dict(params),
            "calibration search candidate: " + json.dumps(params, sort_keys=True),
            prune=True,
        )
        if man["returncode"] != 0:
            # A candidate that EnergyPlus refuses to simulate (e.g. a
            # thermostat dead-band violation) is an invalid candidate, not a
            # crash: it gets an infinite objective.
            print(f"  candidate rejected (EnergyPlus exit {man['returncode']}): {params}")
            return man["run_id"], None
        series = RES.load_run_series(Path(man["cwd"]), man["run_id"])
        return man["run_id"], RES.facility_energy_kwh(series)

    def objective(energy: Any) -> float:
        if energy is None:
            return float("inf")
        return C.cvrmse(meas_energy[cal_idx], energy[cal_idx])

    search = C.coordinate_search(
        GRIDS,
        start=dict(P.DEFAULT_PARAMS),
        simulate=simulate,
        objective=objective,
        max_sweeps=args.max_sweeps,
        parameter_order=PARAM_ORDER,
    )
    print(
        f"calibration: best objective (CV(RMSE) on Jan-Jun) = {search.best_objective:.4f} %"
    )
    print(f"calibrated parameters: {search.best_params}")

    # final calibrated model, annual run, recorded
    calibrated = run_one(
        idf,
        epw,
        "calibrated",
        dict(search.best_params),
        "final calibrated model: " + json.dumps(search.best_params, sort_keys=True),
    )
    print(f"calibrated run id: {calibrated['run_id']}")

    cal_series = RES.load_run_series(Path(calibrated["cwd"]), calibrated["run_id"])
    cal_energy = RES.facility_energy_kwh(cal_series)

    # ---------------- metrics before / after
    periods = {
        "full year": list(range(1, 13)),
        "calibration (Jan-Jun)": list(CALIBRATION_MONTHS),
        "held-out (Jul-Dec)": list(EVALUATION_MONTHS),
    }
    blocks = {
        name: {
            "before": metric_block(
                meas_series, meas_energy, base_series, base_energy, months
            ),
            "after": metric_block(
                meas_series, meas_energy, cal_series, cal_energy, months
            ),
        }
        for name, months in periods.items()
    }

    # ---------------- headline table
    lines: List[str] = []
    lines.append("# Retrofit Copilot — slice 1 headline table")
    lines.append("")
    lines.append(f"EnergyPlus: `{version}`")
    lines.append("")
    lines.append(
        "| series | period | NMBE % (before) | CV(RMSE) % (before) | NMBE % (after) "
        "| CV(RMSE) % (after) | G14 CV(RMSE) <= 30 % | G14 monthly NMBE <= +/-5 % |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, blk in blocks.items():
        for series_name, key in (("facility energy [kWh]", "energy"), ("zone mean air temp [C]", "zone_temperature")):
            before = blk["before"][key]
            after = blk["after"][key]
            lines.append(
                f"| {series_name} | {name} | {fmt(before['nmbe_pct'])} | "
                f"{fmt(before['cvrmse_pct'])} | {fmt(after['nmbe_pct'])} | "
                f"{fmt(after['cvrmse_pct'])} | "
                f"{'PASS' if after['cvrmse_pct'] <= THRESHOLDS['hourly_cvrmse_pct'] else 'FAIL'} | "
                f"{'see monthly table' if series_name.startswith('facility') else 'n/a'} |"
            )
    lines.append("")
    lines.append("## Monthly NMBE of hourly facility energy [%]")
    lines.append("")
    lines.append("| month | before | after |")
    lines.append("|---|---|---|")
    for month in range(1, 13):
        b = blocks["full year"]["before"]["monthly_nmbe_energy"].get(month)
        a = blocks["full year"]["after"]["monthly_nmbe_energy"].get(month)
        lines.append(f"| {month} | {fmt(b)} | {fmt(a)} |")
    lines.append("")
    lines.append("## Run audit trail")
    lines.append("")
    lines.append("| role | run id | parameters |")
    lines.append("|---|---|---|")
    for role, man in (
        ("baseline", baseline),
        ("synthetic measured", measured_run),
        ("calibrated", calibrated),
    ):
        lines.append(
            f"| {role} | `{man['run_id']}` | `{json.dumps(man['params'], sort_keys=True)}` |"
        )
    lines.append("")
    lines.append(f"Calibration search evaluations: {len(search.evaluated)} runs.")

    table_text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table_text)

    results = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "energyplus_version": version,
        "idf": str(idf),
        "epw": str(epw),
        "thresholds": THRESHOLDS,
        "perturbed_params": PERTURBED_PARAMS,
        "grids": GRIDS,
        "calibration_months": list(CALIBRATION_MONTHS),
        "evaluation_months": list(EVALUATION_MONTHS),
        "runs": {
            "baseline": baseline["run_id"],
            "measured": measured_run["run_id"],
            "calibrated": calibrated["run_id"],
        },
        "calibrated_params": search.best_params,
        "calibration_objective_cvrmse_pct": search.best_objective,
        "calibration_evaluations": len(search.evaluated),
        "metrics": blocks,
        "search_trace": [
            {
                "sweep": s.sweep,
                "parameter": s.parameter,
                "value": s.value,
                "objective": s.objective,
                "run_id": s.run_id,
                "accepted": s.accepted,
            }
            for s in search.steps
        ],
    }
    results_path = args.out.with_name("results.json")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    print()
    print(table_text)
    print(f"written: {args.out}")
    print(f"written: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
