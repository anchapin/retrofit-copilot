"""Slice-2 driver: run the 5 ECMs on the calibrated model AND the true model,
rank them, and compare the simulation-grounded ranking with the LLM prior.

    python src/run_ecms.py --all

Twelve EnergyPlus runs are recorded (2 model baselines + 5 ECMs x 2 models);
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


def load_calibrated_params() -> Dict[str, float]:
    """Use the slice-1 results file when present, else the recorded values."""
    results_path = RUNS_DIR / "results.json"
    if results_path.is_file():
        data = json.loads(results_path.read_text())
        params = data.get("calibrated_params")
        if params:
            return {k: float(v) for k, v in params.items()}
    return dict(CALIBRATED_PARAMS)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def fuel_kwh(series: RES.RunSeries) -> Dict[str, float]:
    """Annual energy per fuel [kWh] from the recorded meters."""
    out = {"electricity": 0.0, "natural_gas": 0.0}
    if series.has("Electricity:Facility [J](Hourly)"):
        out["electricity"] = float(series.column("Electricity:Facility [J](Hourly)").sum()) / RES.JOULES_PER_KWH
    if series.has("NaturalGas:Facility [J](Hourly)"):
        out["natural_gas"] = float(series.column("NaturalGas:Facility [J](Hourly)").sum()) / RES.JOULES_PER_KWH
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
) -> Dict[str, Any]:
    """Savings, cost, CO2, payback and comfort of one ECM against a baseline."""
    base_fuel = fuel_kwh(baseline)
    ecm_fuel = fuel_kwh(ecm_series)
    d_elec = base_fuel["electricity"] - ecm_fuel["electricity"]
    d_gas = base_fuel["natural_gas"] - ecm_fuel["natural_gas"]
    saved_kwh = base_fuel["total"] - ecm_fuel["total"]
    saved_pct = 100.0 * saved_kwh / base_fuel["total"] if base_fuel["total"] else 0.0
    cost_saved = (
        d_elec * E.ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER
        + d_gas * E.NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER
    )
    co2_kg = (
        d_elec * E.ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER
        + d_gas * E.NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER
    )
    payback = placeholder_cost_usd / cost_saved if cost_saved > 0 else None
    base_comfort = comfort_metrics(baseline)
    ecm_comfort = comfort_metrics(ecm_series)
    return {
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
        "cost_saved_usd_placeholder": cost_saved,
        "co2_saved_tco2_placeholder": co2_kg / 1000.0,
        "simple_payback_years_placeholder": payback,
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


def ranking_from_metrics(metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    """Rank ECM ids by annual facility kWh saved (descending)."""
    return [
        ecm_id
        for ecm_id, _m in sorted(
            metrics.items(),
            key=lambda kv: (-kv[1]["energy_saved_kwh"], kv[0]),
        )
    ]


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run the full slice-2 ECM pipeline")
    ap.add_argument("--out", type=Path, default=RUNS_DIR / "ecm_headline_table.md")
    args = ap.parse_args(argv)
    if not args.all:
        ap.error("nothing to do: pass --all")

    # -- the LLM prior control must exist BEFORE any simulation --------------
    if not PRIOR_FILE.is_file():
        raise SystemExit(
            f"the LLM-prior control {PRIOR_FILE} does not exist; it must be "
            "written before any ECM simulation (step 2 of the slice brief)"
        )
    prior_text = PRIOR_FILE.read_text()
    prior = json.loads(prior_text)
    prior_ranking = [entry["ecm_id"] for entry in prior["ranking"]]
    prior_mtime = datetime.fromtimestamp(PRIOR_FILE.stat().st_mtime, timezone.utc)

    binary = R.find_energyplus()
    version = R.energyplus_version(binary)
    install = find_install_dir(binary)
    idf = install / "ExampleFiles" / IDF_NAME
    epw = install / "WeatherData" / EPW_NAME
    for path in (idf, epw):
        if not path.is_file():
            raise FileNotFoundError(path)

    calibrated_params = load_calibrated_params()
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
            metrics[ecm.ecm_id] = ecm_metrics(base_series, ecm_series, cost["amount_usd"])
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

    agent_ranking = ranking_from_metrics(model_metrics["calibrated"])
    truth_ranking = ranking_from_metrics(model_metrics["true"])

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
        "placeholders": {
            "electricity_price_usd_per_kwh": E.ELECTRICITY_PRICE_USD_PER_KWH_PLACEHOLDER,
            "natural_gas_price_usd_per_kwh": E.NATURAL_GAS_PRICE_USD_PER_KWH_PLACEHOLDER,
            "electricity_co2_kg_per_kwh": E.ELECTRICITY_CO2_KG_PER_KWH_PLACEHOLDER,
            "natural_gas_co2_kg_per_kwh": E.NATURAL_GAS_CO2_KG_PER_KWH_PLACEHOLDER,
            "label": E.PLACEHOLDER,
        },
    }

    # -- markdown headline table -------------------------------------------
    lines: List[str] = []
    lines.append("# Retrofit Copilot - slice 2 ECM headline table")
    lines.append("")
    lines.append(f"EnergyPlus: `{version}`")
    lines.append("")
    lines.append("All costs and prices are PLACEHOLDER values (see `src/ecms.py`).")
    lines.append("")
    lines.append("## Agent ranking (calibrated model), by annual facility kWh saved")
    lines.append("")
    lines.append("| rank | ECM | kWh saved | % saved | run id |")
    lines.append("|---|---|---|---|---|")
    for i, ecm_id in enumerate(agent_ranking, start=1):
        m = model_metrics["calibrated"][ecm_id]
        lines.append(
            f"| {i} | {ecm_id} {E.get_ecm(ecm_id).name} | {m['energy_saved_kwh']:.1f} | "
            f"{m['energy_saved_pct']:.2f} | `{m['run_id']}` |"
        )
    lines.append("")
    lines.append("## Ground-truth ranking (true model), by annual facility kWh saved")
    lines.append("")
    lines.append("| rank | ECM | kWh saved | % saved | run id |")
    lines.append("|---|---|---|---|---|")
    for i, ecm_id in enumerate(truth_ranking, start=1):
        m = model_metrics["true"][ecm_id]
        lines.append(
            f"| {i} | {ecm_id} {E.get_ecm(ecm_id).name} | {m['energy_saved_kwh']:.1f} | "
            f"{m['energy_saved_pct']:.2f} | `{m['run_id']}` |"
        )
    lines.append("")
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
        "| ECM | model | kWh saved | % saved | USD saved (PLACEHOLDER) | tCO2 saved (PLACEHOLDER) "
        "| payback yr (PLACEHOLDER) | mean zone T [C] | d(zone T) [K] | zone-hrs outside 20-26 C | run id |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model_name in ("calibrated", "true"):
        for ecm_id in E.ECM_IDS:
            m = model_metrics[model_name][ecm_id]
            payback = m["simple_payback_years_placeholder"]
            lines.append(
                f"| {ecm_id} | {model_name} | {m['energy_saved_kwh']:.1f} | {m['energy_saved_pct']:.2f} | "
                f"{m['cost_saved_usd_placeholder']:.2f} | {m['co2_saved_tco2_placeholder']:.3f} | "
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    results_path = args.out.with_name("ecm_results.json")
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    print()
    print("\n".join(lines))
    print(f"written: {args.out}")
    print(f"written: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
