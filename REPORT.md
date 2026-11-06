# Retrofit Copilot — slice 1 note

**Slice 1:** EnergyPlus baseline + synthetic measured data + bounded calibration.
**No Go / Conditional / No-go verdict in this slice** — the verdict belongs to
the full POC: it requires the agent system, the 5-ECM ranking, and the
top-3-vs-ground-truth comparison, all of which are later slices. What follows
is what slice 1 built, what it measured, and what it explicitly does not claim.

## What slice 1 built

All work lives in this repo (standalone, CPU-only, no Fluxion or
Matchline integration):

- `scripts/fetch_energyplus.sh` — pins and fetches EnergyPlus 26.2.0
  (NatLabRockies release; the NREL/EnergyPlus repo moved there) into
  `.scratch/` (gitignored) and verifies `--version`.
- `src/run_energyplus.py` — locates the EnergyPlus binary
  (`$ENERGYPLUS_BIN` or `.scratch/EnergyPlus-*/energyplus`), runs one IDF +
  weather file, records a run ID, run directory, and manifest (command,
  input SHAs, output SHAs, return code, duration).
- `src/idf_params.py` — minimal IDF object model plus the three bounded
  calibration parameters, each applied to a copy (the shipped file is never
  edited in place) with a per-value change record:
  `infiltration_flow_scale` (ZoneInfiltration:DesignFlowRate design flow
  rate), `envelope_conductivity_scale` (every `Material` conductivity),
  `cooling_setpoint_offset_k` (every value of the `Clg-SetP-Sch` schedule).
  A reporting-only `Output:Meter` patch (hourly facility meters) is applied
  to every run.
- `src/calibrate.py` — ASHRAE Guideline 14 metrics (NMBE, CV(RMSE), monthly
  splits; `p = 0`, no regression fitted) and the bounded coordinate search
  over fixed, recorded grids (strict-improvement acceptance, reproducible).
- `src/results.py` — parses `eplus.csv` into hourly series (8760 rows) and
  facility-energy / zone-temperature series.
- `src/run_poc.py` — slice-1 driver (`--all`): baseline → synthetic measured
  → bounded calibration → headline table + `results.json`.
- `tests/` — 30 tests (known-value metric cases, IDF patching, manifest
  round-trip, CSV parsing): `python -m pytest tests/ -q`
  → 30 passed.

## What slice 1 measured

Building: the EnergyPlus-shipped example `5ZoneAirCooled.idf`, weather
`USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw`, EnergyPlus 26.2.0. The
example's geometry is the hand-checked geometry input the issue allows (the
drawing verifier is not done); no drawing was parsed in this slice.

Synthetic measured data (ground truth, recorded): the shipped example with
infiltration ×2.0, envelope conductivity ×1.3, cooling setpoint +1.0 K
(run `measured-20261106T090956Z-1483a5`).

Bounded calibration: coordinate search over the three named parameters on
fixed grids (infiltration 1.0–2.4, envelope 0.8–1.4, setpoint −1.0–+2.0 K),
objective = hourly CV(RMSE) of facility energy on Jan–Jun only; Jul–Dec
held out and never seen by the search. 66 search evaluations + baseline +
measured + final calibrated run = 69 EnergyPlus runs. Calibrated parameters
recovered: infiltration ×2.2, envelope ×1.2, setpoint +1.0 K (truth ×2.0,
×1.3, +1.0 K).

Headline table (Guideline 14 thresholds applied: hourly CV(RMSE) ≤ 30%,
monthly NMBE ±5%):

| series | period | NMBE % before → after | CV(RMSE) % before → after | G14 |
|---|---|---|---|---|
| facility energy [kWh] | full year | 0.90 → −0.18 | 11.43 → 1.63 | PASS |
| facility energy [kWh] | calibration (Jan–Jun) | 2.05 → −0.26 | 12.06 → 1.96 | PASS |
| facility energy [kWh] | held-out (Jul–Dec) | −0.32 → −0.10 | 10.64 → 1.12 | PASS |
| zone mean air temp [C] | full year | 0.07 → −0.04 | 2.02 → 0.15 | PASS |

Monthly NMBE of hourly facility energy: before, 7 of 12 months outside ±5%
(−6.01% to +7.17%); after, all 12 months within ±0.57%.

Run audit trail: baseline `baseline-20261106T090947Z-85e761`, synthetic
measured `measured-20261106T090956Z-1483a5`, calibrated
`calibrated-20261106T092111Z-05d7af`. Every number above traces to these
runs; the full search trace is in `runs/results.json` (scratch, gitignored).

Independent supervisor check: re-reading the three `eplus.csv` files with
only the standard library confirmed 8760 hourly rows each and annual
facility energy of 62,528.5 kWh (baseline), 63,098.8 kWh (measured),
63,213.2 kWh (calibrated) — calibrated within 0.18% of measured on the
annual total, consistent with the reported NMBE of −0.18%.

## What slice 1 does NOT claim

- No Go / Conditional / No-go verdict (needs the ECM ranking).
- No ECM ranking, no agent, no LLM-prior control (later slices).
- No drawing was parsed: the geometry is the shipped example file, not a
  Matchline extraction. The drawing → geometry step is still open.
- Guideline 14 is met on synthetic data whose ground truth the calibration
  was built to recover. That proves the calibration machinery works; it
  says nothing about real-building calibration yet.
- The zone-temperature metric is a secondary sanity check, not a Guideline
  14 gate (Guideline 14 applies to energy). Its CV(RMSE) is small partly
  because temperatures sit near a ~22 °C mean.
- `envelope_conductivity_scale` scales EVERY `Material` conductivity,
  including interior materials (carpet, ceiling tiles), not just the opaque
  envelope. The "envelope" label is approximate; the recovered ×1.2 vs
  truth ×1.3 partly reflects that.

## What failed / what would change the recommendation

- The worker hit the 60-step cap with the code, tests, and all 69 runs
  done but REPORT.md, requirements.txt, the README, and .gitignore
  unwritten; the supervisor completed those docs and verified the numbers
  independently (see above). No code change was needed.
- The EnergyPlus repo moved from NREL to NatLabRockies; the fetch script
  pins the new location and version. If that URL goes stale, the reproduce
  command's first step is the thing that breaks.
- The calibration search is coordinate descent on a fixed grid: it
  recovered the truth closely but not exactly (envelope ×1.2 vs ×1.3,
  infiltration ×2.2 vs ×2.0). A later slice that needs tighter recovery
  should widen the grids deliberately and record the change — not silently.

---

# Slice 2 — the 5-ECM ranking, the LLM-prior control, and the verdict (2026-11-06)

**Verdict: Go**

The agent system ranks the 5 ECMs by simulation-grounded annual kWh saved and
reproduces the ground-truth ranking exactly — top 3 in order (E2, E1, E4) and
the full 5 in order — on both the calibrated model and the true model, while
the LLM-prior-only ranking gets the top 3 wrong (only E1 in place, E2 ranked
last). The simulation adds what the prior cannot: the prior's #1 (E3) is
actually #4, and its #5 (E2, "only 61 m2 of glazing") is actually #1 by a
wide margin. Guideline 14 was re-confirmed on fresh re-runs (see below), and
every ranked ECM cites a run ID with a manifest.

## Agent ranking (calibrated model), by annual facility kWh saved

| rank | ECM | kWh saved | % saved | run id |
|---|---|---|---|---|
| 1 | E2 Glazing upgrade (low-e) | 6357.0 | 10.06 | `ecmE2-calibrated-20261106T110859Z-45ced3` |
| 2 | E1 Opaque envelope insulation | 4790.3 | 7.58 | `ecmE1-calibrated-20261106T110817Z-479c57` |
| 3 | E4 Heating electrification (fuel switch) | 4282.5 | 6.77 | `ecmE4-calibrated-20261106T111026Z-6e6d8e` |
| 4 | E3 HVAC plant efficiency upgrade | 3819.2 | 6.04 | `ecmE3-calibrated-20261106T110942Z-047afe` |
| 5 | E5 Thermostat deadband widening (setback) | 3751.1 | 5.93 | `ecmE5-calibrated-20261106T111115Z-fc7474` |

## Ground-truth ranking (true model), by annual facility kWh saved

| rank | ECM | kWh saved | % saved | run id |
|---|---|---|---|---|
| 1 | E2 Glazing upgrade (low-e) | 6172.0 | 9.78 | `ecmE2-true-20261106T111334Z-b37312` |
| 2 | E1 Opaque envelope insulation | 5164.6 | 8.18 | `ecmE1-true-20261106T111248Z-e7d88d` |
| 3 | E4 Heating electrification (fuel switch) | 4267.1 | 6.76 | `ecmE4-true-20261106T111620Z-c3be43` |
| 4 | E3 HVAC plant efficiency upgrade | 3802.9 | 6.03 | `ecmE3-true-20261106T111428Z-577217` |
| 5 | E5 Thermostat deadband widening (setback) | 3776.3 | 5.98 | `ecmE5-true-20261106T114528Z-59bfd8` |

**Top-3 comparison, stated explicitly:** agent top 3 = E2, E1, E4; ground-truth
top 3 = E2, E1, E4 — they match in order, and the full 5-ECM order matches
too. The agent ranking and the ground-truth ranking agree to within 3.0% on
every ECM's kWh saved (largest gap: E1, 4790.3 vs 5164.6 kWh, 7.3% — still
the same rank).

## LLM-prior ranking (written before any ECM simulation — quoted verbatim from `runs/llm_prior_ranking.json`, 2026-11-06T11:04:11Z)

| rank | ECM | reason (worker's own words) |
|---|---|---|
| 1 | E3 HVAC plant efficiency upgrade | "Chiller COP 3.2 -> 4.5 cuts the largest end use (chiller electricity) by roughly a quarter, and the boiler gain is on top; plant-efficiency ECMs are normally the biggest kWh winners on an all-electric-cooling VAV building." |
| 2 | E1 Opaque envelope insulation | "50 mm of insulation on 738 m2 of roof and exterior walls reduces both the heating and the cooling load of every zone, so it should be the second-largest kWh effect even though the envelope is leaky to begin with." |
| 3 | E5 Thermostat deadband widening (setback) | "Widening the thermostat deadband by 2 K typically returns a high-single-digit percentage of HVAC energy on an office with wide operating hours; smaller than a plant or envelope change but not negligible." |
| 4 | E4 Heating electrification (fuel switch) | "Gas boiler at 80% -> electric resistance at 100% removes ~20% of the heating fuel input, but heating is only a minority of this building's annual load, so the kWh effect should be modest." |
| 5 | E2 Glazing upgrade (low-e) | "Only 61 m2 of glazing on a 738 m2 envelope, so even a good low-e window changes little of the total load; the smallest kWh effect of the five." |

**Prior-vs-truth comparison, stated explicitly:** prior top 3 = E3, E1, E5;
ground-truth top 3 = E2, E1, E4 — they do NOT match in order (overlap: 1 of
3, E1 only, and in the right position by luck). The prior does NOT beat the
simulation-grounded ranking: it puts the true #1 (E2) last and the true #4
(E3) first. The simulation-grounded ranking is strictly better, so the
agent system earns its keep and the No-go branch (prior beats simulation)
does not trigger.

## Full per-ECM detail (agent model / true model)

All costs, prices, and emission factors are PLACEHOLDER values (see
`src/ecms.py`: electricity $0.13/kWh, gas $0.06/kWh, electricity
0.37 kgCO2/kWh, gas 0.18 kgCO2/kWh). Payback = placeholder cost / annual $
saved; n/a where the annual $ saved is not positive.

| ECM | model | kWh saved | % saved | USD saved (PLACEHOLDER) | tCO2 saved (PLACEHOLDER) | payback yr (PLACEHOLDER) | mean zone T [C] | d(zone T) [K] | zone-hrs outside 20–26 C | run id |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | calibrated | 4790.3 | 7.58 | 346.54 | 1.023 | 127.7 | 21.54 | +0.11 | 12169 | `ecmE1-calibrated-20261106T110817Z-479c57` |
| E2 | calibrated | 6357.0 | 10.06 | 467.16 | 1.377 | 58.7 | 21.48 | +0.05 | 12184 | `ecmE2-calibrated-20261106T110859Z-45ced3` |
| E3 | calibrated | 3819.2 | 6.04 | 300.99 | 0.882 | 149.5 | 21.43 | +0.00 | 13475 | `ecmE3-calibrated-20261106T110942Z-047afe` |
| E4 | calibrated | 4282.5 | 6.77 | -942.15 | -2.484 | n/a | 21.43 | +0.00 | 13475 | `ecmE4-calibrated-20261106T111026Z-6e6d8e` |
| E5 | calibrated | 3751.1 | 5.93 | 267.18 | 0.790 | 9.4 | 21.38 | -0.05 | 13631 | `ecmE5-calibrated-20261106T111115Z-fc7474` |
| E1 | true | 5164.6 | 8.18 | 373.08 | 1.101 | 118.7 | 21.54 | +0.12 | 12163 | `ecmE1-true-20261106T111248Z-e7d88d` |
| E2 | true | 6172.0 | 9.78 | 453.15 | 1.336 | 60.5 | 21.47 | +0.05 | 12332 | `ecmE2-true-20261106T111334Z-b37312` |
| E3 | true | 3802.9 | 6.03 | 299.57 | 0.878 | 150.2 | 21.42 | +0.00 | 13582 | `ecmE3-true-20261106T111428Z-577217` |
| E4 | true | 4267.1 | 6.76 | -938.77 | -2.475 | n/a | 21.42 | +0.00 | 13582 | `ecmE4-true-20261106T111620Z-c3be43` |
| E5 | true | 3776.3 | 5.98 | 268.66 | 0.794 | 9.3 | 21.37 | -0.05 | 13725 | `ecmE5-true-20261106T114528Z-59bfd8` |

### Reading the E4 row honestly

E4 is the one ECM where the ranking metric matters. It saves site kWh
(4282.5 calibrated) because electric resistance at 100% beats a gas boiler
at 80%, but under the placeholder factors it costs $942/yr MORE and emits
2.48 tCO2/yr MORE — electricity is $0.13 vs gas $0.06 per kWh and
0.37 vs 0.18 kgCO2 per kWh, and the fuel switch moves heating energy from
the cheap/low-carbon fuel to the expensive/high-carbon one. The kWh ranking
(by the issue's stated metric) puts E4 third; a cost- or carbon-ranked
decision would put it last, and its payback is n/a. That is a real finding
about the placeholder factors, not a code bug: the metrics split fuels
(`src/run_ecms.py::fuel_kwh` + `ecm_metrics`) instead of applying one
$/kWh to net kWh. A real deployment must price E4 with real tariffs and a
real grid factor before acting on it — and a heat-pump variant (COP > 1)
would change this row entirely; the resistance-boiler substitution is
recorded in `src/ecms.py` (a full air-source heat-pump swap is not an IDF
edit of this model, it would be a different building model).

### Comfort check (reported separately from energy savings, per the issue)

No ECM degrades comfort materially: mean zone temperature moves by at most
+0.12 K (E1) and zone-hours outside 20–26 °C stay in the 12163–13725 band
across all 10 runs. E3/E4 leave zone temperatures essentially unchanged
(d = +0.00 K) because they change plant efficiency/fuel, not setpoints —
the same unmet-hours count as their baseline, which is expected, not a
measurement fault. E5 (deadband widening) moves mean zone temperature
−0.05 K with the highest outside-band hours (13631 calibrated), consistent
with a wider deadband.

## Calibration re-confirmation (Guideline 14, fresh re-runs)

Slice 1's headline numbers were re-confirmed on the fresh re-runs from this
slice, not assumed. The supervisor re-ran the repo's own Guideline 14
computation (`run_poc.metric_block`) on the fresh synthetic-measured run
(`measured-20261106T110520Z-9b0901`, truth params) vs the fresh calibrated
run (`ecmbase-calibrated-20261106T110730Z-950f94`, recovered params
infiltration ×2.2, envelope ×1.2, setpoint +1.0 K) and the fresh baseline
(`baseline-20261106T110446Z-cafb7b`):

| series | period | NMBE % before → after | CV(RMSE) % before → after | G14 |
|---|---|---|---|---|
| facility energy [kWh] | full year | 0.90 → −0.18 | 11.43 → 1.63 | PASS |
| facility energy [kWh] | calibration (Jan–Jun) | 2.05 → −0.26 | 12.06 → 1.96 | PASS |
| facility energy [kWh] | held-out (Jul–Dec) | −0.32 → −0.10 | 10.64 → 1.12 | PASS |
| zone mean air temp [C] | full year | 0.07 → −0.04 | 2.02 → 0.15 | PASS |

Monthly NMBE of hourly facility energy after calibration: all 12 months
within ±0.57% (slice 1 recorded the same). The re-run reproduces slice 1's
numbers exactly to two decimals — EnergyPlus is deterministic here, and the
calibrated model behind the agent ranking is the same model slice 1
validated.

## The 5 ECMs (definitions in `src/ecms.py`, deterministic IDF edits on a copy, each with a per-value change record)

- **E1 envelope:** a new 50 mm insulation `Material` is appended and inserted
  as the outside layer of every construction used by an outdoor-facing wall
  or roof surface only. Interior constructions are never touched — this
  deliberately does NOT repeat slice 1's blanket conductivity scale.
- **E2 glazing:** low-e glazing — window U-value / SHGC improvement on the
  5ZoneAirCooled glazing (before/after values in `src/ecms.py`).
- **E3 HVAC efficiency:** cooling COP 3.2 → 4.5 (and the boiler gain stated
  in the module); constant-efficiency equipment, same part-load curves.
- **E4 electrification:** `Boiler:HotWater` "Central Boiler" switched from
  natural gas at 80% to electricity at 100% (electric resistance boiler).
  SUBSTITUTION recorded in the module: a full air-source heat-pump swap is
  not implementable as an IDF edit of this model.
- **E5 controls:** thermostat deadband widened by 2 K (exact schedule change
  in `src/ecms.py`).

Every cost constant carries the PLACEHOLDER marker (`src/ecms.py::
check_placeholder_labels`, covered by tests). No cost figure in code,
tests, or this report is a real cost.

## Tests

`python3 -m pytest tests/ -q` → **55 passed** (30 slice-1 +
25 new: 19 in `tests/test_ecms.py` covering each ECM's IDF edit as a
known-value check, the ranking/cost/payback math, and placeholder labelling;
6 in `tests/test_audit_trail.py` covering the run-ID audit trail — every
ranked ECM cites a run ID present in a manifest).

## Run audit trail

| role | run id |
|---|---|
| baseline_calibrated | `ecmbase-calibrated-20261106T110730Z-950f94` |
| E1_calibrated | `ecmE1-calibrated-20261106T110817Z-479c57` |
| E2_calibrated | `ecmE2-calibrated-20261106T110859Z-45ced3` |
| E3_calibrated | `ecmE3-calibrated-20261106T110942Z-047afe` |
| E4_calibrated | `ecmE4-calibrated-20261106T111026Z-6e6d8e` |
| E5_calibrated | `ecmE5-calibrated-20261106T111115Z-fc7474` |
| baseline_true | `ecmbase-true-20261106T111204Z-398cbb` |
| E1_true | `ecmE1-true-20261106T111248Z-e7d88d` |
| E2_true | `ecmE2-true-20261106T111334Z-b37312` |
| E3_true | `ecmE3-true-20261106T111428Z-577217` |
| E4_true | `ecmE4-true-20261106T111620Z-c3be43` |
| E5_true | `ecmE5-true-20261106T114528Z-59bfd8` |

Plus the slice-1 re-run runs that back the calibration re-confirmation:
baseline `baseline-20261106T110446Z-cafb7b`, synthetic measured
`measured-20261106T110520Z-9b0901`. Full machine-readable results in
`runs/ecm_results.json` (scratch, gitignored); headline table in
`runs/ecm_headline_table.md`. EnergyPlus 26.2.0, 5ZoneAirCooled.idf,
Chicago TMY3, CPU-only.

## What failed / what would change the recommendation

- The worker hit the 60-step cap while waiting on the EnergyPlus runs: its
  `WORKER_REPORT.md` is a think-leak with no verdict, and REPORT.md / README
  were never written by the worker. The ECM pipeline itself kept running as
  an orphaned background process after the cap and finished cleanly at
  11:46 UTC — all 10 ECM runs, `ecm_results.json`, and the headline table.
  The supervisor verified the diff and the run outputs independently and
  authored this addendum + the README update; no code change was needed.
- The slice-1 re-run (`run_poc.py --all`, 69 runs) was killed by a VM
  restart ~12:23 UTC after 46 calibration-sweep evaluations, with no
  summary written — so the sweep did not re-derive the calibrated
  parameters this cycle. The Guideline 14 re-confirmation above does not
  depend on the sweep: it compares the fresh calibrated run (slice 1's
  recorded recovered parameters) against the fresh measured run, and
  reproduces slice 1 exactly.
- The runs were slow under load (~35 s each vs slice 1's ~9 s) because other
  workers were running concurrently; the numbers are unaffected.
- What would change the recommendation: a heat-pump (COP > 1) variant of E4
  would rewrite the E4 row and could change the ranking; real tariffs / a
  real grid factor would change the $ and tCO2 columns (the kWh ranking is
  the robust part); a different building (more glazing, different climate)
  would change which ECM wins — E2's win here rests on this building's
  glazing load. None of those overturn the Go: on this building, with this
  calibration, simulation beats the prior.
