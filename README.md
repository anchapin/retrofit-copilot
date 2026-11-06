# Retrofit Copilot

**Status:** Slice 2 done — **verdict: Go**. The simulation-grounded ranking
(E2 > E1 > E4 > E3 > E5) reproduces the ground-truth ranking exactly, top 3 in
order, while the LLM-prior-only ranking gets it wrong (E3 first, E2 last).
See `REPORT.md` for the full tables, the prior quoted verbatim, and the audit
trail.

Drawing → calibrated baseline → neuro-symbolic agent ranking energy-conservation
measures, simulation-grounded. Standalone POC — no Fluxion or Matchline
integration.

Spun out of [anchapin/building-ai-pocs](https://github.com/anchapin/building-ai-pocs)
(POC #2, PR #14, commit `63bead5f`) on 2026-11-06, after the Go verdict.

POC rules: 25–40 hours, standalone, fixed go/no-go
gate, REPORT.md leading with exactly one label (Go, Go (software only),
Conditional, or No-go).

## Reproduce slice 2 (CPU-only)

```sh
sh scripts/fetch_energyplus.sh   # EnergyPlus 26.2.0, ~242 MB download
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
.venv/bin/python src/run_ecms.py --all   # 10 ECM runs + headline table + ecm_results.json
python3 -m pytest tests/ -q   # 55 passed
```

The ECM pipeline runs each of the 5 ECMs on the calibrated model and on the
true model (12 runs total including the two baselines, ~35 s each under load)
and writes the headline table to `runs/ecm_headline_table.md`
plus `ecm_results.json`. It writes the LLM-prior ranking to
`runs/llm_prior_ranking.json` BEFORE running any ECM
simulation. Runs are scratch (gitignored); the numbers are recorded in
`REPORT.md`.

## Slice-2 headline (EnergyPlus 26.2.0, 5ZoneAirCooled, Chicago TMY3)

Simulation-grounded ranking by annual facility kWh saved — agent (calibrated)
and ground truth (true model) agree exactly, top 3 in order:

| rank | ECM | kWh saved (agent) | kWh saved (truth) |
|---|---|---|---|
| 1 | E2 Glazing upgrade (low-e) | 6357.0 (10.06%) | 6172.0 (9.78%) |
| 2 | E1 Opaque envelope insulation | 4790.3 (7.58%) | 5164.6 (8.18%) |
| 3 | E4 Heating electrification (fuel switch) | 4282.5 (6.77%) | 4267.1 (6.76%) |
| 4 | E3 HVAC plant efficiency upgrade | 3819.2 (6.04%) | 3802.9 (6.03%) |
| 5 | E5 Thermostat deadband widening (setback) | 3751.1 (5.93%) | 3776.3 (5.98%) |

LLM-prior-only ranking (written before the simulations): E3 > E1 > E5 > E4 >
E2 — wrong at both ends. E4 saves site kWh but costs more and emits more
under the PLACEHOLDER factors (see `REPORT.md`); a cost- or carbon-ranked
decision would order it differently. Guideline 14 re-confirmed on fresh
re-runs (full-year CV(RMSE) 1.63%, monthly NMBE within ±0.57%).

## Reproduce slice 1 (CPU-only)

```sh
sh scripts/fetch_energyplus.sh   # EnergyPlus 26.2.0, ~242 MB download
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
.venv/bin/python src/run_poc.py --all --max-sweeps 3 --out runs/headline_table.md
python3 -m pytest tests/ -q   # 30 passed
```

The full pipeline runs 69 EnergyPlus simulations (~10 min, ~9 s each) and
writes the headline table to `runs/headline_table.md` plus
`results.json` with the full search trace. Runs are scratch (gitignored); the
numbers are recorded in `REPORT.md`.

## Slice-1 headline (EnergyPlus 26.2.0, 5ZoneAirCooled, Chicago TMY3)

Synthetic measured data: the shipped example with infiltration ×2.0, envelope
conductivity ×1.3, cooling setpoint +1.0 K. Bounded calibration (3 named
parameters, fixed grids, Jan–Jun objective, Jul–Dec held out) recovered
infiltration ×2.2, envelope ×1.2, setpoint +1.0 K.

| series | period | NMBE % before → after | CV(RMSE) % before → after |
|---|---|---|---|
| facility energy [kWh] | full year | 0.90 → −0.18 | 11.43 → 1.63 |
| facility energy [kWh] | held-out (Jul–Dec) | −0.32 → −0.10 | 10.64 → 1.12 |
| zone mean air temp [C] | full year | 0.07 → −0.04 | 2.02 → 0.15 |

Monthly NMBE of hourly facility energy: before, 7 of 12 months outside ±5%
(−6.01% to +7.17%); after, all 12 months within ±0.57%. Guideline 14
(hourly CV(RMSE) ≤ 30%, monthly NMBE ±5%) met on every reported period.
Full table, run IDs, and caveats: `REPORT.md`.
