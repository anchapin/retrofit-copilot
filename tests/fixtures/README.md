# Recorded-run fixtures (slice 2)

These files are a **trimmed copy** of the slice-2 runs recorded on
2026-11-06 and summarised in `REPORT.md` ("Run audit trail").  They exist so
that the audit-trail guarantees of `tests/test_audit_trail.py` and
`tests/test_manifest.py` are checked in a clean clone -- with no EnergyPlus
install and no `runs/` directory.

## Layout

The directory plays the role of a `runs/` directory:

```
tests/fixtures/
├── ecm_results.json            # trimmed runs/ecm_results.json
├── llm_prior_ranking.json      # trimmed runs/llm_prior_ranking.json
├── ecmbase-calibrated-20261106T110730Z-950f94/manifest.json
├── ecmE1-calibrated-20261106T110817Z-479c57/manifest.json
├── ...                          # one directory per recorded run (12 in all)
└── ecmE5-true-20261106T114528Z-59bfd8/manifest.json
```

Each `manifest.json` is written in exactly the format `run_energyplus.py`
writes (`json.dumps(manifest, indent=2, sort_keys=True) + "\n"`), so a
fixture manifest is byte-identical to a recorded one and reloads to the same
manifest.

## Provenance

* The 12 run ids are the ones in the `REPORT.md` "Run audit trail" table,
  verbatim.
* The headline numbers in `ecm_results.json` (kWh saved, % saved, USD saved,
  tCO2 saved, payback, mean zone temperature, zone-hours outside 20-26 C)
  are the ones in the slice-2 tables of `REPORT.md`.
* `llm_prior.file_mtime_utc` is the recorded write time of the prior
  ranking (2026-11-06T11:04:11Z, before the first ECM run) and
  `llm_prior.sha256` is the sha256 of `llm_prior_ranking.json` next to it.

## Trimmed fields

The EnergyPlus outputs themselves are not committed, so the file hashes and
sizes they carried are recorded as the sentinel `"not-recorded-in-fixture"`
and `null` rather than as invented numbers.  Every other field is the
recorded value.

## Running the same tests against a live `runs/` directory

The fixtures are the default data source.  Set `RETROFIT_RUNS_DIR` to run
the very same tests against a live recorded `runs/` directory (one holding
`ecm_results.json`, `llm_prior_ranking.json` and one directory per run):

```bash
RETROFIT_RUNS_DIR=/path/to/runs python3 -m pytest tests/ -q
```
