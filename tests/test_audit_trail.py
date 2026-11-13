"""Audit-trail tests for the slice-2 ECM runs.

These tests read the recorded outputs of the slice-2 pipeline: the run
manifests, the results file and the LLM-prior ranking.

Data source
-----------
By default the tests read the committed fixtures under ``tests/fixtures/``
-- a trimmed copy of the runs recorded in ``REPORT.md`` ("Run audit
trail") -- so the audit-trail guarantees are checked in a clean clone, with
no EnergyPlus and no ``runs/`` directory.  To run exactly the same tests
against a live ``runs/`` directory (one holding ``ecm_results.json``,
``llm_prior_ranking.json`` and one directory per recorded run), point the
environment variable at it::

    RETROFIT_RUNS_DIR=/path/to/runs python3 -m pytest tests/ -q
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import ecms as E
import run_energyplus as R

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
#: The recorded runs: the committed fixtures by default, or the live
#: ``runs/`` directory given in ``$RETROFIT_RUNS_DIR``.
RUNS_DIR = (
    Path(os.environ["RETROFIT_RUNS_DIR"]).expanduser()
    if os.environ.get("RETROFIT_RUNS_DIR")
    else FIXTURES_DIR
)
RESULTS_PATH = RUNS_DIR / "ecm_results.json"
PRIOR_PATH = RUNS_DIR / "llm_prior_ranking.json"


def _load_results():
    return json.loads(RESULTS_PATH.read_text())


def _prior_write_time(results):
    """When the LLM-prior ranking was written.

    The recorded results carry the prior file's own write stamp
    (``llm_prior.file_mtime_utc``).  That recorded stamp is used because a
    committed fixture's file mtime is the checkout time, not the write
    time; a recorded ``runs/`` directory that predates the field falls back
    to the file's mtime.
    """
    recorded = results.get("llm_prior", {}).get("file_mtime_utc")
    if recorded:
        return datetime.strptime(recorded, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    return datetime.fromtimestamp(PRIOR_PATH.stat().st_mtime, timezone.utc)


def test_every_ranked_ecm_cites_a_run_id_that_exists_in_a_manifest():
    results = _load_results()
    metrics = results["metrics"]
    for model in ("calibrated", "true"):
        for ecm_id in E.ECM_IDS:
            run_id = metrics[model][ecm_id]["run_id"]
            assert run_id
            path = R.manifest_path(run_id, runs_dir=RUNS_DIR)
            assert path.is_file(), f"no manifest for {ecm_id}/{model}: {run_id}"
            manifest = json.loads(path.read_text())
            assert manifest["run_id"] == run_id
            assert manifest["returncode"] == 0
            assert "eplus.csv" in manifest["outputs"]
            assert manifest["inputs"]["idf_simulated"]["patched"] is True


def test_the_ten_ecm_runs_and_the_two_baselines_are_recorded():
    results = _load_results()
    runs = results["runs"]
    assert set(runs) == {
        "baseline_calibrated",
        "baseline_true",
    } | {f"{ecm_id}_{model}" for ecm_id in E.ECM_IDS for model in ("calibrated", "true")}
    assert len(runs) == 12
    for run_id in runs.values():
        assert R.manifest_path(run_id, runs_dir=RUNS_DIR).is_file()


def test_the_run_ids_in_the_rankings_are_the_run_ids_of_the_metrics():
    results = _load_results()
    for ranking_key, model in (
        ("agent_calibrated", "calibrated"),
        ("ground_truth_true", "true"),
    ):
        for ecm_id in results["rankings"][ranking_key]:
            assert results["metrics"][model][ecm_id]["run_id"] == results["runs"][f"{ecm_id}_{model}"]


def test_the_llm_prior_control_predates_the_ecm_simulations():
    results = _load_results()
    assert PRIOR_PATH.is_file()
    assert results["llm_prior"]["sha256"]
    prior_mtime = _prior_write_time(results)
    created = []
    for ecm_id in E.ECM_IDS:
        for model in ("calibrated", "true"):
            run_id = results["runs"][f"{ecm_id}_{model}"]
            manifest = json.loads(
                R.manifest_path(run_id, runs_dir=RUNS_DIR).read_text()
            )
            created.append(datetime.strptime(manifest["created_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
    assert prior_mtime < min(created), "the prior ranking must be written before the ECM runs"


def test_the_prior_ranking_covers_the_five_ecms():
    results = _load_results()
    prior = results["llm_prior"]["ranking"]
    assert sorted(prior) == sorted(E.ECM_IDS)


def test_every_reported_cost_in_the_results_is_a_placeholder():
    results = _load_results()
    for model in ("calibrated", "true"):
        for ecm_id in E.ECM_IDS:
            entry = results["metrics"][model][ecm_id]
            assert entry["cost_label"] == E.PLACEHOLDER
            assert E.PLACEHOLDER in entry["cost_basis"]
