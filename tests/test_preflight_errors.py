"""Pre-flight errors of the two runners (issue #12).

The runners read three inputs before they simulate anything: the LLM-prior
control file (``runs/llm_prior_ranking.json``), the slice-1 results file
(``runs/results.json``) and the EnergyPlus binary.  A broken input must
produce a clean one-line error that names the file and the problem - not a
traceback - and the EnergyPlus hint must name the script that actually
exists in this repository (``scripts/fetch_energyplus.sh``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import ecms as E
import idf_params as P
import run_ecms as RE
import run_energyplus as R
import run_poc as POC

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

#: The name of the script that does NOT exist in this repository (the hint in
#: the "EnergyPlus not found" message used to point at it).  It is spelled
#: out in two pieces so that no source file contains the whole string.
MISSING_SCRIPT = "install_" + "energyplus.sh"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _valid_prior() -> dict:
    return {"ranking": [{"ecm_id": ecm_id} for ecm_id in E.ECM_IDS]}


def _write_prior(tmp_path: Path, payload) -> Path:
    path = tmp_path / "llm_prior_ranking.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def _no_energyplus(monkeypatch, tmp_path) -> None:
    """Make sure no EnergyPlus binary can be found."""
    monkeypatch.delenv("ENERGYPLUS_BIN", raising=False)
    monkeypatch.setattr(R, "SCRATCH_DIR", tmp_path / "no-such-scratch")
    monkeypatch.setattr(shutil, "which", lambda name: None)


# ---------------------------------------------------------------------------
# the LLM-prior control file
# ---------------------------------------------------------------------------
def test_a_malformed_prior_file_is_a_clean_error(monkeypatch, tmp_path):
    path = _write_prior(tmp_path, "{ this is not json")
    monkeypatch.setattr(RE, "PRIOR_FILE", path)
    with pytest.raises(SystemExit) as excinfo:
        RE.main(["--all"])
    message = str(excinfo.value)
    assert "llm_prior_ranking.json" in message
    assert "JSON" in message
    assert "\n" not in message


def test_a_prior_file_without_a_ranking_key_is_a_clean_error(monkeypatch, tmp_path):
    path = _write_prior(tmp_path, {"not_ranking": []})
    monkeypatch.setattr(RE, "PRIOR_FILE", path)
    with pytest.raises(SystemExit) as excinfo:
        RE.main(["--all"])
    message = str(excinfo.value)
    assert "llm_prior_ranking.json" in message
    assert "ranking" in message
    assert "\n" not in message


def test_a_prior_entry_without_an_ecm_id_is_a_clean_error(monkeypatch, tmp_path):
    path = _write_prior(tmp_path, {"ranking": [{"name": "no id here"}]})
    monkeypatch.setattr(RE, "PRIOR_FILE", path)
    with pytest.raises(SystemExit) as excinfo:
        RE.main(["--all"])
    message = str(excinfo.value)
    assert "llm_prior_ranking.json" in message
    assert "ecm_id" in message
    assert "\n" not in message


def test_a_prior_file_with_an_unknown_ecm_id_is_a_clean_error(monkeypatch, tmp_path):
    path = _write_prior(tmp_path, {"ranking": [{"ecm_id": "E99"}]})
    monkeypatch.setattr(RE, "PRIOR_FILE", path)
    with pytest.raises(SystemExit) as excinfo:
        RE.main(["--all"])
    message = str(excinfo.value)
    assert "llm_prior_ranking.json" in message
    assert "E99" in message
    assert "\n" not in message


def test_a_valid_prior_file_parses_to_the_ranking(monkeypatch, tmp_path):
    path = _write_prior(tmp_path, _valid_prior())
    monkeypatch.setattr(RE, "PRIOR_FILE", path)
    assert RE.parse_prior_ranking(path.read_text(), path) == list(E.ECM_IDS)


def test_the_prior_validation_helper_names_the_file_and_the_problem(tmp_path):
    path = tmp_path / "llm_prior_ranking.json"
    with pytest.raises(SystemExit) as excinfo:
        RE.parse_prior_ranking("nope", path)
    message = str(excinfo.value)
    assert "llm_prior_ranking.json" in message
    assert "JSON" in message


# ---------------------------------------------------------------------------
# the slice-1 results file (runs/results.json)
# ---------------------------------------------------------------------------
def test_a_malformed_results_file_is_a_clean_error(monkeypatch, tmp_path):
    (tmp_path / "results.json").write_text("{ nope")
    monkeypatch.setattr(RE, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        RE.load_calibrated_params()
    message = str(excinfo.value)
    assert "results.json" in message
    assert "JSON" in message
    assert "\n" not in message


def test_a_results_file_with_an_unknown_parameter_is_a_clean_error(monkeypatch, tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"calibrated_params": {"not_a_parameter": 1.0}})
    )
    monkeypatch.setattr(RE, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        RE.load_calibrated_params()
    message = str(excinfo.value)
    assert "results.json" in message
    assert "not_a_parameter" in message
    assert "\n" not in message


def test_a_results_file_with_a_non_numeric_value_is_a_clean_error(monkeypatch, tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"calibrated_params": {P.PARAM_NAMES[0]: "not a number"}})
    )
    monkeypatch.setattr(RE, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        RE.load_calibrated_params()
    message = str(excinfo.value)
    assert "results.json" in message
    assert P.PARAM_NAMES[0] in message
    assert "\n" not in message


def test_an_absent_results_file_falls_back_to_the_recorded_params(monkeypatch, tmp_path):
    monkeypatch.setattr(RE, "RUNS_DIR", tmp_path)
    assert not (tmp_path / "results.json").exists()
    assert RE.load_calibrated_params() == RE.CALIBRATED_PARAMS


def test_a_valid_results_file_is_used(monkeypatch, tmp_path):
    params = {name: 1.5 for name in P.PARAM_NAMES}
    (tmp_path / "results.json").write_text(json.dumps({"calibrated_params": params}))
    monkeypatch.setattr(RE, "RUNS_DIR", tmp_path)
    assert RE.load_calibrated_params() == params


def test_the_runner_reports_a_broken_results_file_before_it_looks_for_energyplus(
    monkeypatch, tmp_path
):
    """File inputs are checked first: a broken results file is reported."""
    _no_energyplus(monkeypatch, tmp_path)
    monkeypatch.setattr(RE, "PRIOR_FILE", _write_prior(tmp_path, _valid_prior()))
    monkeypatch.setattr(RE, "RUNS_DIR", tmp_path)
    (tmp_path / "results.json").write_text(
        json.dumps({"calibrated_params": {"not_a_parameter": 1.0}})
    )
    with pytest.raises(SystemExit) as excinfo:
        RE.main(["--all"])
    message = str(excinfo.value)
    assert "results.json" in message
    assert "not_a_parameter" in message
    assert "fetch_energyplus" not in message


# ---------------------------------------------------------------------------
# the EnergyPlus binary
# ---------------------------------------------------------------------------
def test_the_energyplus_hint_names_the_script_that_exists(monkeypatch, tmp_path):
    """The "not found" message points at the script this repo ships."""
    _no_energyplus(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        R.find_energyplus()
    message = str(excinfo.value)
    assert "scripts/fetch_energyplus.sh" in message
    assert MISSING_SCRIPT not in message
    assert (REPO_ROOT / "scripts" / "fetch_energyplus.sh").is_file()


def test_the_source_tree_does_not_mention_the_missing_install_script():
    hits = [
        path.name
        for path in SRC_DIR.glob("*.py")
        if MISSING_SCRIPT in path.read_text()
    ]
    assert hits == []


def test_run_ecms_reports_the_missing_energyplus_cleanly(monkeypatch, tmp_path):
    _no_energyplus(monkeypatch, tmp_path)
    monkeypatch.setattr(RE, "PRIOR_FILE", _write_prior(tmp_path, _valid_prior()))
    monkeypatch.setattr(RE, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        RE.main(["--all"])
    message = str(excinfo.value)
    assert "scripts/fetch_energyplus.sh" in message
    assert "\n" not in message


def test_run_poc_reports_the_missing_energyplus_cleanly(monkeypatch, tmp_path):
    _no_energyplus(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        POC.main(["--all"])
    message = str(excinfo.value)
    assert "scripts/fetch_energyplus.sh" in message
    assert "\n" not in message


def test_find_energyplus_still_raises_file_not_found_for_library_callers(monkeypatch, tmp_path):
    """The library API keeps its exception type; only the CLI wraps it."""
    _no_energyplus(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        R.find_energyplus()


def test_the_runner_exits_without_a_traceback(monkeypatch, tmp_path):
    """The failing run prints the message and no traceback."""
    _no_energyplus(monkeypatch, tmp_path)
    prior = _write_prior(tmp_path, _valid_prior())
    script = (
        "import sys, shutil\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import run_ecms as RE, run_energyplus as R\n"
        "shutil.which = lambda name: None\n"
        "R.SCRATCH_DIR = __import__('pathlib').Path('/nonexistent-scratch')\n"
        f"RE.PRIOR_FILE = __import__('pathlib').Path({str(prior)!r})\n"
        "RE.main(['--all'])\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr
    assert "scripts/fetch_energyplus.sh" in proc.stderr
