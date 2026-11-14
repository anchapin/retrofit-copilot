"""CLI contract of ``src/run_poc.py`` (issue #10).

``run_poc.py`` is a driver: with no arguments it has nothing to do, so it
must fail with an argparse usage error (exit code 2) *before* it looks for
the EnergyPlus binary.  ``--max-sweeps`` must be a positive number, and the
final calibrated run must fail loudly when EnergyPlus refuses to simulate it.

None of these tests need EnergyPlus: the argument checks happen before the
binary lookup, and the failing calibrated run is stubbed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import calibrate as C
import results as RES
import run_energyplus as R
import run_poc as POC


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _no_energyplus(monkeypatch) -> None:
    """Make any EnergyPlus lookup an immediate, loud failure."""
    def boom(*args, **kwargs):
        raise AssertionError("EnergyPlus must not be looked up")

    monkeypatch.setattr(R, "find_energyplus", boom)


def _stub_energyplus(monkeypatch, tmp_path: Path) -> Path:
    """Fake an EnergyPlus install with the two input files."""
    install = tmp_path / "install"
    (install / "ExampleFiles").mkdir(parents=True)
    (install / "WeatherData").mkdir(parents=True)
    (install / "ExampleFiles" / POC.IDF_NAME).write_text("!idf\n")
    (install / "WeatherData" / POC.EPW_NAME).write_text("!epw\n")
    monkeypatch.setattr(R, "find_energyplus", lambda: install / "energyplus")
    monkeypatch.setattr(R, "energyplus_version", lambda binary: "EnergyPlus 1.0")
    monkeypatch.setattr(POC, "find_install_dir", lambda binary: install)
    return install


def _stub_series(monkeypatch) -> None:
    """A fake hourly series, so no run directory has to exist."""
    months = np.array([h // 730 + 1 for h in range(8760)])
    hours = np.arange(8760)

    def load_run_series(run_dir, run_id=None):
        return RES.RunSeries(
            run_id=run_id or "stub",
            months=months,
            hours=hours,
            columns={
                "facility energy": np.zeros(8760),
                "ZONE ONE SPACE:Zone Air Temperature [C](Hourly)": np.full(8760, 20.0),
            },
        )

    monkeypatch.setattr(RES, "load_run_series", load_run_series)
    energy = np.arange(8760, dtype=float) % 97 + 1.0
    monkeypatch.setattr(
        RES, "facility_energy_kwh", lambda series: energy
    )


# ---------------------------------------------------------------------------
# the --all guard
# ---------------------------------------------------------------------------
def test_no_arguments_is_a_usage_error(monkeypatch, capsys):
    """No args: argparse usage error, before any EnergyPlus lookup."""
    _no_energyplus(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        POC.main([])
    assert excinfo.value.code == 2
    assert "pass --all" in capsys.readouterr().err


def test_the_all_guard_runs_before_the_energyplus_lookup(monkeypatch):
    """The guard must fire first: no EnergyPlus lookup may happen."""
    _no_energyplus(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        POC.main([])
    assert excinfo.value.code == 2


def test_an_unknown_flag_is_a_usage_error(monkeypatch, capsys):
    _no_energyplus(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        POC.main(["--not-a-flag"])
    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the --max-sweeps validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sweeps", ["0", "-1", "-3"])
def test_a_non_positive_max_sweeps_is_a_usage_error(monkeypatch, capsys, sweeps):
    """--max-sweeps must be >= 1, checked before any simulation."""
    _no_energyplus(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        POC.main(["--all", "--max-sweeps", sweeps])
    assert excinfo.value.code == 2
    assert "--max-sweeps must be >= 1" in capsys.readouterr().err


def test_a_positive_max_sweeps_passes_the_validation(monkeypatch):
    """A valid --max-sweeps gets past the argument checks."""
    _no_energyplus(monkeypatch)
    with pytest.raises(AssertionError):
        # the EnergyPlus lookup is the next step -> it must be reached
        POC.main(["--all", "--max-sweeps", "3"])


# ---------------------------------------------------------------------------
# the calibrated run
# ---------------------------------------------------------------------------
def _run_poc_with_stubbed_runs(monkeypatch, tmp_path, calibrated_returncode):
    """Run main() with every EnergyPlus run stubbed out."""
    _stub_energyplus(monkeypatch, tmp_path)
    _stub_series(monkeypatch)

    calls = []

    def fake_run_one(idf, epw, label, params, notes, prune=False):
        calls.append(label)
        returncode = calibrated_returncode if label == "calibrated" else 0
        return {
            "run_id": f"stub-{label}",
            "cwd": str(tmp_path),
            "params": dict(params),
            "returncode": returncode,
            "failure": None if returncode == 0 else "EnergyPlus exited with errors",
        }

    monkeypatch.setattr(POC, "run_one", fake_run_one)
    monkeypatch.setattr(
        C,
        "coordinate_search",
        lambda *a, **kw: C.SearchResult(
            best_params=dict(POC.P.DEFAULT_PARAMS),
            best_objective=0.0,
            steps=[],
            evaluated=[],
        ),
    )
    return calls


def test_a_failed_calibrated_run_raises(monkeypatch, tmp_path):
    """A non-zero calibrated return code is an explicit, loud failure."""
    _run_poc_with_stubbed_runs(monkeypatch, tmp_path, calibrated_returncode=1)
    with pytest.raises(RuntimeError) as excinfo:
        POC.main(["--all", "--max-sweeps", "3"])
    message = str(excinfo.value)
    assert "calibrated EnergyPlus run failed" in message


def test_a_failed_calibrated_run_stops_the_pipeline(monkeypatch, tmp_path):
    """The failure happens before the metrics/table are produced."""
    calls = _run_poc_with_stubbed_runs(monkeypatch, tmp_path, calibrated_returncode=1)
    with pytest.raises(RuntimeError):
        POC.main(["--all", "--max-sweeps", "3"])
    assert calls[-1] == "calibrated"


def test_a_successful_calibrated_run_does_not_raise(monkeypatch, tmp_path):
    """A zero return code is fine: no RuntimeError."""
    _run_poc_with_stubbed_runs(monkeypatch, tmp_path, calibrated_returncode=0)
    out = tmp_path / "headline.md"
    assert POC.main(["--all", "--max-sweeps", "3", "--out", str(out)]) == 0
    assert out.is_file()
