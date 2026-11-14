"""The ``--idf`` / ``--epw`` options of both runners (issue #6).

The building and the weather used to be module constants (``IDF_NAME`` /
``EPW_NAME``).  Both runners now take them as command line options, default
to the very same constants, and resolve an explicit name against the
EnergyPlus install exactly like the constants were resolved:

* the IDF against ``install/ExampleFiles/<name>``
* the EPW against ``install/WeatherData/<name>``

Every run manifest records the file names it used next to the input SHAs.

None of these tests need EnergyPlus: the path resolution is exercised
against a stubbed install directory and the manifest is written with a
stubbed ``subprocess.run``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import run_ecms as RE
import run_energyplus as R
import run_poc as POC


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _stub_install(tmp_path: Path) -> Path:
    """A fake EnergyPlus install with the two default input files."""
    install = tmp_path / "install"
    (install / "ExampleFiles").mkdir(parents=True)
    (install / "WeatherData").mkdir(parents=True)
    (install / "ExampleFiles" / POC.IDF_NAME).write_text("!idf\n")
    (install / "WeatherData" / POC.EPW_NAME).write_text("!epw\n")
    return install


# ---------------------------------------------------------------------------
# the defaults
# ---------------------------------------------------------------------------
def test_both_runners_default_to_the_module_constants():
    assert POC.IDF_NAME == "5ZoneAirCooled.idf"
    assert POC.EPW_NAME == "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
    assert RE.IDF_NAME == POC.IDF_NAME
    assert RE.EPW_NAME == POC.EPW_NAME


@pytest.mark.parametrize("module", [POC, RE])
def test_the_idf_and_epw_options_default_to_the_constants(module):
    args = module.build_parser().parse_args([])
    assert args.idf == module.IDF_NAME
    assert args.epw == module.EPW_NAME


@pytest.mark.parametrize("module", [POC, RE])
def test_the_idf_and_epw_options_accept_explicit_values(module):
    args = module.build_parser().parse_args(["--idf", "Other.idf", "--epw", "Other.epw"])
    assert args.idf == "Other.idf"
    assert args.epw == "Other.epw"


@pytest.mark.parametrize("module", [POC, RE])
def test_the_help_text_shows_both_options(module):
    help_text = module.build_parser().format_help()
    assert "--idf" in help_text
    assert "--epw" in help_text


# ---------------------------------------------------------------------------
# the path resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", [POC, RE])
def test_the_default_names_resolve_against_the_install(module, tmp_path):
    install = _stub_install(tmp_path)
    idf, epw = module.resolve_input_paths(install, module.IDF_NAME, module.EPW_NAME)
    assert idf == install / "ExampleFiles" / module.IDF_NAME
    assert epw == install / "WeatherData" / module.EPW_NAME
    assert idf.is_file() and epw.is_file()


@pytest.mark.parametrize("module", [POC, RE])
def test_explicit_names_resolve_against_the_install(module, tmp_path):
    install = _stub_install(tmp_path)
    (install / "ExampleFiles" / "Other.idf").write_text("!idf\n")
    (install / "WeatherData" / "Other.epw").write_text("!epw\n")
    idf, epw = module.resolve_input_paths(install, "Other.idf", "Other.epw")
    assert idf == install / "ExampleFiles" / "Other.idf"
    assert epw == install / "WeatherData" / "Other.epw"
    assert idf.is_file() and epw.is_file()


@pytest.mark.parametrize("module", [POC, RE])
def test_the_resolution_is_the_same_mechanism_as_the_constants(module, tmp_path):
    """The resolver does exactly what the old constant lines did."""
    install = tmp_path / "install"
    idf, epw = module.resolve_input_paths(install, module.IDF_NAME, module.EPW_NAME)
    assert idf == install / "ExampleFiles" / module.IDF_NAME
    assert epw == install / "WeatherData" / module.EPW_NAME


@pytest.mark.parametrize("module", [POC, RE])
def test_an_absolute_path_is_used_as_given(module, tmp_path):
    install = _stub_install(tmp_path)
    idf = tmp_path / "elsewhere" / "MyBuilding.idf"
    epw = tmp_path / "elsewhere" / "MyWeather.epw"
    idf.parent.mkdir(parents=True)
    idf.write_text("!idf\n")
    epw.write_text("!epw\n")
    got_idf, got_epw = module.resolve_input_paths(install, str(idf), str(epw))
    assert got_idf == idf
    assert got_epw == epw


# ---------------------------------------------------------------------------
# the argument plumbing (no EnergyPlus needed)
# ---------------------------------------------------------------------------
def test_run_poc_uses_the_default_files_when_no_flag_is_given(monkeypatch, tmp_path):
    install = _stub_install(tmp_path)
    monkeypatch.setattr(R, "find_energyplus", lambda: install / "energyplus")
    monkeypatch.setattr(R, "energyplus_version", lambda binary: "EnergyPlus 1.0")
    monkeypatch.setattr(POC, "find_install_dir", lambda binary: install)

    seen = {}

    def fake_run_one(idf, epw, label, params, notes, prune=False):
        seen.setdefault("idf", idf)
        seen.setdefault("epw", epw)
        raise SystemExit("stop after the first run")

    monkeypatch.setattr(POC, "run_one", fake_run_one)
    with pytest.raises(SystemExit):
        POC.main(["--all"])
    assert seen["idf"] == install / "ExampleFiles" / POC.IDF_NAME
    assert seen["epw"] == install / "WeatherData" / POC.EPW_NAME


def test_run_poc_threads_the_explicit_files_through(monkeypatch, tmp_path):
    install = _stub_install(tmp_path)
    (install / "ExampleFiles" / "Other.idf").write_text("!idf\n")
    (install / "WeatherData" / "Other.epw").write_text("!epw\n")
    monkeypatch.setattr(R, "find_energyplus", lambda: install / "energyplus")
    monkeypatch.setattr(R, "energyplus_version", lambda binary: "EnergyPlus 1.0")
    monkeypatch.setattr(POC, "find_install_dir", lambda binary: install)

    seen = {}

    def fake_run_one(idf, epw, label, params, notes, prune=False):
        seen.setdefault("idf", idf)
        seen.setdefault("epw", epw)
        raise SystemExit("stop after the first run")

    monkeypatch.setattr(POC, "run_one", fake_run_one)
    with pytest.raises(SystemExit):
        POC.main(["--all", "--idf", "Other.idf", "--epw", "Other.epw"])
    assert seen["idf"] == install / "ExampleFiles" / "Other.idf"
    assert seen["epw"] == install / "WeatherData" / "Other.epw"


def _stub_ecms_inputs(monkeypatch, tmp_path):
    """The pre-flight inputs of the slice-2 runner, all valid."""
    install = _stub_install(tmp_path)
    prior = tmp_path / "llm_prior_ranking.json"
    prior.write_text(
        json.dumps({"ranking": [{"ecm_id": ecm_id} for ecm_id in RE.E.ECM_IDS]})
    )
    monkeypatch.setattr(RE, "PRIOR_FILE", prior)
    monkeypatch.setattr(RE, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(R, "find_energyplus", lambda: install / "energyplus")
    monkeypatch.setattr(R, "energyplus_version", lambda binary: "EnergyPlus 1.0")
    monkeypatch.setattr(RE, "find_install_dir", lambda binary: install)
    return install


def test_run_ecms_uses_the_default_files_when_no_flag_is_given(monkeypatch, tmp_path):
    install = _stub_ecms_inputs(monkeypatch, tmp_path)

    seen = {}

    def fake_run_one(idf, epw, label, idf_text, notes, params):
        seen.setdefault("idf", idf)
        seen.setdefault("epw", epw)
        raise SystemExit("stop after the first run")

    monkeypatch.setattr(RE, "run_one", fake_run_one)
    with pytest.raises(SystemExit):
        RE.main(["--all"])
    assert seen["idf"] == install / "ExampleFiles" / RE.IDF_NAME
    assert seen["epw"] == install / "WeatherData" / RE.EPW_NAME


def test_run_ecms_threads_the_explicit_files_through(monkeypatch, tmp_path):
    install = _stub_ecms_inputs(monkeypatch, tmp_path)
    (install / "ExampleFiles" / "Other.idf").write_text("!idf\n")
    (install / "WeatherData" / "Other.epw").write_text("!epw\n")

    seen = {}

    def fake_run_one(idf, epw, label, idf_text, notes, params):
        seen.setdefault("idf", idf)
        seen.setdefault("epw", epw)
        raise SystemExit("stop after the first run")

    monkeypatch.setattr(RE, "run_one", fake_run_one)
    with pytest.raises(SystemExit):
        RE.main(["--all", "--idf", "Other.idf", "--epw", "Other.epw"])
    assert seen["idf"] == install / "ExampleFiles" / "Other.idf"
    assert seen["epw"] == install / "WeatherData" / "Other.epw"


# ---------------------------------------------------------------------------
# the manifest records the file names
# ---------------------------------------------------------------------------
def _stub_subprocess(monkeypatch):
    """Replace the EnergyPlus call with a stub that writes one output."""

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "eplus.csv").write_text("a,b\n")
        return _Proc()

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    monkeypatch.setattr(R, "find_energyplus", lambda *a, **k: Path("/fake/energyplus"))
    monkeypatch.setattr(R, "energyplus_version", lambda binary: "EnergyPlus 1.0")


def test_the_manifest_records_the_idf_and_epw_file_names(monkeypatch, tmp_path):
    _stub_subprocess(monkeypatch)
    idf = tmp_path / "MyBuilding.idf"
    epw = tmp_path / "MyWeather.epw"
    idf.write_text("!idf\n")
    epw.write_text("!epw\n")
    manifest = R.run_energyplus(
        idf, epw, "unit-test", runs_dir=tmp_path / "runs"
    )
    assert manifest["inputs"]["idf_source"]["name"] == "MyBuilding.idf"
    assert manifest["inputs"]["weather"]["name"] == "MyWeather.epw"
    # the names sit next to the input SHAs
    assert manifest["inputs"]["idf_source"]["sha256"]
    assert manifest["inputs"]["weather"]["sha256"]


def test_the_manifest_records_the_names_of_the_default_files(monkeypatch, tmp_path):
    _stub_subprocess(monkeypatch)
    install = _stub_install(tmp_path)
    manifest = R.run_energyplus(
        install / "ExampleFiles" / POC.IDF_NAME,
        install / "WeatherData" / POC.EPW_NAME,
        "unit-test",
        runs_dir=tmp_path / "runs",
    )
    assert manifest["inputs"]["idf_source"]["name"] == POC.IDF_NAME
    assert manifest["inputs"]["weather"]["name"] == POC.EPW_NAME
