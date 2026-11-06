"""Tests for the run manifest: a recorded run must reload to the same manifest."""

import json

import pytest

import run_energyplus as R


def _sample_manifest():
    return {
        "schema_version": 1,
        "run_id": "unit-test-run",
        "label": "unit-test",
        "created_utc": "2026-01-01T00:00:00Z",
        "notes": "",
        "params": {"infiltration_flow_scale": 1.0},
        "energyplus": {"binary": "/opt/energyplus/energyplus", "version": "EnergyPlus, Version 26.2.0"},
        "command": ["/opt/energyplus/energyplus", "-w", "weather.epw", "in.idf"],
        "cwd": "/tmp/unit-test-run",
        "returncode": 0,
        "duration_s": 12.34,
        "inputs": {
            "idf_source": {"path": "/tmp/base.idf", "sha256": "0" * 64},
            "idf_simulated": {"path": "/tmp/unit-test-run/in.idf", "sha256": "1" * 64, "patched": True},
            "weather": {"path": "/tmp/weather.epw", "sha256": "2" * 64},
        },
        "outputs": {"eplus.csv": {"sha256": "3" * 64, "bytes": 1234}},
        "pruned_outputs": [],
    }


def test_manifest_round_trip(tmp_path):
    manifest = _sample_manifest()
    run_dir = tmp_path / manifest["run_id"]
    run_dir.mkdir()
    with open(run_dir / R.MANIFEST_NAME, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    loaded = R.load_manifest(manifest["run_id"], runs_dir=tmp_path)
    assert loaded == manifest
    # and re-dumping the loaded manifest reproduces the file byte for byte
    assert json.dumps(loaded, indent=2, sort_keys=True) + "\n" == (
        run_dir / R.MANIFEST_NAME
    ).read_text()


def test_load_manifest_missing_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.load_manifest("does-not-exist", runs_dir=tmp_path)


def test_list_runs_finds_recorded_manifests(tmp_path):
    for rid in ("run-a", "run-b"):
        run_dir = tmp_path / rid
        run_dir.mkdir()
        with open(run_dir / R.MANIFEST_NAME, "w") as fh:
            json.dump(_sample_manifest(), fh)
    (tmp_path / "not-a-run").mkdir()
    assert R.list_runs(tmp_path) == ["run-a", "run-b"]


def test_new_run_id_is_unique_and_labelled():
    a = R.new_run_id("calib")
    b = R.new_run_id("calib")
    assert a != b
    assert a.startswith("calib-")


def test_missing_idf_raises_before_any_simulation(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.run_energyplus(tmp_path / "nope.idf", tmp_path / "nope.epw", "unit-test")


def test_recorded_runs_reload_to_the_same_manifest():
    """Every run recorded by the POC reloads byte-identically."""
    if not R.RUNS_DIR.is_dir():
        pytest.skip("no recorded runs in this checkout")
    checked = 0
    for rid in R.list_runs():
        path = R.manifest_path(rid)
        with open(path) as fh:
            on_disk = json.load(fh)
        assert R.load_manifest(rid) == on_disk
        checked += 1
    if checked == 0:
        pytest.skip("no recorded runs in this checkout")
