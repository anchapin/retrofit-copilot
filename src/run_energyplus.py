"""Run the local EnergyPlus binary on an IDF file and record the run.

Every run is recorded: a run directory under ``runs/<run_id>/``
holds the exact IDF that was simulated, the EnergyPlus outputs, and a
``manifest.json`` mapping run id -> command -> input files -> outputs.

Nothing in here re-runs a simulation silently: a run id is allocated once, the
manifest is written once, and re-reading a manifest returns exactly what was
written (round-trip tested in ``tests/test_manifest.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

POC_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = POC_ROOT / "runs"
SCRATCH_DIR = POC_ROOT / ".scratch"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# locating the binary
# --------------------------------------------------------------------------
def find_energyplus(explicit: Optional[str] = None) -> Path:
    """Return the path to the EnergyPlus executable.

    Search order: explicit argument, ``$ENERGYPLUS_BIN``, the worktree scratch
    install, then ``$PATH``.  Raises FileNotFoundError with the searched
    locations when nothing is found.
    """
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_bin = os.environ.get("ENERGYPLUS_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    if SCRATCH_DIR.is_dir():
        for child in sorted(SCRATCH_DIR.glob("EnergyPlus-*/energyplus")):
            candidates.append(child)
    found_on_path = shutil.which("energyplus")
    if found_on_path:
        candidates.append(Path(found_on_path))
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    raise FileNotFoundError(
        "EnergyPlus executable not found. Install it with "
        "`bash scripts/install_energyplus.sh` (or set "
        "$ENERGYPLUS_BIN). Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def energyplus_version(binary: Optional[Path] = None) -> str:
    """Return the version string reported by ``energyplus --version``."""
    exe = binary or find_energyplus()
    out = subprocess.run(
        [str(exe), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "run"
    return f"{safe}-{stamp}-{uuid.uuid4().hex[:6]}"


def run_dir(run_id: str, runs_dir: Optional[Path] = None) -> Path:
    return (runs_dir or RUNS_DIR) / run_id


def manifest_path(run_id: str, runs_dir: Optional[Path] = None) -> Path:
    return run_dir(run_id, runs_dir) / MANIFEST_NAME


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def run_energyplus(
    idf: Path,
    epw: Path,
    label: str,
    *,
    run_id: Optional[str] = None,
    idf_text: Optional[str] = None,
    extra_args: Sequence[str] = (),
    notes: str = "",
    params: Optional[Dict[str, Any]] = None,
    binary: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
    timeout_s: int = 3600,
    prune_globs: Sequence[str] = (),
) -> Dict[str, Any]:
    """Simulate ``idf`` with ``epw`` and write a manifest for the run.

    ``idf_text`` (optional) overrides the IDF content that is written into the
    run directory; this is how perturbed / calibrated models are simulated
    without touching the original example file.
    """
    idf = Path(idf)
    epw = Path(epw)
    if not idf.is_file():
        raise FileNotFoundError(f"IDF not found: {idf}")
    if not epw.is_file():
        raise FileNotFoundError(f"EPW not found: {epw}")

    exe = find_energyplus(str(binary) if binary else None)
    version = energyplus_version(exe)

    runs_root = Path(runs_dir) if runs_dir else RUNS_DIR
    runs_root.mkdir(parents=True, exist_ok=True)
    rid = run_id or new_run_id(label)
    rdir = runs_root / rid
    if rdir.exists():
        raise FileExistsError(f"run directory already exists: {rdir}")
    rdir.mkdir(parents=True)

    content = idf_text if idf_text is not None else idf.read_text()
    sim_idf = rdir / "in.idf"
    sim_idf.write_text(content)

    command = [
        str(exe),
        "-w",
        str(epw.resolve()),
        "-d",
        str(rdir.resolve()),
        "-p",
        "eplus",
        "-s",
        "C",
        "-r",
        *extra_args,
        str(sim_idf.resolve()),
    ]

    stdout_path = rdir / "energyplus.stdout.log"
    start = time.time()
    with open(stdout_path, "w") as log:
        proc = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(rdir),
            timeout=timeout_s,
        )
    duration = time.time() - start

    pruned: List[str] = []
    for pattern in prune_globs:
        for path in sorted(rdir.glob(pattern)):
            if path.is_file():
                path.unlink()
                pruned.append(path.name)

    outputs: Dict[str, Any] = {}
    for path in sorted(rdir.iterdir()):
        if path.is_file() and path.name not in (MANIFEST_NAME,):
            outputs[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": rid,
        "label": label,
        "created_utc": utc_now_iso(),
        "notes": notes,
        "params": params or {},
        "energyplus": {
            "binary": str(exe),
            "version": version,
        },
        "command": command,
        "cwd": str(rdir.resolve()),
        "returncode": proc.returncode,
        "duration_s": round(duration, 2),
        "inputs": {
            "idf_source": {
                "path": str(idf.resolve()),
                "sha256": sha256_file(idf),
            },
            "idf_simulated": {
                "path": str(sim_idf.resolve()),
                "sha256": sha256_file(sim_idf),
                "patched": idf_text is not None,
            },
            "weather": {
                "path": str(epw.resolve()),
                "sha256": sha256_file(epw),
            },
        },
        "outputs": outputs,
        "pruned_outputs": pruned,
    }

    if proc.returncode != 0:
        manifest["failure"] = (
            f"energyplus exited with code {proc.returncode}; "
            f"see {stdout_path.name}"
        )

    with open(manifest_path(rid, runs_root), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


# --------------------------------------------------------------------------
# manifest access
# --------------------------------------------------------------------------
def load_manifest(run_id: str, runs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a recorded manifest.  Round-trips to the exact dict written."""
    path = manifest_path(run_id, runs_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no manifest for run id {run_id!r}: {path}")
    with open(path) as fh:
        return json.load(fh)


def list_runs(runs_dir: Optional[Path] = None) -> List[str]:
    root = Path(runs_dir) if runs_dir else RUNS_DIR
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / MANIFEST_NAME).is_file())


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import argparse

    ap = argparse.ArgumentParser(description="Run EnergyPlus and record the run.")
    ap.add_argument("--idf", required=True, type=Path)
    ap.add_argument("--epw", required=True, type=Path)
    ap.add_argument("--label", default="manual")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    man = run_energyplus(args.idf, args.epw, args.label, notes=args.note)
    print(json.dumps(man, indent=2, sort_keys=True))
