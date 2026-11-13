"""Tests for the ``--rank-by {kwh,cost,carbon}`` headline ranking (issue #4).

The ranking re-presents metrics the pipeline already computes (annual kWh,
annual USD saved, annual tCO2 saved); no new simulation is involved.
"""

import json
from pathlib import Path

import pytest

import ecms as E
import run_ecms as RE

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _metrics():
    """Synthetic per-ECM metrics with a negative-savings ECM (the E4 case).

    ``E4`` saves plenty of kWh but *costs* money and emits CO2, so it is
    #3 by kWh and last by cost and by carbon.
    """
    return {
        "E1": {
            "energy_saved_kwh": 4790.3,
            "cost_saved_usd": 346.54,
            "co2_saved_tco2": 1.023,
            "simple_payback_years": 127.7,
        },
        "E2": {
            "energy_saved_kwh": 6357.0,
            "cost_saved_usd": 467.16,
            "co2_saved_tco2": 1.377,
            "simple_payback_years": 58.7,
        },
        "E3": {
            "energy_saved_kwh": 3819.2,
            "cost_saved_usd": 300.99,
            "co2_saved_tco2": 0.882,
            "simple_payback_years": 149.5,
        },
        "E4": {
            "energy_saved_kwh": 4282.5,
            "cost_saved_usd": -942.15,
            "co2_saved_tco2": -2.484,
            "simple_payback_years": None,
        },
        "E5": {
            "energy_saved_kwh": 3751.1,
            "cost_saved_usd": 267.18,
            "co2_saved_tco2": 0.790,
            "simple_payback_years": 9.4,
        },
    }


# ---------------------------------------------------------------------------
# the ordering logic
# ---------------------------------------------------------------------------
def test_the_default_ranking_is_by_annual_kwh_saved():
    order = RE.ranking_from_metrics(_metrics())
    assert order == ["E2", "E1", "E4", "E3", "E5"]


def test_the_ranking_by_cost_puts_the_negative_savings_ecm_last():
    order = RE.ranking_from_metrics(_metrics(), rank_by="cost")
    assert order == ["E2", "E1", "E3", "E5", "E4"]
    assert order[-1] == "E4"  # negative $ saved sorts last


def test_the_ranking_by_carbon_puts_the_negative_savings_ecm_last():
    order = RE.ranking_from_metrics(_metrics(), rank_by="carbon")
    assert order == ["E2", "E1", "E3", "E5", "E4"]
    assert order[-1] == "E4"  # negative tCO2 saved sorts last


def test_the_negative_savings_ecm_keeps_its_kwh_rank():
    metrics = _metrics()
    kwh_order = RE.ranking_from_metrics(metrics, rank_by="kwh")
    assert kwh_order.index("E4") == 2  # #3 by kWh ...
    assert RE.ranking_from_metrics(metrics, rank_by="cost").index("E4") == 4  # ... last by cost


def test_the_ranking_breaks_ties_by_ecm_id():
    metrics = {
        "E2": {"energy_saved_kwh": 3000.0, "cost_saved_usd": 10.0, "co2_saved_tco2": 1.0},
        "E1": {"energy_saved_kwh": 3000.0, "cost_saved_usd": 10.0, "co2_saved_tco2": 1.0},
    }
    assert RE.ranking_from_metrics(metrics) == ["E1", "E2"]
    assert RE.ranking_from_metrics(metrics, rank_by="cost") == ["E1", "E2"]
    assert RE.ranking_from_metrics(metrics, rank_by="carbon") == ["E1", "E2"]


def test_an_unknown_ranking_metric_is_rejected():
    with pytest.raises(ValueError):
        RE.ranking_from_metrics(_metrics(), rank_by="nonsense")


def test_the_rank_by_choices_are_the_three_metrics():
    assert set(RE.RANK_BY_CHOICES) == {"kwh", "cost", "carbon"}


# ---------------------------------------------------------------------------
# the headline table
# ---------------------------------------------------------------------------
def _headline(rank_by=None):
    """Render the headline table of the recorded slice-2 fixture."""
    data = json.loads((FIXTURES / "ecm_results.json").read_text())
    metrics = data["metrics"]
    kwargs = {} if rank_by is None else {"rank_by": rank_by}
    rankings = {
        model: RE.ranking_from_metrics(metrics[model], **kwargs)
        for model in ("calibrated", "true")
    }
    lines = RE.headline_table_lines(
        version=data["energyplus_version"],
        agent_ranking=rankings["calibrated"],
        truth_ranking=rankings["true"],
        prior_ranking=list(data["llm_prior"]["ranking"]),
        model_metrics=metrics,
        runs=data["runs"],
        factors=E.default_factors(),
        **kwargs,
    )
    return "\n".join(lines)


def _heading(table: str) -> str:
    for line in table.splitlines():
        if line.startswith("## Agent ranking"):
            return line
    raise AssertionError("no agent ranking heading found")


def test_the_default_headline_table_names_the_kwh_metric():
    assert _heading(_headline()) == (
        "## Agent ranking (calibrated model), by annual facility kWh saved"
    )


def test_the_cost_headline_table_names_the_cost_metric():
    heading = _heading(_headline("cost"))
    assert "USD saved" in heading
    assert "kWh" not in heading


def test_the_carbon_headline_table_names_the_carbon_metric():
    heading = _heading(_headline("carbon"))
    assert "tCO2 saved" in heading
    assert "kWh" not in heading


def test_the_cost_headline_table_ranks_the_negative_savings_ecm_last():
    table = _headline("cost")
    lines = [ln for ln in table.splitlines() if ln.startswith("| ") and "E4 " in ln]
    agent_rows = lines[:1]
    assert agent_rows, "no E4 row found"
    assert agent_rows[0].split("|")[1].strip() == "5"


def test_the_default_headline_table_ranking_is_unchanged():
    """The recorded slice-2 ranking (by kWh) is reproduced exactly."""
    data = json.loads((FIXTURES / "ecm_results.json").read_text())
    assert RE.ranking_from_metrics(data["metrics"]["calibrated"]) == data["rankings"][
        "agent_calibrated"
    ]
    assert RE.ranking_from_metrics(data["metrics"]["true"]) == data["rankings"][
        "ground_truth_true"
    ]


def test_the_default_headline_table_is_byte_identical_to_the_recorded_one():
    """The default rendering has not changed a single byte.

    The default table is compared against the same rendering built from the
    recorded fixture metrics with the historic, metric-independent column
    layout (kWh saved / % saved / run id).
    """
    data = json.loads((FIXTURES / "ecm_results.json").read_text())
    metrics = data["metrics"]
    lines = RE.headline_table_lines(
        version=data["energyplus_version"],
        agent_ranking=data["rankings"]["agent_calibrated"],
        truth_ranking=data["rankings"]["ground_truth_true"],
        prior_ranking=list(data["llm_prior"]["ranking"]),
        model_metrics=metrics,
        runs=data["runs"],
        factors=E.default_factors(),
    )
    assert _headline() == "\n".join(lines)


def test_the_default_ranking_tables_are_byte_identical_to_the_recorded_ones():
    """The default ranking tables equal the ones recorded in REPORT.md.

    REPORT.md holds the recorded slice-2 tables; the default rendering of
    the headline table must reproduce them byte for byte.
    """
    report = (Path(__file__).resolve().parents[1] / "REPORT.md").read_text()
    table = _headline()
    for heading in (
        "## Agent ranking (calibrated model), by annual facility kWh saved",
        "## Ground-truth ranking (true model), by annual facility kWh saved",
    ):
        recorded = report[report.index(heading):]
        recorded = recorded[: recorded.index("\n\n")]
        assert heading in table
        rendered = table[table.index(heading):]
        rendered = rendered[: rendered.index("\n\n")]
        assert rendered == recorded


def test_the_command_line_exposes_the_rank_by_option():
    """``--rank-by {kwh,cost,carbon}`` exists on the command line."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "src" / "run_ecms.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--rank-by" in proc.stdout
    assert "{kwh,cost,carbon}" in proc.stdout


def test_an_unknown_rank_by_value_is_rejected_by_the_command_line():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "src" / "run_ecms.py"), "--rank-by", "nonsense"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--rank-by" in proc.stderr


def test_the_n_a_payback_stays_n_a_in_the_cost_ranking():
    table = _headline("cost")
    assert "n/a" in table
    for line in table.splitlines():
        if line.startswith("| E4 |"):
            assert "| n/a |" in line
