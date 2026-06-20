import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_economics", ROOT / "experiments" / "run_economics.py")
assert SPEC is not None
run_economics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_economics)


def test_economics_summary_matches_reported_break_even() -> None:
    summary = run_economics._summary()
    costs = summary["costs"]

    assert costs["api_usd_per_request"] == 0.00016125
    assert costs["local_variable_usd_per_request"] == 0.00000851
    assert costs["break_even_requests_per_month"] == 127301
    assert costs["cached_break_even_requests_per_month"] == 138569


def test_cost_curve_contains_expected_volume_points() -> None:
    curve = run_economics._curve(run_economics._summary()["costs"])

    assert curve[0]["requests_per_month"] == 0
    assert curve[-1]["requests_per_month"] == 200_000
    assert curve[-1]["api_usd"] > curve[-1]["local_usd"]
