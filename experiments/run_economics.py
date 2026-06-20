"""Generate on-prem versus API economics artifacts for the benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
INPUT_TOKENS = 23
OUTPUT_TOKENS = 32
RUNTIME_SECONDS = 13.4804
MAX_REQUESTS = 200_000
MAX_MONTHLY_COST = 33
VOLUMES = [0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000]
ASSUMPTIONS = {
    "pricing_accessed": "2026-06-20",
    "api_provider": "OpenAI",
    "api_model": "GPT-5.4 mini",
    "api_input_usd_per_1m": 0.75,
    "api_cached_input_usd_per_1m": 0.075,
    "api_output_usd_per_1m": 4.50,
    "cached_input_fraction": 0.80,
    "hardware_cost_usd": 700.0,
    "hardware_lifetime_months": 36,
    "electricity_usd_per_kwh": 0.20,
    "estimated_power_watts": 45.0,
}


def main() -> None:
    """Write the Phase 8 economics table, summary JSON, and break-even figure."""

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    summary = _summary()
    curve = _curve(summary["costs"])
    (RESULTS / "economics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(RESULTS / "economics_curve.csv", curve)
    _write_svg(FIGURES / "cost_break_even.svg", curve, summary["costs"])


def _summary() -> dict[str, object]:
    local_variable = _local_variable_cost()
    hardware_monthly = _monthly_hardware_cost()
    api_request = _api_request_cost(cached=False)
    cached_api_request = _api_request_cost(cached=True)
    costs = {
        "local_variable_usd_per_request": round(local_variable, 8),
        "local_fixed_usd_per_month": round(hardware_monthly, 4),
        "api_usd_per_request": round(api_request, 8),
        "api_cached_usd_per_request": round(cached_api_request, 8),
        "break_even_requests_per_month": _break_even(
            hardware_monthly, local_variable, api_request
        ),
        "cached_break_even_requests_per_month": _break_even(
            hardware_monthly, local_variable, cached_api_request
        ),
    }
    return {
        "assumptions": ASSUMPTIONS,
        "workload": {
            "input_tokens": INPUT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "runtime_seconds": RUNTIME_SECONDS,
        },
        "costs": costs,
    }


def _monthly_hardware_cost() -> float:
    return ASSUMPTIONS["hardware_cost_usd"] / ASSUMPTIONS["hardware_lifetime_months"]


def _local_variable_cost() -> float:
    watt_seconds = ASSUMPTIONS["estimated_power_watts"] * RUNTIME_SECONDS
    return watt_seconds / 3_600_000 * ASSUMPTIONS["electricity_usd_per_kwh"]


def _api_request_cost(cached: bool) -> float:
    input_price = ASSUMPTIONS["api_input_usd_per_1m"]
    if cached:
        cached_part = INPUT_TOKENS * ASSUMPTIONS["cached_input_fraction"]
        normal_part = INPUT_TOKENS - cached_part
        input_tokens_cost = normal_part * input_price
        input_tokens_cost += cached_part * ASSUMPTIONS["api_cached_input_usd_per_1m"]
    else:
        input_tokens_cost = INPUT_TOKENS * input_price
    output_tokens_cost = OUTPUT_TOKENS * ASSUMPTIONS["api_output_usd_per_1m"]
    return (input_tokens_cost + output_tokens_cost) / 1_000_000


def _break_even(fixed: float, local_variable: float, api_request: float) -> int | None:
    savings = api_request - local_variable
    return None if savings <= 0 else round(fixed / savings)


def _curve(costs: dict[str, float]) -> list[dict[str, float]]:
    fixed = float(costs["local_fixed_usd_per_month"])
    local_var = float(costs["local_variable_usd_per_request"])
    api_var = float(costs["api_usd_per_request"])
    cached_api_var = float(costs["api_cached_usd_per_request"])
    return [
        {
            "requests_per_month": volume,
            "local_usd": round(fixed + volume * local_var, 4),
            "api_usd": round(volume * api_var, 4),
            "api_cached_usd": round(volume * cached_api_var, 4),
        }
        for volume in VOLUMES
    ]


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_svg(path: Path, rows: list[dict[str, float]], costs: dict[str, float]) -> None:
    left, top, plot_w, plot_h = 78, 70, 760, 270
    lines = [_svg_header()]
    for index, (key, color, label) in enumerate(_legend()):
        points = " ".join(_point(row, key, left, top, plot_w, plot_h) for row in rows)
        lines.append(_polyline(points, color))
        lines.append(_legend_text(left + 20, top + 295 + 22 * index, color, label))
    be = costs["break_even_requests_per_month"]
    lines.append(_text(left, 382, "note", f"Break-even: about {be:,} requests/month"))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _legend() -> list[tuple[str, str, str]]:
    return [
        ("local_usd", "#2f855a", "Local on-prem"),
        ("api_usd", "#2b6cb0", "API"),
        ("api_cached_usd", "#805ad5", "API with cached input"),
    ]


def _point(row: dict[str, float], key: str, left: int, top: int, width: int, height: int) -> str:
    x = left + row["requests_per_month"] / MAX_REQUESTS * width
    y = top + height - row[key] / MAX_MONTHLY_COST * height
    return f"{x:.1f},{y:.1f}"


def _svg_header() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="920" height="420"
viewBox="0 0 920 420">
<style>
.title{font:700 26px Arial;fill:#1a202c}
.note{font:15px Arial;fill:#2d3748}
text{font:14px Arial;fill:#2d3748}
</style>
<rect width="100%" height="100%" fill="#f7fafc"/>
<text x="24" y="38" class="title">On-prem versus API monthly cost</text>
<line x1="78" y1="340" x2="838" y2="340" stroke="#718096"/>
<line x1="78" y1="70" x2="78" y2="340" stroke="#718096"/>
<text x="360" y="395">Requests per month</text>
<text x="12" y="210" transform="rotate(-90 12,210)">Monthly cost (USD)</text>"""


def _polyline(points: str, color: str) -> str:
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'


def _legend_text(x: int, y: int, color: str, value: str) -> str:
    return f'<text x="{x}" y="{y}" fill="{color}">{value}</text>'


def _text(x: int, y: int, class_name: str, value: str) -> str:
    return f'<text x="{x}" y="{y}" class="{class_name}">{value}</text>'


if __name__ == "__main__":
    main()
