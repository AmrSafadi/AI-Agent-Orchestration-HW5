"""Generate simple SVG figures from saved benchmark JSON files."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = [
    ("Tiny GPT-2", ROOT / "results" / "baseline_tiny_gpt2.json"),
    ("Qwen HF", ROOT / "results" / "baseline_qwen_qwen2_5_3b_instruct.json"),
    ("AirLLM Qwen", ROOT / "results" / "airllm_qwen_qwen2_5_3b_instruct.json"),
    ("AirLLM Phi", ROOT / "results" / "airllm_phi3_mini_instruct.json"),
    ("Ollama Q4", ROOT / "results" / "gguf_qwen2_5_3b_instruct_q4_k_m.json"),
]
COLORS = {
    "success": "#2f855a",
    "timeout": "#b7791f",
    "failed": "#c53030",
    "missing": "#718096",
}


def main() -> None:
    """Generate Phase 7 comparison figures without rerunning experiments."""

    FIGURES.mkdir(exist_ok=True)
    rows = [(label, _load(path)) for label, path in RESULTS]
    charts = [
        ("throughput_comparison.svg", "Throughput", "Output tokens/sec", "tokens_per_second"),
        ("decode_latency_comparison.svg", "Decode latency", "TPOT seconds", "tpot_seconds"),
        ("memory_comparison.svg", "Peak RAM comparison", "Peak RAM in MB", "peak_ram_mb"),
    ]
    for filename, title, subtitle, metric in charts:
        _write_bar_chart(FIGURES / filename, title, subtitle, _metric_rows(rows, metric))
    _write_status_chart(FIGURES / "run_status_summary.svg", rows)

def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _metric_rows(rows: list[tuple[str, dict[str, Any]]], metric: str) -> list[dict[str, Any]]:
    output = []
    for label, row in rows:
        value = row.get(metric)
        output.append(
            {
                "label": label,
                "value": value if isinstance(value, int | float) else None,
                "status": str(row.get("status") or "missing"),
            }
        )
    return output


def _write_bar_chart(path: Path, title: str, subtitle: str, rows: list[dict[str, Any]]) -> None:
    chart_x, chart_y, chart_w, bar_h, gap = 190, 82, 620, 42, 22
    values = [row["value"] for row in rows if row["value"] is not None]
    max_value = max(values) if values else 1
    parts = [_svg_header(920, 420, title, subtitle)]
    for index, row in enumerate(rows):
        y = chart_y + index * (bar_h + gap)
        parts.append(_text(24, y + 27, "label", row["label"]))
        value, status = row["value"], row["status"]
        if value is None:
            parts.append(_rect(chart_x, y, 120, bar_h, "missing"))
            parts.append(_text(chart_x + 134, y + 27, "note", f"n/a ({status})"))
            continue
        width = max(6, int((float(value) / max_value) * chart_w))
        parts.append(_rect(chart_x, y, width, bar_h, fill=COLORS.get(status, COLORS["missing"])))
        parts.append(_text(chart_x + width + 14, y + 27, "value", _fmt(value)))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_status_chart(path: Path, rows: list[tuple[str, dict[str, Any]]]) -> None:
    parts = [_svg_header(920, 420, "Run status summary", "Completed, timed out, and failed runs")]
    for index, (label, row) in enumerate(rows):
        status = str(row.get("status") or "missing")
        color = COLORS.get(status, COLORS["missing"])
        x, y = 58 + index * 172, 96
        parts.extend(
            [
                _rect(x, y, 150, 190, fill="#ffffff", stroke="#cbd5e0"),
                f'<circle cx="{x + 75}" cy="{y + 54}" r="28" fill="{color}" />',
                _text(x + 75, y + 103, "status", status, anchor="middle"),
                _text(x + 75, y + 136, "cardlabel", label, anchor="middle"),
                _text(x + 75, y + 164, "small", _evidence(row), anchor="middle"),
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _evidence(row: dict[str, Any]) -> str:
    if row.get("status") == "success":
        return f'{row.get("output_tokens") or "?"} output tokens'
    if row.get("status") == "timeout":
        return "900 s timeout"
    error = str(row.get("error") or "no generation")
    if error.startswith("IndexError"):
        return "layout issue"
    if error.startswith("NotImplementedError"):
        return "unsupported phi3"
    return "failed"


def _svg_header(width: int, height: int, title: str, subtitle: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 26px Arial, sans-serif; fill: #1a202c; }}
  .subtitle {{ font: 15px Arial, sans-serif; fill: #4a5568; }}
  .label {{ font: 15px Arial, sans-serif; fill: #2d3748; }}
  .value {{ font: 15px Arial, sans-serif; fill: #1a202c; }}
  .note {{ font: 14px Arial, sans-serif; fill: #4a5568; }}
  .missing {{ fill: #e2e8f0; }}
  .status {{ font: 700 16px Arial, sans-serif; fill: #1a202c; }}
  .cardlabel {{ font: 14px Arial, sans-serif; fill: #2d3748; }}
  .small {{ font: 12px Arial, sans-serif; fill: #4a5568; }}
</style>
<rect width="100%" height="100%" fill="#f7fafc"/>
{_text(24, 38, "title", title)}
{_text(24, 62, "subtitle", subtitle)}'''


def _rect(x: int, y: int, width: int, height: int, class_name: str = "", **attrs: str) -> str:
    attr_text = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    class_text = f' class="{class_name}"' if class_name else ""
    base = f'<rect x="{x}" y="{y}" width="{width}" height="{height}"'
    return f'{base} rx="4"{class_text} {attr_text}/>'


def _text(x: int, y: int, class_name: str, value: object, anchor: str = "") -> str:
    anchor_text = f' text-anchor="{anchor}"' if anchor else ""
    content = html.escape(str(value))
    return f'<text x="{x}" y="{y}" class="{class_name}"{anchor_text}>{content}</text>'


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


if __name__ == "__main__":
    main()
