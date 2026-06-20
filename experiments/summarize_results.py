"""Print a Markdown summary table from saved benchmark JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = [
    PROJECT_ROOT / "results" / "baseline_tiny_gpt2.json",
    PROJECT_ROOT / "results" / "baseline_qwen_qwen2_5_3b_instruct.json",
    PROJECT_ROOT / "results" / "airllm_qwen_qwen2_5_3b_instruct.json",
    PROJECT_ROOT / "results" / "airllm_phi3_mini_instruct.json",
    PROJECT_ROOT / "results" / "gguf_qwen2_5_3b_instruct_q4_k_m.json",
]


def main() -> None:
    """Load existing benchmark outputs and print a compact Markdown table."""

    rows = [_load_result(path) for path in RESULTS]
    print(
        "| Run | Backend | Model | Quantization | Status | Runtime (s) | "
        "Tokens/s | Peak RAM (MB) | Evidence |"
    )
    print("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |")
    for path, row in zip(RESULTS, rows, strict=True):
        print(
            "| "
            + " | ".join(
                [
                    _run_label(path),
                    _cell(row.get("backend")),
                    _cell(row.get("model")),
                    _cell(row.get("quantization")),
                    _cell(row.get("status")),
                    _number(row.get("total_runtime_seconds")),
                    _number(row.get("tokens_per_second")),
                    _number(row.get("peak_ram_mb")),
                    _evidence(row),
                ]
            )
            + " |"
        )


def _load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_label(path: Path) -> str:
    labels = {
        "baseline_tiny_gpt2.json": "Tiny Transformers smoke test",
        "baseline_qwen_qwen2_5_3b_instruct.json": "Direct Qwen Transformers",
        "airllm_qwen_qwen2_5_3b_instruct.json": "AirLLM Qwen",
        "airllm_phi3_mini_instruct.json": "AirLLM Phi-3 backup",
        "gguf_qwen2_5_3b_instruct_q4_k_m.json": "Ollama GGUF Q4",
    }
    return labels[path.name]


def _cell(value: object) -> str:
    if value is None:
        return "n/a"
    return str(value).replace("|", "\\|")


def _number(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _evidence(row: dict[str, Any]) -> str:
    if row.get("status") == "success":
        output_tokens = row.get("output_tokens")
        if output_tokens is not None:
            return f"Completed {output_tokens} output tokens"
        return "Completed generation"
    error = row.get("error")
    if not error:
        return "No completed generation"
    return str(error).split(". ")[0].replace("|", "\\|")


if __name__ == "__main__":
    main()
