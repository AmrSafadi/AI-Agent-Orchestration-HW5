"""Collect local hardware information for the AirLLM benchmark."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airllm_benchmark.hardware import collect_hardware_report, write_hardware_report


def main() -> None:
    output_path = PROJECT_ROOT / "results" / "hardware.json"
    report = collect_hardware_report(PROJECT_ROOT)
    write_hardware_report(report, output_path)
    print(json.dumps(asdict(report), indent=2))
    print(f"\nSaved hardware report to {output_path}")


if __name__ == "__main__":
    main()
