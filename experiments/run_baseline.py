"""Run the tiny baseline smoke test."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airllm_benchmark.runners.baseline import run_tiny_gpt2_baseline


def main() -> None:
    output_path = PROJECT_ROOT / "results" / "baseline_tiny_gpt2.json"
    result = run_tiny_gpt2_baseline(output_path)
    print(json.dumps(asdict(result), indent=2))
    print(f"\nSaved tiny baseline result to {output_path}")


if __name__ == "__main__":
    main()

