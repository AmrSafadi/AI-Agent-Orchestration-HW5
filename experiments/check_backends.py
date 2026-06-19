"""Check local backend availability without downloading model weights."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airllm_benchmark.backends import collect_backend_check, write_backend_check


def main() -> None:
    output_path = PROJECT_ROOT / "results" / "backend_check.json"
    report = collect_backend_check()
    write_backend_check(report, output_path)
    print(json.dumps(asdict(report), indent=2))
    print(f"\nSaved backend check report to {output_path}")


if __name__ == "__main__":
    main()

