"""Run Transformers baseline experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DEFAULT_MAIN_PROMPT = "Explain the difference between prefill and decode in local LLM inference."

from airllm_benchmark.config import BaselineRunConfig, load_baseline_config
from airllm_benchmark.runners.baseline import (
    DEFAULT_TIMEOUT_SECONDS,
    run_tiny_gpt2_baseline,
    run_transformers_baseline,
)


def main() -> None:
    args = _parse_args()

    if args.config is None and args.model_id is None:
        output_path = PROJECT_ROOT / "results" / "baseline_tiny_gpt2.json"
        result = run_tiny_gpt2_baseline(output_path)
        print(json.dumps(asdict(result), indent=2))
        print(f"\nSaved tiny baseline result to {output_path}")
        return

    config = _resolve_config(args)
    if args.dry_run:
        print(json.dumps(_config_as_json(config), indent=2))
        print("\nDry run only; no model was loaded and no output file was written.")
        return

    result = run_transformers_baseline(
        config.output_path,
        model_id=config.model_id,
        prompt=config.prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        local_files_only=config.local_files_only,
        timeout_seconds=config.timeout_seconds,
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"\nSaved baseline result to {config.output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Experiment JSON config. Defaults to the tiny GPT-2 smoke test when omitted.",
    )
    parser.add_argument("--model-id", help="Override model ID from config.")
    parser.add_argument("--prompt", help="Override prompt from config.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation token limit.")
    parser.add_argument("--temperature", type=float, help="Override generation temperature.")
    parser.add_argument("--output", type=Path, help="Override output JSON path.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to download missing model files.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Terminate the baseline worker after this many seconds.",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Run in-process without timeout protection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config without loading a model.",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> BaselineRunConfig:
    if args.config is None:
        config = BaselineRunConfig(
            model_id=args.model_id,
            prompt=args.prompt or DEFAULT_MAIN_PROMPT,
            max_new_tokens=args.max_new_tokens or 64,
            temperature=0.0 if args.temperature is None else args.temperature,
            output_path=args.output or PROJECT_ROOT / "results" / "baseline_override.json",
            local_files_only=not args.allow_download,
            timeout_seconds=None if args.no_timeout else args.timeout_seconds,
        )
    else:
        config = load_baseline_config(args.config, PROJECT_ROOT)
        config = BaselineRunConfig(
            model_id=args.model_id or config.model_id,
            prompt=args.prompt or config.prompt,
            max_new_tokens=args.max_new_tokens or config.max_new_tokens,
            temperature=config.temperature if args.temperature is None else args.temperature,
            output_path=args.output or config.output_path,
            local_files_only=not args.allow_download,
            timeout_seconds=None if args.no_timeout else args.timeout_seconds,
        )

    if not config.model_id:
        raise SystemExit("--model-id is required when --config is omitted.")
    return config


def _config_as_json(config: BaselineRunConfig) -> dict[str, object]:
    return {
        "model_id": config.model_id,
        "prompt": config.prompt,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "output_path": str(config.output_path),
        "local_files_only": config.local_files_only,
        "timeout_seconds": config.timeout_seconds,
    }


if __name__ == "__main__":
    main()
