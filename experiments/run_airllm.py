"""Run AirLLM benchmark experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airllm_benchmark.config import AirLLMRunConfig, load_airllm_config
from airllm_benchmark.runners.airllm import (
    DEFAULT_AIRLLM_TIMEOUT_SECONDS,
    run_airllm_baseline,
)


def main() -> None:
    args = _parse_args()
    config = _resolve_config(args)

    if args.dry_run:
        print(json.dumps(_config_as_json(config), indent=2))
        print("\nDry run only; no AirLLM model was loaded and no output file was written.")
        return

    result = run_airllm_baseline(
        config.output_path,
        model_id=config.model_id,
        prompt=config.prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        layer_shards_saving_path=config.layer_shards_saving_path,
        compression=config.compression,
        delete_original=config.delete_original,
        timeout_seconds=config.timeout_seconds,
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"\nSaved AirLLM result to {config.output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "experiment.example.json",
        help="Experiment JSON config.",
    )
    parser.add_argument("--model-id", help="Override model ID from config.")
    parser.add_argument("--prompt", help="Override prompt from config.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation token limit.")
    parser.add_argument("--temperature", type=float, help="Override generation temperature.")
    parser.add_argument("--output", type=Path, help="Override output JSON path.")
    parser.add_argument("--layer-shards-saving-path", type=Path, help="Override AirLLM shard path.")
    parser.add_argument("--compression", help="AirLLM compression setting, such as 4bit or 8bit.")
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Ask AirLLM to delete original downloaded model files after sharding.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Terminate the AirLLM worker after this many seconds.",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Run in-process without timeout protection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config without loading an AirLLM model.",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> AirLLMRunConfig:
    config = load_airllm_config(args.config, PROJECT_ROOT)
    timeout_seconds = (
        None
        if args.no_timeout
        else args.timeout_seconds
        if args.timeout_seconds is not None
        else config.timeout_seconds or DEFAULT_AIRLLM_TIMEOUT_SECONDS
    )

    return AirLLMRunConfig(
        model_id=args.model_id or config.model_id,
        prompt=args.prompt or config.prompt,
        max_new_tokens=args.max_new_tokens or config.max_new_tokens,
        temperature=config.temperature if args.temperature is None else args.temperature,
        output_path=args.output or config.output_path,
        layer_shards_saving_path=args.layer_shards_saving_path or config.layer_shards_saving_path,
        compression=args.compression if args.compression is not None else config.compression,
        delete_original=args.delete_original or config.delete_original,
        timeout_seconds=timeout_seconds,
    )


def _config_as_json(config: AirLLMRunConfig) -> dict[str, object]:
    return {
        "model_id": config.model_id,
        "prompt": config.prompt,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "output_path": str(config.output_path),
        "layer_shards_saving_path": str(config.layer_shards_saving_path),
        "compression": config.compression,
        "delete_original": config.delete_original,
        "timeout_seconds": config.timeout_seconds,
    }


if __name__ == "__main__":
    main()
