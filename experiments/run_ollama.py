"""Run Ollama/GGUF benchmark experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airllm_benchmark.config import GGUFRunConfig, load_gguf_config
from airllm_benchmark.runners.ollama import (
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_URL,
    run_ollama_gguf,
)


def main() -> None:
    args = _parse_args()
    config = _resolve_config(args)

    if config.backend != "ollama":
        raise SystemExit(f"Unsupported GGUF backend: {config.backend}")

    if args.dry_run:
        print(json.dumps(_config_as_json(config, args.api_url, args.stream), indent=2))
        print("\nDry run only; no Ollama request was sent and no output file was written.")
        return

    result = run_ollama_gguf(
        config.output_path,
        model_id=config.model_id,
        quantization=config.quantization,
        prompt=config.prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
        api_url=args.api_url,
        stream=args.stream,
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"\nSaved Ollama/GGUF result to {config.output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "experiment.example.json",
        help="Experiment JSON config.",
    )
    parser.add_argument("--model-id", help="Override model ID from config.")
    parser.add_argument("--quantization", help="Override quantization label from config.")
    parser.add_argument("--prompt", help="Override prompt from config.")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation token limit.")
    parser.add_argument("--temperature", type=float, help="Override generation temperature.")
    parser.add_argument("--output", type=Path, help="Override output JSON path.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        help="Terminate the Ollama worker after this many seconds.",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Run in-process without timeout protection.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama generate endpoint.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use Ollama streaming output so TTFT can be measured.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config without running Ollama.",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> GGUFRunConfig:
    config = load_gguf_config(args.config, PROJECT_ROOT)
    return GGUFRunConfig(
        backend=config.backend,
        model_id=args.model_id or config.model_id,
        quantization=args.quantization or config.quantization,
        prompt=args.prompt or config.prompt,
        max_new_tokens=args.max_new_tokens or config.max_new_tokens,
        temperature=config.temperature if args.temperature is None else args.temperature,
        output_path=args.output or config.output_path,
        timeout_seconds=None if args.no_timeout else args.timeout_seconds,
    )


def _config_as_json(config: GGUFRunConfig, api_url: str, stream: bool) -> dict[str, object]:
    return {
        "backend": config.backend,
        "model_id": config.model_id,
        "quantization": config.quantization,
        "prompt": config.prompt,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "output_path": str(config.output_path),
        "timeout_seconds": config.timeout_seconds,
        "api_url": api_url,
        "stream": stream,
    }


if __name__ == "__main__":
    main()
