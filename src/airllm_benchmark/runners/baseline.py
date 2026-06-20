"""Transformers baseline runners."""

from __future__ import annotations

from pathlib import Path

from airllm_benchmark.metrics import (
    BenchmarkResult,
    error_text,
    new_run_id,
    write_benchmark_result,
)
from airllm_benchmark.runners.baseline_results import empty_result, timeout_result
from airllm_benchmark.runners.baseline_transformers import (
    run_transformers_success_path,
    run_transformers_with_error_capture,
    run_transformers_worker,
)
from airllm_benchmark.runners.process import run_with_timeout

TINY_GPT2_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_PROMPT = "Local LLM benchmarking checks"
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_TIMEOUT_SECONDS = 900


def run_tiny_gpt2_baseline(output_path: Path) -> BenchmarkResult:
    """Run a very small Transformers generation to validate benchmark plumbing."""

    run_id = new_run_id("baseline-tiny-gpt2")
    try:
        result = run_transformers_success_path(
            run_id=run_id,
            model_id=TINY_GPT2_MODEL,
            prompt=DEFAULT_PROMPT,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=0.0,
            local_files_only=False,
            notes_prefix=[
                "Tiny GPT-2 validates dependency, loading, generation, and JSON output plumbing.",
                "This is not the final assignment model.",
            ],
        )
    except Exception as exc:  # pragma: no cover - depends on external model access
        result = BenchmarkResult(
            run_id=run_id,
            backend="transformers",
            model=TINY_GPT2_MODEL,
            quantization=None,
            status="failed",
            prompt=DEFAULT_PROMPT,
            input_tokens=None,
            output_tokens=None,
            ttft_seconds=None,
            tpot_seconds=None,
            tokens_per_second=None,
            total_runtime_seconds=None,
            peak_ram_mb=None,
            peak_vram_mb=None,
            output_sample=None,
            error=error_text(exc),
            notes="Tiny smoke-test failed before producing a complete benchmark.",
        )

    write_benchmark_result(result, output_path)
    return result


def run_transformers_baseline(
    output_path: Path,
    *,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool = True,
    timeout_seconds: int | None = DEFAULT_TIMEOUT_SECONDS,
) -> BenchmarkResult:
    """Run a configurable Transformers baseline with failure and timeout capture."""

    run_id = new_run_id("baseline-transformers")
    if timeout_seconds is None:
        result = run_transformers_with_error_capture(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            local_files_only=local_files_only,
        )
    else:
        result = run_with_timeout(
            worker=run_transformers_worker,
            worker_kwargs={
                "run_id": run_id,
                "model_id": model_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "local_files_only": local_files_only,
            },
            timeout_seconds=timeout_seconds,
            timeout_result=lambda peak: timeout_result(
                run_id, model_id, prompt, timeout_seconds, peak, local_files_only
            ),
            empty_result=lambda code, peak: empty_result(
                run_id, model_id, prompt, code, peak, local_files_only
            ),
            sample_memory=True,
        )

    write_benchmark_result(result, output_path)
    return result
