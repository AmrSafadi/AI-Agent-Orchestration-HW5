"""AirLLM benchmark runner."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from airllm_benchmark.metrics import (
    BenchmarkResult,
    compact_notes,
    error_text,
    new_run_id,
    write_benchmark_result,
)
from airllm_benchmark.runners.airllm_runtime import run_airllm_success_path
from airllm_benchmark.runners.process import run_with_timeout

DEFAULT_AIRLLM_TIMEOUT_SECONDS = 1800


def run_airllm_baseline(
    output_path: Path,
    *,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    layer_shards_saving_path: Path,
    huggingface_cache_dir: Path,
    compression: str | None,
    delete_original: bool,
    timeout_seconds: int | None = DEFAULT_AIRLLM_TIMEOUT_SECONDS,
) -> BenchmarkResult:
    """Run AirLLM with timeout and failure capture."""

    run_id = new_run_id("airllm")
    kwargs = {
        "run_id": run_id,
        "model_id": model_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "layer_shards_saving_path": layer_shards_saving_path,
        "huggingface_cache_dir": huggingface_cache_dir,
        "compression": compression,
        "delete_original": delete_original,
    }
    if timeout_seconds is None:
        result = _run_airllm_with_error_capture(**kwargs)
    else:
        result = run_with_timeout(
            worker=_run_airllm_worker,
            worker_kwargs=kwargs,
            timeout_seconds=timeout_seconds,
            timeout_result=lambda peak: _timeout_result(
                run_id, model_id, prompt, compression, timeout_seconds, peak
            ),
            empty_result=lambda code, peak: _empty_result(
                run_id, model_id, prompt, compression, code, peak, layer_shards_saving_path
            ),
            sample_memory=True,
        )

    write_benchmark_result(result, output_path)
    return result


def _run_airllm_worker(queue: mp.Queue[BenchmarkResult], **kwargs: object) -> None:
    queue.put(_run_airllm_with_error_capture(**kwargs))


def _run_airllm_with_error_capture(**kwargs: object) -> BenchmarkResult:
    try:
        return run_airllm_success_path(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on AirLLM/model behavior
        return _failed_result(
            kwargs["run_id"],
            kwargs["model_id"],
            kwargs["prompt"],
            kwargs["compression"],
            error_text(exc),
            compact_notes(
                [
                    "AirLLM failed before producing a complete benchmark.",
                    f"Layer shards path: {kwargs['layer_shards_saving_path']}",
                    f"Hugging Face cache path: {kwargs['huggingface_cache_dir']}",
                ]
            ),
        )


def _timeout_result(
    run_id: str,
    model_id: str,
    prompt: str,
    compression: str | None,
    timeout_seconds: int,
    peak_ram_mb: float | None,
) -> BenchmarkResult:
    return _failed_result(
        run_id,
        model_id,
        prompt,
        compression,
        f"Timed out after {timeout_seconds} seconds.",
        "The AirLLM worker was terminated after exceeding the configured timeout.",
        status="timeout",
        total_runtime_seconds=float(timeout_seconds),
        peak_ram_mb=peak_ram_mb,
    )


def _empty_result(
    run_id: str,
    model_id: str,
    prompt: str,
    compression: str | None,
    exitcode: int | None,
    peak_ram_mb: float | None,
    layer_shards_saving_path: Path,
) -> BenchmarkResult:
    return _failed_result(
        run_id,
        model_id,
        prompt,
        compression,
        f"Worker exited with code {exitcode} before returning a result.",
        f"Layer shards path: {layer_shards_saving_path}",
        peak_ram_mb=peak_ram_mb,
    )


def _failed_result(
    run_id: object,
    model_id: object,
    prompt: object,
    compression: object,
    error: str,
    notes: str,
    *,
    status: str = "failed",
    total_runtime_seconds: float | None = None,
    peak_ram_mb: float | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        str(run_id), "airllm", str(model_id), compression, status, str(prompt), None, None, None,
        None, None, total_runtime_seconds, peak_ram_mb, None, None, error, notes
    )
