"""Shared result builders for Transformers baseline failures."""

from __future__ import annotations

from airllm_benchmark.metrics import BenchmarkResult, compact_notes, error_text


def download_note(local_files_only: bool) -> str:
    if local_files_only:
        return "Transformers was run with local_files_only=True to prevent model downloads."
    return "Transformers was allowed to use the network/cache according to its default behavior."


def failed_result(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    local_files_only: bool,
    error: BaseException,
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id=run_id,
        backend="transformers",
        model=model_id,
        quantization=None,
        status="failed",
        prompt=prompt,
        input_tokens=None,
        output_tokens=None,
        ttft_seconds=None,
        tpot_seconds=None,
        tokens_per_second=None,
        total_runtime_seconds=None,
        peak_ram_mb=None,
        peak_vram_mb=None,
        output_sample=None,
        error=error_text(error),
        notes=compact_notes(
            [
                "Transformers baseline failed before producing a complete benchmark.",
                download_note(local_files_only),
            ]
        ),
    )


def timeout_result(
    run_id: str,
    model_id: str,
    prompt: str,
    timeout_seconds: int,
    peak_ram_mb: float | None,
    local_files_only: bool,
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id,
        "transformers",
        model_id,
        None,
        "timeout",
        prompt,
        None,
        None,
        None,
        None,
        None,
        float(timeout_seconds),
        peak_ram_mb,
        None,
        None,
        f"Timed out after {timeout_seconds} seconds.",
        compact_notes(
            [
                "The baseline worker was terminated after exceeding the configured timeout.",
                download_note(local_files_only),
            ]
        ),
    )


def empty_result(
    run_id: str,
    model_id: str,
    prompt: str,
    exitcode: int | None,
    peak_ram_mb: float | None,
    local_files_only: bool,
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id,
        "transformers",
        model_id,
        None,
        "failed",
        prompt,
        None,
        None,
        None,
        None,
        None,
        None,
        peak_ram_mb,
        None,
        None,
        f"Worker exited with code {exitcode} before returning a result.",
        download_note(local_files_only),
    )
