"""Ollama/GGUF benchmark runner."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from airllm_benchmark.metrics import BenchmarkResult, error_text, new_run_id, write_benchmark_result
from airllm_benchmark.runners.ollama_api import run_ollama_success_path
from airllm_benchmark.runners.process import run_with_timeout

DEFAULT_OLLAMA_TIMEOUT_SECONDS = 900
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def run_ollama_gguf(
    output_path: Path,
    *,
    model_id: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: int | None = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    api_url: str = DEFAULT_OLLAMA_URL,
    stream: bool = False,
) -> BenchmarkResult:
    """Run an Ollama-hosted GGUF model with timeout and failure capture."""

    run_id = new_run_id("ollama-gguf")
    kwargs = {
        "run_id": run_id,
        "model_id": model_id,
        "quantization": quantization,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "api_url": api_url,
        "stream": stream,
    }
    if timeout_seconds is None:
        result = _run_ollama_with_error_capture(**kwargs)
    else:
        result = run_with_timeout(
            worker=_run_ollama_worker,
            worker_kwargs=kwargs,
            timeout_seconds=timeout_seconds,
            timeout_result=lambda _peak: _timeout_result(
                run_id, model_id, quantization, prompt, timeout_seconds
            ),
            empty_result=lambda code, _peak: _empty_result(
                run_id, model_id, quantization, prompt, code
            ),
            sample_memory=False,
        )

    write_benchmark_result(result, output_path)
    return result


def _run_ollama_worker(queue: mp.Queue[BenchmarkResult], **kwargs: object) -> None:
    queue.put(_run_ollama_with_error_capture(**kwargs))


def _run_ollama_with_error_capture(**kwargs: object) -> BenchmarkResult:
    try:
        return run_ollama_success_path(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on local Ollama service
        return _failed_result(
            kwargs["run_id"],
            kwargs["model_id"],
            kwargs["quantization"],
            kwargs["prompt"],
            error_text(exc),
            "Ollama/GGUF run failed before producing a complete benchmark.",
        )


def _timeout_result(
    run_id: str, model_id: str, quantization: str, prompt: str, timeout_seconds: int
) -> BenchmarkResult:
    return _failed_result(
        run_id,
        model_id,
        quantization,
        prompt,
        f"Timed out after {timeout_seconds} seconds.",
        "The Ollama worker was terminated after exceeding the configured timeout.",
        status="timeout",
        total_runtime_seconds=float(timeout_seconds),
    )


def _empty_result(
    run_id: str, model_id: str, quantization: str, prompt: str, exitcode: int | None
) -> BenchmarkResult:
    return _failed_result(
        run_id,
        model_id,
        quantization,
        prompt,
        f"Worker exited with code {exitcode} before returning a result.",
        "Ollama API call did not return a benchmark result.",
    )


def _failed_result(
    run_id: object,
    model_id: object,
    quantization: object,
    prompt: object,
    error: str,
    notes: str,
    *,
    status: str = "failed",
    total_runtime_seconds: float | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        str(run_id), "ollama", str(model_id), str(quantization), status, str(prompt), None,
        None, None, None, None, total_runtime_seconds, None, None, None, error, notes
    )
