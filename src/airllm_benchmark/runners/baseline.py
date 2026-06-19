"""Transformers baseline runners."""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty

from airllm_benchmark.metrics import (
    BenchmarkResult,
    ChildProcessMemorySampler,
    ProcessMemorySampler,
    compact_notes,
    error_text,
    new_run_id,
    safe_round,
    write_benchmark_result,
)

TINY_GPT2_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_PROMPT = "Local LLM benchmarking checks"
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_TIMEOUT_SECONDS = 900


def run_tiny_gpt2_baseline(output_path: Path) -> BenchmarkResult:
    """Run a very small Transformers generation to validate benchmark plumbing."""

    run_id = new_run_id("baseline-tiny-gpt2")
    try:
        result = _run_transformers_success_path(
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
        result = _run_transformers_with_error_capture(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            local_files_only=local_files_only,
        )
    else:
        result = _run_transformers_with_timeout(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            local_files_only=local_files_only,
            timeout_seconds=timeout_seconds,
        )

    write_benchmark_result(result, output_path)
    return result


def _run_transformers_with_timeout(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
    timeout_seconds: int,
) -> BenchmarkResult:
    queue: mp.Queue[BenchmarkResult] = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_run_transformers_worker,
        kwargs={
            "queue": queue,
            "run_id": run_id,
            "model_id": model_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "local_files_only": local_files_only,
        },
    )
    process.start()
    with ChildProcessMemorySampler(process.pid) as memory:
        process.join(timeout_seconds)
        peak_ram_mb = memory.peak_ram_mb

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        peak_ram_mb = memory.peak_ram_mb or peak_ram_mb
        return BenchmarkResult(
            run_id=run_id,
            backend="transformers",
            model=model_id,
            quantization=None,
            status="timeout",
            prompt=prompt,
            input_tokens=None,
            output_tokens=None,
            ttft_seconds=None,
            tpot_seconds=None,
            tokens_per_second=None,
            total_runtime_seconds=float(timeout_seconds),
            peak_ram_mb=peak_ram_mb,
            peak_vram_mb=None,
            output_sample=None,
            error=f"Timed out after {timeout_seconds} seconds.",
            notes=compact_notes(
                [
                    "The baseline worker was terminated after exceeding the configured timeout.",
                    _download_note(local_files_only),
                ]
            ),
        )

    try:
        return queue.get(timeout=1)
    except Empty:
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
            peak_ram_mb=peak_ram_mb,
            peak_vram_mb=None,
            output_sample=None,
            error=f"Worker exited with code {process.exitcode} before returning a result.",
            notes=_download_note(local_files_only),
        )


def _run_transformers_worker(
    *,
    queue: mp.Queue[BenchmarkResult],
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
) -> None:
    result = _run_transformers_with_error_capture(
        run_id=run_id,
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        local_files_only=local_files_only,
    )
    queue.put(result)


def _run_transformers_with_error_capture(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
) -> BenchmarkResult:
    try:
        return _run_transformers_success_path(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            local_files_only=local_files_only,
            notes_prefix=None,
        )
    except Exception as exc:  # pragma: no cover - depends on model/backend behavior
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
            error=error_text(exc),
            notes=compact_notes(
                [
                    "Transformers baseline failed before producing a complete benchmark.",
                    _download_note(local_files_only),
                ]
            ),
        )


def _run_transformers_success_path(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
    notes_prefix: list[str] | None,
) -> BenchmarkResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    with ProcessMemorySampler() as memory:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[-1])

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        generation_start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        generation_seconds = time.perf_counter() - generation_start
        total_seconds = time.perf_counter() - start

    generated_ids = output_ids[0][input_tokens:]
    output_tokens = int(generated_ids.shape[-1])
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tokens_per_second = output_tokens / generation_seconds if generation_seconds > 0 else None
    tpot_seconds = generation_seconds / output_tokens if output_tokens else None

    notes = compact_notes(
        (notes_prefix or [])
        + [
            "TTFT is null because this non-streaming baseline measures total generation only.",
            _download_note(local_files_only),
        ]
    )
    return BenchmarkResult(
        run_id=run_id,
        backend="transformers",
        model=model_id,
        quantization=None,
        status="success",
        prompt=prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_seconds=None,
        tpot_seconds=safe_round(tpot_seconds),
        tokens_per_second=safe_round(tokens_per_second),
        total_runtime_seconds=safe_round(total_seconds),
        peak_ram_mb=memory.peak_ram_mb,
        peak_vram_mb=None,
        output_sample=output_text,
        error=None,
        notes=notes,
    )


def _download_note(local_files_only: bool) -> str:
    if local_files_only:
        return "Transformers was run with local_files_only=True to prevent model downloads."
    return "Transformers was allowed to use the network/cache according to its default behavior."
