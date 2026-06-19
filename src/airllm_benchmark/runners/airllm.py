"""AirLLM benchmark runner."""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty

from airllm_benchmark.metrics import (
    BenchmarkResult,
    ProcessMemorySampler,
    compact_notes,
    error_text,
    new_run_id,
    safe_round,
    write_benchmark_result,
)

DEFAULT_AIRLLM_TIMEOUT_SECONDS = 1800


def run_airllm_baseline(
    output_path: Path,
    *,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    layer_shards_saving_path: Path,
    compression: str | None,
    delete_original: bool,
    timeout_seconds: int | None = DEFAULT_AIRLLM_TIMEOUT_SECONDS,
) -> BenchmarkResult:
    """Run AirLLM with timeout and failure capture."""

    run_id = new_run_id("airllm")
    if timeout_seconds is None:
        result = _run_airllm_with_error_capture(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            layer_shards_saving_path=layer_shards_saving_path,
            compression=compression,
            delete_original=delete_original,
        )
    else:
        result = _run_airllm_with_timeout(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            layer_shards_saving_path=layer_shards_saving_path,
            compression=compression,
            delete_original=delete_original,
            timeout_seconds=timeout_seconds,
        )

    write_benchmark_result(result, output_path)
    return result


def _run_airllm_with_timeout(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    layer_shards_saving_path: Path,
    compression: str | None,
    delete_original: bool,
    timeout_seconds: int,
) -> BenchmarkResult:
    queue: mp.Queue[BenchmarkResult] = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_run_airllm_worker,
        kwargs={
            "queue": queue,
            "run_id": run_id,
            "model_id": model_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "layer_shards_saving_path": layer_shards_saving_path,
            "compression": compression,
            "delete_original": delete_original,
        },
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        return BenchmarkResult(
            run_id=run_id,
            backend="airllm",
            model=model_id,
            quantization=compression,
            status="timeout",
            prompt=prompt,
            input_tokens=None,
            output_tokens=None,
            ttft_seconds=None,
            tpot_seconds=None,
            tokens_per_second=None,
            total_runtime_seconds=float(timeout_seconds),
            peak_ram_mb=None,
            peak_vram_mb=None,
            output_sample=None,
            error=f"Timed out after {timeout_seconds} seconds.",
            notes=compact_notes(
                [
                    "The AirLLM worker was terminated after exceeding the configured timeout.",
                    f"Layer shards path: {layer_shards_saving_path}",
                ]
            ),
        )

    try:
        return queue.get(timeout=1)
    except Empty:
        return BenchmarkResult(
            run_id=run_id,
            backend="airllm",
            model=model_id,
            quantization=compression,
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
            error=f"Worker exited with code {process.exitcode} before returning a result.",
            notes=f"Layer shards path: {layer_shards_saving_path}",
        )


def _run_airllm_worker(
    *,
    queue: mp.Queue[BenchmarkResult],
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    layer_shards_saving_path: Path,
    compression: str | None,
    delete_original: bool,
) -> None:
    result = _run_airllm_with_error_capture(
        run_id=run_id,
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        layer_shards_saving_path=layer_shards_saving_path,
        compression=compression,
        delete_original=delete_original,
    )
    queue.put(result)


def _run_airllm_with_error_capture(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    layer_shards_saving_path: Path,
    compression: str | None,
    delete_original: bool,
) -> BenchmarkResult:
    try:
        return _run_airllm_success_path(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            layer_shards_saving_path=layer_shards_saving_path,
            compression=compression,
            delete_original=delete_original,
        )
    except Exception as exc:  # pragma: no cover - depends on AirLLM/model behavior
        return BenchmarkResult(
            run_id=run_id,
            backend="airllm",
            model=model_id,
            quantization=compression,
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
                    "AirLLM failed before producing a complete benchmark.",
                    f"Layer shards path: {layer_shards_saving_path}",
                ]
            ),
        )


def _run_airllm_success_path(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    layer_shards_saving_path: Path,
    compression: str | None,
    delete_original: bool,
) -> BenchmarkResult:
    import torch
    from airllm import AutoModel

    layer_shards_saving_path.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with ProcessMemorySampler() as memory:
        model = AutoModel.from_pretrained(
            model_id,
            device="cpu",
            dtype=torch.float32,
            layer_shards_saving_path=str(layer_shards_saving_path),
            compression=compression,
            delete_original=delete_original,
            prefetching=False,
        )
        tokenizer = model.tokenizer
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
            output_ids = model.generate(inputs["input_ids"], **generation_kwargs)
        generation_seconds = time.perf_counter() - generation_start
        total_seconds = time.perf_counter() - start

    generated_ids = output_ids[0][input_tokens:]
    output_tokens = int(generated_ids.shape[-1])
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tokens_per_second = output_tokens / generation_seconds if generation_seconds > 0 else None
    tpot_seconds = generation_seconds / output_tokens if output_tokens else None

    return BenchmarkResult(
        run_id=run_id,
        backend="airllm",
        model=model_id,
        quantization=compression,
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
        notes=compact_notes(
            [
                "TTFT is null because this non-streaming AirLLM runner measures total generation only.",
                f"Layer shards path: {layer_shards_saving_path}",
                f"Compression: {compression or 'none'}",
            ]
        ),
    )

