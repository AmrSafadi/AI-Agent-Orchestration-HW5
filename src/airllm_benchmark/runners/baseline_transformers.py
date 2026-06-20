"""Transformers execution paths for baseline benchmarks."""

from __future__ import annotations

import multiprocessing as mp
import time

from airllm_benchmark.metrics import (
    BenchmarkResult,
    ProcessMemorySampler,
    compact_notes,
    safe_round,
)
from airllm_benchmark.runners.baseline_results import download_note, failed_result


def run_transformers_worker(
    *,
    queue: mp.Queue[BenchmarkResult],
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
) -> None:
    result = run_transformers_with_error_capture(
        run_id=run_id,
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        local_files_only=local_files_only,
    )
    queue.put(result)


def run_transformers_with_error_capture(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
) -> BenchmarkResult:
    try:
        return run_transformers_success_path(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            local_files_only=local_files_only,
            notes_prefix=None,
        )
    except Exception as exc:  # pragma: no cover - depends on model/backend behavior
        return failed_result(
            run_id=run_id,
            model_id=model_id,
            prompt=prompt,
            local_files_only=local_files_only,
            error=exc,
        )


def run_transformers_success_path(
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
        generation_kwargs = _generation_kwargs(max_new_tokens, temperature, tokenizer.eos_token_id)

        generation_start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        generation_seconds = time.perf_counter() - generation_start
        total_seconds = time.perf_counter() - start

    generated_ids = output_ids[0][input_tokens:]
    output_tokens = int(generated_ids.shape[-1])
    tokens_per_second = output_tokens / generation_seconds if generation_seconds > 0 else None
    tpot_seconds = generation_seconds / output_tokens if output_tokens else None
    notes = compact_notes(
        (notes_prefix or [])
        + [
            "TTFT is null because this non-streaming baseline measures total generation only.",
            download_note(local_files_only),
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
        output_sample=tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        error=None,
        notes=notes,
    )


def _generation_kwargs(
    max_new_tokens: int, temperature: float, eos_token_id: int
) -> dict[str, object]:
    kwargs: dict[str, object] = {"max_new_tokens": max_new_tokens, "pad_token_id": eos_token_id}
    kwargs["do_sample"] = temperature > 0
    if temperature > 0:
        kwargs["temperature"] = temperature
    return kwargs
