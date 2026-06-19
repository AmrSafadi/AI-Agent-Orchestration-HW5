"""Tiny Transformers baseline runner for pipeline smoke testing."""

from __future__ import annotations

import time
from pathlib import Path

from airllm_benchmark.metrics import (
    BenchmarkResult,
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


def run_tiny_gpt2_baseline(output_path: Path) -> BenchmarkResult:
    """Run a very small Transformers generation to validate benchmark plumbing."""

    run_id = new_run_id("baseline-tiny-gpt2")
    try:
        result = _run_success_path(run_id)
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


def _run_success_path(run_id: str) -> BenchmarkResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    with ProcessMemorySampler() as memory:
        tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_GPT2_MODEL)
        model.eval()

        inputs = tokenizer(DEFAULT_PROMPT, return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[-1])

        generation_start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_seconds = time.perf_counter() - generation_start
        total_seconds = time.perf_counter() - start

    generated_ids = output_ids[0][input_tokens:]
    output_tokens = int(generated_ids.shape[-1])
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tokens_per_second = output_tokens / generation_seconds if generation_seconds > 0 else None
    tpot_seconds = generation_seconds / output_tokens if output_tokens else None

    notes = compact_notes(
        [
            "Tiny GPT-2 validates dependency, loading, generation, and JSON output plumbing.",
            "This is not the final assignment model.",
            "TTFT is null because this non-streaming smoke test measures total generation only.",
        ]
    )
    return BenchmarkResult(
        run_id=run_id,
        backend="transformers",
        model=TINY_GPT2_MODEL,
        quantization=None,
        status="success",
        prompt=DEFAULT_PROMPT,
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

