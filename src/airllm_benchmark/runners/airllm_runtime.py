"""AirLLM in-process execution path and cache configuration."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from airllm_benchmark.metrics import (
    BenchmarkResult,
    ProcessMemorySampler,
    compact_notes,
    safe_round,
)


def run_airllm_success_path(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    layer_shards_saving_path: Path,
    huggingface_cache_dir: Path,
    compression: str | None,
    delete_original: bool,
) -> BenchmarkResult:
    import torch
    from airllm import AutoModel

    configure_huggingface_cache(huggingface_cache_dir)
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
        generation_kwargs = _generation_kwargs(max_new_tokens, temperature, tokenizer.eos_token_id)

        generation_start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(inputs["input_ids"], **generation_kwargs)
        generation_seconds = time.perf_counter() - generation_start
        total_seconds = time.perf_counter() - start

    generated_ids = output_ids[0][input_tokens:]
    output_tokens = int(generated_ids.shape[-1])
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
        output_sample=tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        error=None,
        notes=compact_notes(
            [
                "TTFT is null because this non-streaming AirLLM runner "
                "measures total generation only.",
                f"Layer shards path: {layer_shards_saving_path}",
                f"Hugging Face cache path: {huggingface_cache_dir}",
                f"Compression: {compression or 'none'}",
            ]
        ),
    )


def configure_huggingface_cache(cache_dir: Path) -> None:
    """Use a project-local Hugging Face cache and avoid Windows symlink privileges."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache_dir = cache_dir / "hub"
    hub_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache_dir)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    import huggingface_hub.constants as hf_constants
    import huggingface_hub.file_download as hf_file_download

    hf_constants.HF_HOME = str(cache_dir)
    hf_constants.HF_HUB_CACHE = str(hub_cache_dir)

    if sys.platform == "win32":
        hf_file_download.are_symlinks_supported = lambda cache_dir=None: False


def _generation_kwargs(
    max_new_tokens: int, temperature: float, eos_token_id: int
) -> dict[str, object]:
    kwargs: dict[str, object] = {"max_new_tokens": max_new_tokens, "pad_token_id": eos_token_id}
    kwargs["do_sample"] = temperature > 0
    if temperature > 0:
        kwargs["temperature"] = temperature
    return kwargs
