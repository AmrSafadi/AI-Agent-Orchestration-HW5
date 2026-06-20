"""Ollama HTTP API execution for GGUF benchmarks."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from airllm_benchmark.metrics import BenchmarkResult, compact_notes, safe_round
from airllm_benchmark.runners.ollama_memory import OllamaProcessMemorySampler


def run_ollama_success_path(
    *,
    run_id: str,
    model_id: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    api_url: str,
    stream: bool,
) -> BenchmarkResult:
    request = _request(api_url, model_id, prompt, max_new_tokens, temperature, stream)
    start = time.perf_counter()
    with OllamaProcessMemorySampler() as memory:
        try:
            with urllib.request.urlopen(request, timeout=None) as response:
                if stream:
                    raw, response_text, ttft_seconds = read_streaming_response(response, start)
                else:
                    raw_response = response.read()
                    raw = json.loads(raw_response.decode("utf-8"))
                    response_text = str(raw.get("response", "")).strip()
                    ttft_seconds = None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama API request failed: {exc}") from exc
    wall_seconds = time.perf_counter() - start

    output_tokens = _optional_int(raw.get("eval_count"))
    eval_duration = _duration_seconds(raw.get("eval_duration"))
    tokens_per_second = output_tokens / eval_duration if eval_duration and output_tokens else None
    tpot_seconds = eval_duration / output_tokens if eval_duration and output_tokens else None
    return BenchmarkResult(
        run_id=run_id,
        backend="ollama",
        model=model_id,
        quantization=quantization,
        status="success",
        prompt=prompt,
        input_tokens=_optional_int(raw.get("prompt_eval_count")),
        output_tokens=output_tokens,
        ttft_seconds=safe_round(ttft_seconds),
        tpot_seconds=safe_round(tpot_seconds),
        tokens_per_second=safe_round(tokens_per_second),
        total_runtime_seconds=safe_round(
            _duration_seconds(raw.get("total_duration")) or wall_seconds
        ),
        peak_ram_mb=memory.peak_ram_mb,
        peak_vram_mb=None,
        output_sample=response_text,
        error=None,
        notes=_notes(stream, wall_seconds, raw),
    )


def read_streaming_response(
    response: object, start: float
) -> tuple[dict[str, Any], str, float | None]:
    chunks = []
    final: dict[str, Any] | None = None
    ttft_seconds = None
    for line in response:
        if not line:
            continue
        item = json.loads(line.decode("utf-8"))
        text = str(item.get("response", ""))
        if text and ttft_seconds is None:
            ttft_seconds = time.perf_counter() - start
        chunks.append(text)
        final = item
    if final is None:
        raise RuntimeError("Ollama streaming response ended without a final JSON object.")
    return final, "".join(chunks).strip(), ttft_seconds


def _request(
    api_url: str, model_id: str, prompt: str, max_new_tokens: int, temperature: float, stream: bool
) -> urllib.request.Request:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": stream,
        "options": {"num_predict": max_new_tokens, "temperature": temperature},
    }
    return urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _notes(stream: bool, wall_seconds: float, raw: dict[str, Any]) -> str:
    return compact_notes(
        [
            "Ollama/GGUF was run through the local /api/generate endpoint "
            f"with stream={str(stream).lower()}.",
            (
                "TTFT was measured from the first streamed response chunk."
                if stream
                else "TTFT is null because the non-streaming API response "
                "does not expose first-token timing."
            ),
            f"Measured wall-clock request time: {safe_round(wall_seconds)} seconds.",
            "Prompt eval duration: "
            f"{safe_round(_duration_seconds(raw.get('prompt_eval_duration')))} seconds.",
            f"Load duration: {safe_round(_duration_seconds(raw.get('load_duration')))} seconds.",
        ]
    )


def _duration_seconds(value: Any) -> float | None:
    numeric = _optional_int(value)
    if numeric is None:
        return None
    return numeric / 1_000_000_000


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
