"""Ollama/GGUF benchmark runner."""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from queue import Empty
from typing import Any

import psutil

from airllm_benchmark.metrics import (
    BenchmarkResult,
    compact_notes,
    error_text,
    new_run_id,
    safe_round,
    write_benchmark_result,
)

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
    if timeout_seconds is None:
        result = _run_ollama_with_error_capture(
            run_id=run_id,
            model_id=model_id,
            quantization=quantization,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            api_url=api_url,
            stream=stream,
        )
    else:
        result = _run_ollama_with_timeout(
            run_id=run_id,
            model_id=model_id,
            quantization=quantization,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            api_url=api_url,
            stream=stream,
            timeout_seconds=timeout_seconds,
        )

    write_benchmark_result(result, output_path)
    return result


def _run_ollama_with_timeout(
    *,
    run_id: str,
    model_id: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    api_url: str,
    stream: bool,
    timeout_seconds: int,
) -> BenchmarkResult:
    queue: mp.Queue[BenchmarkResult] = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_run_ollama_worker,
        kwargs={
            "queue": queue,
            "run_id": run_id,
            "model_id": model_id,
            "quantization": quantization,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "api_url": api_url,
            "stream": stream,
        },
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        return BenchmarkResult(
            run_id=run_id,
            backend="ollama",
            model=model_id,
            quantization=quantization,
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
            notes="The Ollama worker was terminated after exceeding the configured timeout.",
        )

    try:
        return queue.get(timeout=1)
    except Empty:
        return BenchmarkResult(
            run_id=run_id,
            backend="ollama",
            model=model_id,
            quantization=quantization,
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
            notes="Ollama API call did not return a benchmark result.",
        )


def _run_ollama_worker(
    *,
    queue: mp.Queue[BenchmarkResult],
    run_id: str,
    model_id: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    api_url: str,
    stream: bool,
) -> None:
    result = _run_ollama_with_error_capture(
        run_id=run_id,
        model_id=model_id,
        quantization=quantization,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        api_url=api_url,
        stream=stream,
    )
    queue.put(result)


def _run_ollama_with_error_capture(
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
    try:
        return _run_ollama_success_path(
            run_id=run_id,
            model_id=model_id,
            quantization=quantization,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            api_url=api_url,
            stream=stream,
        )
    except Exception as exc:  # pragma: no cover - depends on local Ollama service
        return BenchmarkResult(
            run_id=run_id,
            backend="ollama",
            model=model_id,
            quantization=quantization,
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
            notes="Ollama/GGUF run failed before producing a complete benchmark.",
        )


def _run_ollama_success_path(
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
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with OllamaProcessMemorySampler() as memory:
        try:
            with urllib.request.urlopen(request, timeout=None) as response:
                if stream:
                    raw, response_text, ttft_seconds = _read_streaming_response(response, start)
                else:
                    raw_response = response.read()
                    raw = json.loads(raw_response.decode("utf-8"))
                    response_text = str(raw.get("response", "")).strip()
                    ttft_seconds = None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama API request failed: {exc}") from exc
    wall_seconds = time.perf_counter() - start

    input_tokens = _optional_int(raw.get("prompt_eval_count"))
    output_tokens = _optional_int(raw.get("eval_count"))
    eval_duration = _duration_seconds(raw.get("eval_duration"))
    prompt_eval_duration = _duration_seconds(raw.get("prompt_eval_duration"))
    total_duration = _duration_seconds(raw.get("total_duration"))
    load_duration = _duration_seconds(raw.get("load_duration"))

    tpot_seconds = eval_duration / output_tokens if eval_duration and output_tokens else None
    tokens_per_second = output_tokens / eval_duration if eval_duration and output_tokens else None

    return BenchmarkResult(
        run_id=run_id,
        backend="ollama",
        model=model_id,
        quantization=quantization,
        status="success",
        prompt=prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_seconds=safe_round(ttft_seconds),
        tpot_seconds=safe_round(tpot_seconds),
        tokens_per_second=safe_round(tokens_per_second),
        total_runtime_seconds=safe_round(total_duration or wall_seconds),
        peak_ram_mb=memory.peak_ram_mb,
        peak_vram_mb=None,
        output_sample=response_text,
        error=None,
        notes=compact_notes(
            [
                f"Ollama/GGUF was run through the local /api/generate endpoint with stream={str(stream).lower()}.",
                (
                    "TTFT was measured from the first streamed response chunk."
                    if stream
                    else "TTFT is null because the non-streaming API response does not expose first-token timing."
                ),
                f"Measured wall-clock request time: {safe_round(wall_seconds)} seconds.",
                f"Prompt eval duration: {safe_round(prompt_eval_duration)} seconds.",
                f"Load duration: {safe_round(load_duration)} seconds.",
            ]
        ),
    )


def _read_streaming_response(
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


class OllamaProcessMemorySampler:
    """Sample combined RSS for local Ollama processes while a request runs."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self._interval_seconds = interval_seconds
        self._peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def __enter__(self) -> "OllamaProcessMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._record_current_rss()
        self._stop_event.set()
        self._thread.join(timeout=1)
        self._record_current_rss()

    @property
    def peak_ram_mb(self) -> float | None:
        if self._peak_rss_bytes <= 0:
            return None
        return round(self._peak_rss_bytes / (1024**2), 2)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._record_current_rss()
            time.sleep(self._interval_seconds)

    def _record_current_rss(self) -> None:
        rss = 0
        for process in psutil.process_iter(["name"]):
            try:
                name = (process.info.get("name") or "").lower()
                if name.startswith("ollama") or name.startswith("llama-server"):
                    rss += process.memory_info().rss
            except psutil.Error:
                continue
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)


def _duration_seconds(value: Any) -> float | None:
    numeric = _optional_int(value)
    if numeric is None:
        return None
    return numeric / 1_000_000_000


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
