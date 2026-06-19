"""Benchmark result schema and lightweight process memory sampling."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


@dataclass
class BenchmarkResult:
    """Serializable benchmark result matching the project PRD schema."""

    run_id: str
    backend: str
    model: str
    quantization: str | None
    status: str
    prompt: str
    input_tokens: int | None
    output_tokens: int | None
    ttft_seconds: float | None
    tpot_seconds: float | None
    tokens_per_second: float | None
    total_runtime_seconds: float | None
    peak_ram_mb: float | None
    peak_vram_mb: float | None
    output_sample: str | None
    error: str | None
    notes: str | None


class ProcessMemorySampler:
    """Sample current process RSS while a benchmark runs."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._peak_rss_bytes = self._process.memory_info().rss
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def __enter__(self) -> "ProcessMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1)
        self._record_current_rss()

    @property
    def peak_ram_mb(self) -> float:
        return round(self._peak_rss_bytes / (1024**2), 2)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._record_current_rss()
            time.sleep(self._interval_seconds)

    def _record_current_rss(self) -> None:
        rss = self._process.memory_info().rss
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)


def new_run_id(prefix: str) -> str:
    """Create a compact traceable run ID."""

    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def write_benchmark_result(result: BenchmarkResult, output_path: Path) -> None:
    """Write a benchmark result as pretty JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def error_text(exc: BaseException) -> str:
    """Format an exception for result JSON."""

    return f"{type(exc).__name__}: {exc}"


def safe_round(value: float | None, digits: int = 4) -> float | None:
    """Round optional floats for stable result files."""

    if value is None:
        return None
    return round(value, digits)


def compact_notes(parts: list[str | None]) -> str:
    """Join optional note fragments."""

    return " ".join(part for part in parts if part)

