"""Shared child-process timeout orchestration for benchmark runners."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from queue import Empty

from airllm_benchmark.metrics import BenchmarkResult, ChildProcessMemorySampler


def run_with_timeout(
    *,
    worker: Callable[..., None],
    worker_kwargs: dict[str, object],
    timeout_seconds: int,
    timeout_result: Callable[[float | None], BenchmarkResult],
    empty_result: Callable[[int | None, float | None], BenchmarkResult],
    sample_memory: bool,
) -> BenchmarkResult:
    """Run a benchmark worker in a child process and convert exits to result rows."""

    queue: mp.Queue[BenchmarkResult] = mp.Queue(maxsize=1)
    process = mp.Process(target=worker, kwargs={"queue": queue, **worker_kwargs})
    process.start()

    peak_ram_mb = None
    if sample_memory:
        with ChildProcessMemorySampler(process.pid) as memory:
            process.join(timeout_seconds)
            peak_ram_mb = memory.peak_ram_mb
    else:
        process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        return timeout_result(peak_ram_mb)

    try:
        return queue.get(timeout=1)
    except Empty:
        return empty_result(process.exitcode, peak_ram_mb)
