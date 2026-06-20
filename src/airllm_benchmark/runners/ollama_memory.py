"""Memory sampling for local Ollama runtime processes."""

from __future__ import annotations

import threading
import time

import psutil


class OllamaProcessMemorySampler:
    """Sample combined RSS for local Ollama processes while a request runs."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self._interval_seconds = interval_seconds
        self._peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def __enter__(self) -> OllamaProcessMemorySampler:
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
