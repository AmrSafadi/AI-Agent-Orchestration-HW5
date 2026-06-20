import json
import urllib.error

import pytest

from airllm_benchmark.runners.ollama_api import (
    _duration_seconds,
    _notes,
    _optional_int,
    _request,
    read_streaming_response,
    run_ollama_success_path,
)


class FakeMemorySampler:
    peak_ram_mb = 12.5

    def __enter__(self) -> "FakeMemorySampler":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_ollama_helpers_build_request_and_notes() -> None:
    request = _request("http://local", "model", "prompt", 3, 0.2, True)

    assert request.full_url == "http://local"
    assert request.get_method() == "POST"
    assert _optional_int("4") == 4
    assert _optional_int(None) is None
    assert _duration_seconds(2_000_000_000) == 2.0
    assert "first streamed response chunk" in _notes(True, 1.25, {})
    assert "does not expose first-token timing" in _notes(False, 1.25, {})


def test_read_streaming_response_rejects_empty_stream() -> None:
    with pytest.raises(RuntimeError, match="final JSON object"):
        read_streaming_response([], 0.0)


def test_run_ollama_success_path_non_streaming(monkeypatch) -> None:
    raw = {
        "response": "answer",
        "prompt_eval_count": 5,
        "eval_count": 10,
        "eval_duration": 2_000_000_000,
        "total_duration": 3_000_000_000,
        "prompt_eval_duration": 500_000_000,
        "load_duration": 250_000_000,
    }

    monkeypatch.setattr(
        "airllm_benchmark.runners.ollama_api.OllamaProcessMemorySampler",
        FakeMemorySampler,
    )
    monkeypatch.setattr(
        "airllm_benchmark.runners.ollama_api.urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(raw),
    )

    result = run_ollama_success_path(
        run_id="run",
        model_id="model",
        quantization="Q4",
        prompt="prompt",
        max_new_tokens=10,
        temperature=0.0,
        api_url="http://local",
        stream=False,
    )

    assert result.status == "success"
    assert result.output_sample == "answer"
    assert result.input_tokens == 5
    assert result.output_tokens == 10
    assert result.tokens_per_second == 5.0
    assert result.tpot_seconds == 0.2
    assert result.total_runtime_seconds == 3.0
    assert result.peak_ram_mb == 12.5


def test_run_ollama_success_path_wraps_url_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "airllm_benchmark.runners.ollama_api.OllamaProcessMemorySampler",
        FakeMemorySampler,
    )
    monkeypatch.setattr(
        "airllm_benchmark.runners.ollama_api.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("offline")
        ),
    )

    with pytest.raises(RuntimeError, match="Ollama API request failed"):
        run_ollama_success_path(
            run_id="run",
            model_id="model",
            quantization="Q4",
            prompt="prompt",
            max_new_tokens=10,
            temperature=0.0,
            api_url="http://local",
            stream=False,
        )
