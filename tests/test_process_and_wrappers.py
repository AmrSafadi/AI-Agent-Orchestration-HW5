import time
from pathlib import Path

from airllm_benchmark.metrics import BenchmarkResult
from airllm_benchmark.runners import airllm, baseline, ollama
from airllm_benchmark.runners.process import run_with_timeout


def sample_result(status: str = "success") -> BenchmarkResult:
    return BenchmarkResult(
        "run", "backend", "model", None, status, "prompt", None, None, None,
        None, None, None, None, None, None, None, None
    )


def queue_worker(queue, *, status: str = "success") -> None:
    queue.put(sample_result(status))


def empty_worker(queue) -> None:
    return None


def slow_worker(queue) -> None:
    time.sleep(2)


def test_run_with_timeout_success_empty_and_timeout() -> None:
    success = run_with_timeout(
        worker=queue_worker,
        worker_kwargs={"status": "ok"},
        timeout_seconds=5,
        timeout_result=lambda _peak: sample_result("timeout"),
        empty_result=lambda _code, _peak: sample_result("empty"),
        sample_memory=False,
    )
    empty = run_with_timeout(
        worker=empty_worker,
        worker_kwargs={},
        timeout_seconds=5,
        timeout_result=lambda _peak: sample_result("timeout"),
        empty_result=lambda _code, _peak: sample_result("empty"),
        sample_memory=False,
    )
    timed_out = run_with_timeout(
        worker=slow_worker,
        worker_kwargs={},
        timeout_seconds=0,
        timeout_result=lambda _peak: sample_result("timeout"),
        empty_result=lambda _code, _peak: sample_result("empty"),
        sample_memory=False,
    )

    assert success.status == "ok"
    assert empty.status == "empty"
    assert timed_out.status == "timeout"


def test_public_runner_wrappers_use_success_and_timeout_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        baseline,
        "run_transformers_success_path",
        lambda **_kwargs: sample_result("tiny"),
    )
    assert baseline.run_tiny_gpt2_baseline(tmp_path / "tiny.json").status == "tiny"

    monkeypatch.setattr(
        baseline,
        "run_transformers_with_error_capture",
        lambda **_kwargs: sample_result("baseline"),
    )
    assert baseline.run_transformers_baseline(
        tmp_path / "baseline.json",
        model_id="model",
        prompt="prompt",
        max_new_tokens=1,
        temperature=0.0,
        timeout_seconds=None,
    ).status == "baseline"

    monkeypatch.setattr(airllm, "run_airllm_success_path", lambda **_kwargs: sample_result("air"))
    assert airllm.run_airllm_baseline(
        tmp_path / "air.json",
        model_id="model",
        prompt="prompt",
        max_new_tokens=1,
        temperature=0.0,
        layer_shards_saving_path=tmp_path / "shards",
        huggingface_cache_dir=tmp_path / "hf",
        compression=None,
        delete_original=False,
        timeout_seconds=None,
    ).status == "air"

    monkeypatch.setattr(
        ollama,
        "run_ollama_success_path",
        lambda **_kwargs: sample_result("ollama"),
    )
    assert ollama.run_ollama_gguf(
        tmp_path / "ollama.json",
        model_id="model",
        quantization="Q4",
        prompt="prompt",
        max_new_tokens=1,
        temperature=0.0,
        timeout_seconds=None,
    ).status == "ollama"


def test_runner_wrapper_failure_capture(tmp_path, monkeypatch) -> None:
    def fail(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(airllm, "run_airllm_success_path", fail)
    air = airllm.run_airllm_baseline(
        tmp_path / "air.json",
        model_id="model",
        prompt="prompt",
        max_new_tokens=1,
        temperature=0.0,
        layer_shards_saving_path=Path("shards"),
        huggingface_cache_dir=Path("hf"),
        compression=None,
        delete_original=False,
        timeout_seconds=None,
    )
    monkeypatch.setattr(ollama, "run_ollama_success_path", fail)
    ol = ollama.run_ollama_gguf(
        tmp_path / "ollama.json",
        model_id="model",
        quantization="Q4",
        prompt="prompt",
        max_new_tokens=1,
        temperature=0.0,
        timeout_seconds=None,
    )

    assert air.error == "RuntimeError: boom"
    assert ol.error == "RuntimeError: boom"
