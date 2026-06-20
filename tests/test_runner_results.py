from airllm_benchmark.runners import airllm, ollama
from airllm_benchmark.runners.baseline_results import (
    download_note,
    empty_result,
    failed_result,
    timeout_result,
)


def test_baseline_result_builders_include_status_and_notes() -> None:
    failed = failed_result(
        run_id="run",
        model_id="model",
        prompt="prompt",
        local_files_only=True,
        error=RuntimeError("boom"),
    )
    timed_out = timeout_result("run", "model", "prompt", 7, 42.0, False)
    empty = empty_result("run", "model", "prompt", 9, None, True)

    assert failed.status == "failed"
    assert failed.error == "RuntimeError: boom"
    assert "local_files_only=True" in failed.notes
    assert timed_out.status == "timeout"
    assert timed_out.total_runtime_seconds == 7.0
    assert timed_out.peak_ram_mb == 42.0
    assert empty.error == "Worker exited with code 9 before returning a result."
    assert download_note(False).startswith("Transformers was allowed")


def test_airllm_and_ollama_failure_rows_preserve_core_fields(tmp_path) -> None:
    airllm_empty = airllm._empty_result(
        "run",
        "model",
        "prompt",
        "4bit",
        2,
        128.0,
        tmp_path / "shards",
    )
    ollama_timeout = ollama._timeout_result("run", "model", "Q4", "prompt", 5)

    assert airllm_empty.backend == "airllm"
    assert airllm_empty.quantization == "4bit"
    assert airllm_empty.peak_ram_mb == 128.0
    assert "shards" in airllm_empty.notes
    assert ollama_timeout.backend == "ollama"
    assert ollama_timeout.status == "timeout"
    assert ollama_timeout.total_runtime_seconds == 5.0
