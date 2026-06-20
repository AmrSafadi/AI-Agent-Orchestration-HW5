from types import SimpleNamespace

import pytest

import experiments.run_airllm as run_airllm
import experiments.run_baseline as run_baseline
import experiments.run_ollama as run_ollama
from airllm_benchmark.config import AirLLMRunConfig, BaselineRunConfig, GGUFRunConfig
from airllm_benchmark.metrics import BenchmarkResult


def result(backend: str) -> BenchmarkResult:
    return BenchmarkResult(
        "run", backend, "model", None, "success", "prompt", 1, 1, None, None,
        None, 1.0, None, None, "ok", None, None
    )


def test_baseline_resolve_config_branches(tmp_path, monkeypatch) -> None:
    args = SimpleNamespace(
        config=None,
        model_id="model",
        prompt=None,
        max_new_tokens=None,
        temperature=None,
        output=None,
        allow_download=True,
        no_timeout=True,
        timeout_seconds=5,
    )
    config = run_baseline._resolve_config(args)

    assert config.prompt == run_baseline.DEFAULT_MAIN_PROMPT
    assert config.local_files_only is False
    assert config.timeout_seconds is None

    loaded = BaselineRunConfig("cfg", "p", 2, 0.1, tmp_path / "out.json", True, None)
    monkeypatch.setattr(run_baseline, "load_baseline_config", lambda *_args: loaded)
    args.config = tmp_path / "config.json"
    args.model_id = None
    args.prompt = "override"
    args.max_new_tokens = 3
    args.temperature = 0.2
    args.output = tmp_path / "override.json"
    args.allow_download = False
    args.no_timeout = False

    config = run_baseline._resolve_config(args)
    assert config.model_id == "cfg"
    assert config.prompt == "override"
    assert config.local_files_only is True
    assert run_baseline._config_as_json(config)["max_new_tokens"] == 3

    args.config = None
    args.model_id = None
    with pytest.raises(SystemExit):
        run_baseline._resolve_config(args)


def test_cli_main_dry_runs_and_tiny_branch(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_baseline,
        "_parse_args",
        lambda: SimpleNamespace(config=None, model_id=None),
    )
    monkeypatch.setattr(run_baseline, "run_tiny_gpt2_baseline", lambda path: result("tiny"))
    run_baseline.main()
    assert "Saved tiny baseline result" in capsys.readouterr().out

    config = BaselineRunConfig("model", "prompt", 1, 0.0, tmp_path / "out.json")
    monkeypatch.setattr(
        run_baseline,
        "_parse_args",
        lambda: SimpleNamespace(config=tmp_path / "c.json", model_id="model", dry_run=True),
    )
    monkeypatch.setattr(run_baseline, "_resolve_config", lambda _args: config)
    run_baseline.main()
    assert "Dry run only" in capsys.readouterr().out


def test_airllm_resolve_and_main_dry_run(tmp_path, monkeypatch, capsys) -> None:
    loaded = AirLLMRunConfig(
        "cfg", "p", 2, 0.1, tmp_path / "out.json", tmp_path / "hf",
        tmp_path / "shards", None, False, None
    )
    monkeypatch.setattr(run_airllm, "load_airllm_config", lambda *_args: loaded)
    args = SimpleNamespace(
        config=tmp_path / "config.json",
        model_id="model",
        prompt="prompt",
        max_new_tokens=3,
        temperature=0.2,
        output=tmp_path / "override.json",
        huggingface_cache_dir=None,
        layer_shards_saving_path=None,
        compression="4bit",
        delete_original=True,
        timeout_seconds=None,
        no_timeout=False,
        dry_run=True,
    )

    config = run_airllm._resolve_config(args)
    assert config.timeout_seconds == run_airllm.DEFAULT_AIRLLM_TIMEOUT_SECONDS
    assert config.compression == "4bit"
    assert run_airllm._config_as_json(config)["delete_original"] is True

    monkeypatch.setattr(run_airllm, "_parse_args", lambda: args)
    run_airllm.main()
    assert "Dry run only" in capsys.readouterr().out


def test_ollama_resolve_and_main_paths(tmp_path, monkeypatch, capsys) -> None:
    loaded = GGUFRunConfig("ollama", "cfg", "Q4", "p", 2, 0.1, tmp_path / "out.json")
    monkeypatch.setattr(run_ollama, "load_gguf_config", lambda *_args: loaded)
    args = SimpleNamespace(
        config=tmp_path / "config.json",
        model_id=None,
        quantization=None,
        prompt=None,
        max_new_tokens=None,
        temperature=None,
        output=None,
        timeout_seconds=5,
        no_timeout=True,
        api_url="http://local",
        stream=True,
        dry_run=True,
    )

    config = run_ollama._resolve_config(args)
    assert config.timeout_seconds is None
    assert run_ollama._config_as_json(config, args.api_url, args.stream)["stream"] is True

    monkeypatch.setattr(run_ollama, "_parse_args", lambda: args)
    run_ollama.main()
    assert "Dry run only" in capsys.readouterr().out

    bad_config = GGUFRunConfig("bad", "cfg", "Q4", "p", 2, 0.1, tmp_path / "out.json")
    monkeypatch.setattr(run_ollama, "_resolve_config", lambda _args: bad_config)
    with pytest.raises(SystemExit):
        run_ollama.main()
