from pathlib import Path

import pytest

from airllm_benchmark.config import (
    _airllm_output_name,
    _baseline_output_name,
    _gguf_output_name,
    _required_dict,
    _required_str,
    _slug,
    load_airllm_config,
    load_baseline_config,
    load_gguf_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "experiment.example.json"


def test_baseline_config_uses_model_slug_for_output() -> None:
    config = load_baseline_config(CONFIG, ROOT)

    assert config.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert config.output_path == ROOT / "results" / "baseline_qwen_qwen2_5_3b_instruct.json"
    assert config.local_files_only is True


def test_airllm_config_uses_project_local_paths() -> None:
    config = load_airllm_config(CONFIG, ROOT)

    assert config.huggingface_cache_dir == ROOT / "model_cache" / "huggingface"
    assert config.layer_shards_saving_path == ROOT / "airllm_cache" / "qwen2_5_3b_instruct"
    assert config.timeout_seconds == 1800


def test_gguf_config_matches_saved_result_name() -> None:
    config = load_gguf_config(CONFIG, ROOT)

    assert config.backend == "ollama"
    assert config.quantization == "Q4_K_M"
    assert config.output_path == ROOT / "results" / "gguf_qwen2_5_3b_instruct_q4_k_m.json"


def test_slug_helpers_normalize_model_ids_and_quantization() -> None:
    assert _slug("Org/Model-1.5:Q4-K-M") == "org_model_1_5_q4_k_m"
    assert _baseline_output_name("Org/Model-1.5") == "baseline_org_model_1_5.json"
    assert _airllm_output_name("Org/Model-1.5") == "airllm_org_model_1_5.json"
    assert _gguf_output_name("Org/Model-1.5:Q4", "Q4-K-M") == "gguf_org_model_1_5_q4_q4_k_m.json"


def test_required_config_helpers_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        _required_dict({"model": "not-an-object"}, "model")

    with pytest.raises(ValueError, match="must be a non-empty string"):
        _required_str({"id": ""}, "id")
