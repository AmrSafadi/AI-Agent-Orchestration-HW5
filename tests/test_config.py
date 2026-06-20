from pathlib import Path

from airllm_benchmark.config import load_airllm_config, load_baseline_config, load_gguf_config


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
