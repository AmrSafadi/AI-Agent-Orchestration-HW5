"""Experiment config loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineRunConfig:
    """Configuration needed for a Transformers baseline run."""

    model_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    output_path: Path
    local_files_only: bool = True
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class AirLLMRunConfig:
    """Configuration needed for an AirLLM run."""

    model_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    output_path: Path
    huggingface_cache_dir: Path
    layer_shards_saving_path: Path
    compression: str | None
    delete_original: bool
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class GGUFRunConfig:
    """Configuration needed for an Ollama/GGUF run."""

    backend: str
    model_id: str
    quantization: str
    prompt: str
    max_new_tokens: int
    temperature: float
    output_path: Path
    timeout_seconds: int | None = None


def load_baseline_config(config_path: Path, project_root: Path) -> BaselineRunConfig:
    """Load the baseline subset of the experiment JSON config."""

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    model = _required_dict(raw, "model")
    generation = _required_dict(raw, "generation")
    paths = _required_dict(raw, "paths")

    results_dir = project_root / str(paths.get("results_dir", "results"))
    model_id = _required_str(model, "id")
    output_name = _baseline_output_name(model_id)

    return BaselineRunConfig(
        model_id=model_id,
        prompt=_required_str(generation, "prompt"),
        max_new_tokens=int(generation.get("max_new_tokens", 64)),
        temperature=float(generation.get("temperature", 0.0)),
        output_path=results_dir / output_name,
        local_files_only=True,
        timeout_seconds=None,
    )


def load_airllm_config(config_path: Path, project_root: Path) -> AirLLMRunConfig:
    """Load the AirLLM subset of the experiment JSON config."""

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    model = _required_dict(raw, "model")
    generation = _required_dict(raw, "generation")
    paths = _required_dict(raw, "paths")
    airllm = _required_dict(raw, "airllm")

    model_id = str(airllm.get("model_id") or _required_str(model, "id"))
    results_dir = project_root / str(paths.get("results_dir", "results"))
    result_file = str(airllm.get("result_file") or _airllm_output_name(model_id))
    hf_cache = project_root / str(airllm.get("huggingface_cache_dir", "model_cache/huggingface"))
    layer_path = project_root / _required_str(airllm, "layer_shards_saving_path")

    compression = airllm.get("compression")
    if compression is not None and not isinstance(compression, str):
        raise ValueError("Config key 'airllm.compression' must be null or a string.")

    return AirLLMRunConfig(
        model_id=model_id,
        prompt=_required_str(generation, "prompt"),
        max_new_tokens=int(generation.get("max_new_tokens", 64)),
        temperature=float(generation.get("temperature", 0.0)),
        output_path=results_dir / result_file,
        huggingface_cache_dir=hf_cache,
        layer_shards_saving_path=layer_path,
        compression=compression,
        delete_original=bool(airllm.get("delete_original", False)),
        timeout_seconds=int(airllm["timeout_seconds"]) if airllm.get("timeout_seconds") else None,
    )


def load_gguf_config(config_path: Path, project_root: Path) -> GGUFRunConfig:
    """Load the GGUF/Ollama subset of the experiment JSON config."""

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    generation = _required_dict(raw, "generation")
    paths = _required_dict(raw, "paths")
    gguf = _required_dict(raw, "gguf")

    backend = str(gguf.get("backend", "ollama"))
    model_id = _required_str(gguf, "model_id")
    quantization = _required_str(gguf, "quantization")
    results_dir = project_root / str(paths.get("results_dir", "results"))
    result_file = str(gguf.get("result_file") or _gguf_output_name(model_id, quantization))

    return GGUFRunConfig(
        backend=backend,
        model_id=model_id,
        quantization=quantization,
        prompt=_required_str(generation, "prompt"),
        max_new_tokens=int(generation.get("max_new_tokens", 64)),
        temperature=float(generation.get("temperature", 0.0)),
        output_path=results_dir / result_file,
        timeout_seconds=int(gguf["timeout_seconds"]) if gguf.get("timeout_seconds") else None,
    )


def _baseline_output_name(model_id: str) -> str:
    slug = (
        model_id.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
    return f"baseline_{slug}.json"


def _airllm_output_name(model_id: str) -> str:
    slug = (
        model_id.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
    return f"airllm_{slug}.json"


def _gguf_output_name(model_id: str, quantization: str) -> str:
    slug = (
        model_id.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
    quant_slug = quantization.lower().replace("-", "_")
    return f"gguf_{slug}_{quant_slug}.json"


def _required_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be an object.")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config key '{key}' must be a non-empty string.")
    return value
