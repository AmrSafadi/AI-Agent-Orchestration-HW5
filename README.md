# AirLLM Local Inference Benchmark

This repository is the submission project for Exercise 05: running a large language model locally with AirLLM, quantization, and performance benchmarking.

The goal is to document a complete local/on-prem LLM experiment: hardware limits, baseline behavior, AirLLM behavior, performance metrics, and economic comparison against external API usage.

## Current Status

- Documentation scaffold exists under `docs/`.
- Project directories exist for source code, experiments, config, results, and figures.
- Hardware collection is implemented as the first reproducible experiment.
- Main assignment model is `Qwen/Qwen2.5-3B-Instruct`.
- Direct Transformers baseline timed out after 900 seconds.
- AirLLM is installed and an AirLLM runner entry point exists.

## Repository Structure

```text
.
|-- config/
|   `-- experiment.example.json
|-- docs/
|   |-- AIRLLM_PLAN.md
|   |-- BASELINE_RESULTS.md
|   |-- BACKEND_COMPATIBILITY.md
|   |-- MEMORY_ESTIMATES.md
|   |-- MODEL_SELECTION.md
|   |-- PLAN.md
|   |-- PRD.md
|   |-- PRD_benchmarking.md
|   `-- TODO.md
|-- experiments/
|   |-- collect_hardware.py
|   |-- check_backends.py
|   |-- run_airllm.py
|   `-- run_baseline.py
|-- figures/
|-- materials/
|-- results/
|-- src/
|   `-- airllm_benchmark/
|       |-- __init__.py
|       |-- hardware.py
|       |-- metrics.py
|       `-- runners/
|           |-- __init__.py
|           |-- airllm.py
|           `-- baseline.py
|-- pyproject.toml
`-- README.md
```

## Setup

This project uses `uv`.

```powershell
uv run python experiments/collect_hardware.py
```

The command writes hardware information to:

```text
results/hardware.json
```

## Tiny Baseline Smoke Test

The tiny baseline smoke test validates the benchmark pipeline with
`sshleifer/tiny-gpt2`. This is only a plumbing check for Transformers loading,
generation, timing, memory sampling, and JSON output. It is not the final model
for the assignment.

```powershell
uv run python experiments/run_baseline.py
```

The command writes the smoke-test benchmark result to:

```text
results/baseline_tiny_gpt2.json
```

## Planned Model Strategy

| Role | Model |
| --- | --- |
| Completed pipeline validation | `sshleifer/tiny-gpt2` |
| Main HF/AirLLM candidate | `Qwen/Qwen2.5-3B-Instruct` |
| Optional quantized GGUF comparison | `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` |
| Backup HF candidate | `microsoft/Phi-3-mini-4k-instruct` |
| Deferred as too large for first download | `Qwen/Qwen2.5-7B-Instruct` |

The selected 3B Qwen model is intended to stress the 15.8 GB RAM laptop without
making the first real experiment as risky as a full 7B BF16 direct load.

## Current Baseline Result

The direct Transformers baseline for `Qwen/Qwen2.5-3B-Instruct` timed out after
900 seconds with the fixed prompt and a 32-token generation limit. The raw result
is stored in:

```text
results/baseline_qwen_qwen2_5_3b_instruct.json
```

This is a valid negative baseline outcome: direct BF16 Transformers execution is
not comfortable on the current CPU/RAM-only laptop setup.

## Planned AirLLM Experiment

The next planned intervention is AirLLM with the same
`Qwen/Qwen2.5-3B-Instruct` model, prompt, and generation settings. The planned
layer/cache path is:

```text
airllm_cache/qwen2_5_3b_instruct
```

AirLLM has been installed and verified through the project `uv` environment.
The exact `airllm.__version__` attribute check fails because the package does
not expose that attribute, but package metadata reports `airllm==2.11.0`. The
install check is stored in `results/airllm_install_check.json`, and the detailed
plan is documented in `docs/AIRLLM_PLAN.md`.

## Planned Report Sections

- Hardware specification.
- Model selection and justification.
- Direct baseline run.
- AirLLM and quantization run.
- Performance comparison.
- Economic comparison: on-prem versus API.
- Lecture concept analysis: Prefill, Decode, VRAM, paging, and memory-bound behavior.
- Original extension and conclusions.
