# Technical Plan

## Architecture Summary

This repository is organized as a reproducible local LLM benchmark workflow.
The current experiment has moved past model execution and into report assembly:
hardware, baseline, AirLLM attempts, GGUF quantization, result tables, and Phase
7 figures are already captured. The next planned checkpoint is the economic
analysis required by the assignment.

## Current Workflow

1. Inspect local hardware and storage.
2. Choose a model that is large enough to stress the laptop.
3. Run a direct Transformers baseline.
4. Run AirLLM with the same prompt/settings and preserve failure evidence.
5. Run a quantized GGUF comparison through Ollama.
6. Save raw benchmark data as JSON.
7. Generate comparison tables and figures from saved results.
8. Add on-prem versus API economics and a break-even figure.
9. Finish the README as the main submission report.

## Repository Structure

```text
.
|-- README.md
|-- config/
|   `-- experiment.example.json
|-- docs/
|   |-- AIRLLM_PLAN.md
|   |-- AIRLLM_RESULTS.md
|   |-- BASELINE_RESULTS.md
|   |-- BACKEND_COMPATIBILITY.md
|   |-- ECONOMIC_ANALYSIS.md
|   |-- GGUF_RESULTS.md
|   |-- GGUF_QUANTIZATION_PLAN.md
|   |-- MEMORY_ESTIMATES.md
|   |-- MODEL_SELECTION.md
|   |-- PLAN.md
|   |-- PRD.md
|   |-- PRD_benchmarking.md
|   |-- RESULT_SUMMARY.md
|   `-- TODO.md
|-- experiments/
|   |-- check_backends.py
|   |-- collect_hardware.py
|   |-- make_figures.py
|   |-- run_airllm.py
|   |-- run_baseline.py
|   |-- run_ollama.py
|   `-- summarize_results.py
|-- figures/
|   |-- cost_break_even.svg
|   |-- decode_latency_comparison.svg
|   |-- memory_comparison.svg
|   |-- run_status_summary.svg
|   `-- throughput_comparison.svg
|-- materials/
|-- results/
|-- src/
|   `-- airllm_benchmark/
|       |-- __init__.py
|       |-- backends.py
|       |-- config.py
|       |-- hardware.py
|       |-- metrics.py
|       `-- runners/
|           |-- __init__.py
|           |-- airllm.py
|           |-- baseline.py
|           `-- ollama.py
|-- pyproject.toml
`-- uv.lock
```

## Main Components

### Hardware Collection

`experiments/collect_hardware.py` records CPU, core count, RAM, GPU, reported
VRAM, storage, operating system, and Python version to `results/hardware.json`.

### Backend Check

`experiments/check_backends.py` records which local inference dependencies and
backend tools are available. The output is saved in `results/backend_check.json`.

### Baseline Runner

`experiments/run_baseline.py` supports the tiny Transformers smoke test and the
main Qwen 3B Transformers baseline. It records success, failure, timeout,
runtime, token counts where available, memory where measurable, output samples,
and errors in the shared benchmark JSON schema.

### AirLLM Runner

`experiments/run_airllm.py` runs the same fixed prompt/settings through AirLLM.
The current evidence includes a Qwen 3B model-layout failure and a Phi-3 backup
failure caused by BetterTransformer support limits. These are valid deployment
compatibility findings for the final report.

### GGUF/Ollama Runner

`experiments/run_ollama.py` benchmarks the quantized
`hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` model through local Ollama. This is
the successful local inference result.

### Analysis Scripts

`experiments/summarize_results.py` prints the Markdown result summary table from
existing JSON files. `experiments/make_figures.py` generates the Phase 7 SVG
figures from those same saved result files without rerunning any models.

## Result Files

Current raw outputs include:

- `results/hardware.json`
- `results/backend_check.json`
- `results/baseline_tiny_gpt2.json`
- `results/baseline_qwen_qwen2_5_3b_instruct.json`
- `results/airllm_install_check.json`
- `results/airllm_qwen_qwen2_5_3b_instruct.json`
- `results/airllm_phi3_mini_instruct.json`
- `results/gguf_qwen2_5_3b_instruct_q4_k_m.json`
- `results/economics_summary.json`
- `results/economics_curve.csv`

Temporary smoke-test files use the `.tmp` extension and are ignored by Git.

## Figure Files

Current Phase 7 figures:

- `figures/run_status_summary.svg`
- `figures/throughput_comparison.svg`
- `figures/decode_latency_comparison.svg`
- `figures/memory_comparison.svg`
- `figures/cost_break_even.svg`

The next report checkpoint should promote the remaining concept explanations and
limitations into the README.

## Remaining Plan

### Phase 8: Economic Analysis

Add reproducible on-prem versus API cost calculations:

- Hardware cost and lifetime assumptions.
- Electricity price and estimated power draw.
- API input/output token pricing with date.
- Local cost per request/token-volume estimate.
- API cost for the same workload.
- Break-even table and figure.
- Optional prompt/context caching sensitivity.

### Phase 9: Final README Report

Promote the existing supporting docs into a complete README report:

- Hardware specification.
- Model selection justification.
- Experiment methodology.
- Baseline, AirLLM, and GGUF result interpretation.
- Lecture concept analysis: Prefill, Decode, compute-bound, memory-bound, VRAM,
  virtual memory, paging, `mmap`, and quantization.
- Economic conclusion.
- Limitations and recommendation.

### Phase 10: Verification

Before submission:

- Re-run non-expensive reproduction commands.
- Confirm figures can be regenerated.
- Check for secrets/tokens.
- Verify required deliverables are present.
- Record missing optional tooling explicitly if tests/linting are unavailable.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| API prices change | Economics can become stale | Record pricing date and assumptions |
| TTFT is unavailable for Ollama result | Metric table has a null field | Explain that the run used non-streaming API output |
| AirLLM did not complete generation | Fewer comparable performance metrics | Treat failures as backend compatibility evidence |
| Direct Transformers timed out | No full baseline token metrics | Use timeout as valid negative baseline evidence |
| README becomes too scattered | Submission clarity suffers | Consolidate final conclusions in README, keep docs as appendices |

## Quality Plan

- Keep future changes scoped to the active phase.
- Prefer generated tables/figures from saved JSON over manual numbers.
- Avoid rerunning expensive model downloads unless explicitly needed.
- Do not commit model caches, shards, tokens, or temporary smoke files.
- Use `uv` for reproducible commands.
