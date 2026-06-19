# Technical Plan

## Architecture Summary

The project will be organized as a reproducible benchmark workflow. Scripts will collect hardware details, run controlled inference experiments, save raw measurements, and generate figures for the final README report.

The planned flow is:

1. Inspect local hardware and storage.
2. Choose a model that stresses the machine.
3. Run a direct baseline attempt.
4. Run AirLLM with quantization options.
5. Save raw benchmark data.
6. Generate comparison tables and figures.
7. Write the final README report and economic analysis.

## Proposed Repository Structure

```text
.
├── README.md
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   └── PRD_benchmarking.md
├── src/
│   └── airllm_benchmark/
│       ├── __init__.py
│       ├── config.py
│       ├── hardware.py
│       ├── metrics.py
│       ├── economics.py
│       └── runners/
│           ├── baseline.py
│           └── airllm_runner.py
├── experiments/
│   ├── collect_hardware.py
│   ├── run_baseline.py
│   ├── run_airllm.py
│   └── make_figures.py
├── config/
│   └── experiment.example.json
├── results/
├── figures/
├── materials/
├── pyproject.toml
└── .gitignore
```

This structure may be adjusted after implementation begins, but the repository should remain easy to navigate.

## Main Components

### Hardware Collection

Collects and records:

- CPU model and core count.
- Total RAM.
- GPU model and VRAM when available.
- Storage type or available disk information.
- Operating system and Python version.

Output target: `results/hardware.json`.

### Baseline Runner

Attempts direct local inference using the selected model. The exact backend will be chosen after hardware inspection, likely Hugging Face Transformers or Ollama.

Responsibilities:

- Load or invoke the selected model.
- Run a fixed prompt with fixed generation settings.
- Record success, failure, error messages, runtime, token counts, and memory usage.
- Capture enough evidence to explain the bottleneck.

### AirLLM Runner

Runs the same prompt and generation settings through AirLLM.

Responsibilities:

- Configure AirLLM cache and layer shard paths.
- Run one or more quantization settings where supported.
- Record latency, throughput, token counts, memory usage, and output sample.
- Preserve errors or warnings for the final analysis.

### Metrics Module

Defines consistent metrics across all runs:

- `ttft_seconds`
- `tpot_seconds`
- `tokens_per_second`
- `input_tokens`
- `output_tokens`
- `peak_ram_mb`
- `peak_vram_mb`
- `total_runtime_seconds`
- `status`
- `notes`

### Economics Module

Computes local/on-prem and API costs.

Inputs:

- Hardware purchase cost estimate.
- Hardware lifetime in months.
- Electricity price.
- Power draw estimate.
- Monthly request or token volume.
- API input and output token pricing.
- Optional prompt caching discount assumptions.

Outputs:

- Cost per request.
- Monthly local cost.
- Monthly API cost.
- Break-even request or token volume.
- Data table for a cost curve.

## Experiment Design

### Controlled Prompt

Use one or more fixed prompts that are long enough to exercise both Prefill and Decode but short enough to keep local runs practical.

Initial prompt category:

- Technical explanation or summarization task.

Generation settings should include:

- Fixed maximum output tokens.
- Fixed temperature.
- Fixed model identifier.
- Fixed quantization setting per run.

### Baseline Conditions

At minimum:

- Direct model load or Ollama execution.
- Same prompt and output length target as AirLLM.
- Document whether the run completes, fails, or becomes unusably slow.

### AirLLM Conditions

At minimum:

- AirLLM run with the same model family when feasible.
- Quantization setting documented.

Preferred if time and hardware allow:

- Compare at least two quantization levels such as FP16/Q8/Q4, or the nearest supported equivalents.

### Original Extension

Candidate extension:

- Add a break-even sensitivity analysis showing how API prompt caching changes the on-prem break-even point.

Alternative extensions:

- Compare two model sizes.
- Add a memory timeline plot.
- Add a qualitative output degradation table across quantization levels.

## Data Outputs

Raw data should be stored as JSON or CSV in `results/`.

Recommended files:

- `results/hardware.json`
- `results/baseline_results.json`
- `results/airllm_results.json`
- `results/economics.csv`
- `results/summary.csv`

Figures should be stored in `figures/`.

Recommended figures:

- `figures/performance_comparison.png`
- `figures/memory_comparison.png`
- `figures/cost_break_even.png`
- `figures/ttft_tpot_comparison.png`

## README Report Outline

The final README should include:

- Project overview.
- Hardware specification.
- Model selection justification.
- Experiment setup and reproduction commands.
- Baseline results.
- AirLLM and quantization results.
- Performance comparison tables and graphs.
- Economic analysis and break-even point.
- Lecture concept analysis.
- Original extension.
- Limitations and conclusions.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Model download is too large | Blocks experiments | Start with hardware/disk inspection and choose a feasible model |
| AirLLM dependency incompatibility | Blocks AirLLM run | Pin Python/dependency versions and document fallback evidence |
| Baseline freezes or swaps heavily | Slows work | Use small token limits and timeouts |
| GPU unavailable | Limits comparison | Emphasize CPU/RAM/AirLLM paging analysis |
| Metrics differ across backends | Reduces comparability | Use shared prompt, token limits, and normalized result schema |
| API prices change | Affects economics | Record pricing date and assumptions in README |

## Quality Plan

- Keep source files small and focused.
- Use `uv` for environment and dependency management where possible.
- Avoid committing secrets.
- Store generated raw data separately from code.
- Prefer reproducible scripts over manual notebook-only analysis.
- Add focused tests for cost calculations and metric utilities if implementation includes non-trivial logic.

