# Product Requirements Document

## Project Overview

This project implements Exercise 05: a local/on-premises LLM deployment experiment using AirLLM, quantization, and performance benchmarking. The final submission must demonstrate a practical engineering understanding of why large language models are difficult to run locally, how AirLLM changes memory allocation behavior, and when local execution is economically justified compared with external API usage.

## Problem Statement

Large language models often exceed the RAM or VRAM available on personal hardware. Direct execution may fail, hang, swap heavily, or produce unusable latency. The project must show this limitation with a baseline experiment, then evaluate whether AirLLM and quantization can make the same model runnable on modest hardware.

The assignment is evaluated mainly on experimental design, measurements, analysis, and explanation, not on producing the highest quality model output.

## Target Reader

The target reader is an external technical reviewer who wants to reproduce the experiment and understand:

- What hardware was used.
- Which model was selected and why.
- What failed or bottlenecked in the baseline.
- How AirLLM and quantization changed memory and latency behavior.
- Whether local inference is worth using compared with an API.

## Goals

- Document the local hardware precisely: CPU, core count, RAM, GPU, VRAM, and storage type.
- Select a Hugging Face model that is large enough to stress the hardware but still reasonable for an AirLLM experiment.
- Run a direct baseline using Hugging Face, Ollama, or another clearly documented path.
- Run the same task through AirLLM with quantization where feasible.
- Measure latency, throughput, memory, runtime, and qualitative output quality.
- Connect observed results to lecture concepts: Prefill, Decode, compute-bound execution, memory-bound execution, VRAM, virtual memory, paging, and mmap.
- Compare local/on-prem cost against third-party API cost and identify a break-even point.
- Produce a reproducible repository with scripts, results, figures, screenshots, and a detailed README.

## Non-Goals

- Training a large model from scratch.
- Optimizing for production-grade serving latency.
- Building a web application or user-facing UI.
- Proving that AirLLM is always faster or better than direct inference.
- Hiding negative results. Failed or slow runs are valid if documented and analyzed.

## Functional Requirements

- The project shall include scripts for collecting hardware information.
- The project shall include scripts for running a baseline inference attempt.
- The project shall include scripts for running an AirLLM inference attempt.
- The project shall include benchmark collection for:
  - Time to first token, `TTFT`.
  - Time per output token, `TPOT`, or inter-token latency.
  - Throughput in tokens per second.
  - Peak RAM usage.
  - Peak VRAM usage when a GPU exists.
  - Total runtime.
  - Approximate electricity or power cost.
- The project shall save raw benchmark results in a machine-readable format.
- The project shall generate tables or figures from the collected results.
- The project shall include an economic comparison of local execution versus API usage.
- The project shall include at least one original extension beyond the minimum required workflow.

## Non-Functional Requirements

- The experiment must be reproducible from documented commands.
- Secrets, API keys, and Hugging Face tokens must not be committed.
- Configuration values should live in config files or environment variables rather than being hard-coded in logic.
- Code should be modular enough to inspect and extend.
- Generated results should be traceable to the exact command, model, prompt, and quantization setting used.
- The final README must contain the technical report, tables, graphs, screenshots, conclusions, and reproduction instructions.

## Success Criteria

- A reviewer can understand the selected hardware, model, and experiment setup from the README.
- Baseline and AirLLM results are presented in comparable tables and figures.
- The report explicitly answers the research questions from the assignment.
- The cost analysis includes assumptions, formulas, and a break-even graph or table.
- The project includes enough raw data to reproduce the reported graphs.
- The final repository structure is clear and aligned with the course submission guidelines.

## Assumptions

- The local machine may have limited or no usable GPU VRAM.
- Some baseline runs may fail or become too slow; those outcomes are acceptable evidence.
- The exact model and quantization levels may change after hardware inspection.
- Network-dependent model downloads may require manual authentication or prior local cache setup.

## Constraints

- The work should avoid downloading unrealistically large models before the hardware and disk space are checked.
- Initial tests should use low token limits to confirm that the pipeline works.
- AirLLM layer shard/cache paths should be directed to a known fast storage location.
- The final submission must be a GitHub repository with documentation and runnable code.

## Deliverables

- `README.md` as the main technical report.
- `docs/PRD.md`, `docs/PLAN.md`, and `docs/TODO.md`.
- Experiment scripts under `src/` or `experiments/`.
- Raw benchmark outputs under `results/`.
- Generated figures under `figures/`.
- Screenshots or terminal evidence where useful.
- Dependency files such as `pyproject.toml` and `uv.lock` or `requirements.txt`.

