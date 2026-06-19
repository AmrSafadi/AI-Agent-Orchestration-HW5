# TODO

## Phase 1: Documentation Setup

- [x] Create `docs/PRD.md`.
- [x] Create `docs/PLAN.md`.
- [x] Create `docs/TODO.md`.
- [x] Create `docs/PRD_benchmarking.md`.
- [x] Create model selection note.
- [x] Create initial `README.md` report shell.

## Phase 2: Project Skeleton

- [x] Create Python project metadata.
- [x] Install minimum Python dependencies for tiny baseline smoke test.
- [x] Create source package structure.
- [x] Create experiment script directory.
- [x] Create config directory and example config.
- [x] Create results and figures directories.
- [x] Add `.gitignore` for environments, caches, model shards, and secrets.

## Phase 3: Hardware Inspection

- [x] Implement hardware collection script.
- [x] Record CPU model and core count.
- [x] Record RAM size.
- [x] Record GPU model and VRAM if available.
- [x] Record storage and available disk space.
- [x] Save hardware output to `results/hardware.json`.

## Phase 4: Model Selection

- [x] Choose candidate Hugging Face model.
- [x] Check preliminary license and format.
- [x] Create model selection note.
- [x] Create backend compatibility note.
- [x] Estimate memory requirements.
- [x] Justify why the model stresses the local machine.
- [x] Verify installed backend availability without downloading model weights.
- [x] Finalize backend-specific model IDs.
- [x] Record selected model in config and README.

## Phase 5: Baseline Experiment

- [x] Implement tiny baseline smoke-test runner.
- [x] Run tiny baseline smoke test with `sshleifer/tiny-gpt2`.
- [x] Save tiny baseline result to `results/baseline_tiny_gpt2.json`.
- [x] Implement Qwen 3B baseline runner.
- [x] Run fixed prompt with fixed generation settings.
- [x] Capture load failure, timeout, or successful output.
- [ ] Measure runtime, memory, and token statistics where possible.
- [x] Save raw baseline results.
- [ ] Add baseline screenshots or terminal evidence if useful.

## Phase 6: AirLLM Experiment

- [x] Create AirLLM experiment plan.
- [x] Configure planned model shard/cache path.
- [x] Install and verify AirLLM dependencies.
- [x] Implement AirLLM runner.
- [x] Run the same prompt and generation settings.
- [x] Run AirLLM backup model attempt.
- [ ] Test quantization settings supported by the environment.
- [x] Save raw AirLLM results.

## Phase 6B: Quantized GGUF Experiment

- [x] Create GGUF quantization plan.
- [x] Select recommended quantized model candidate.
- [x] Select preferred GGUF backend path.
- [x] Install approved GGUF backend.
- [x] Verify GGUF backend availability.
- [x] Download approved quantized GGUF model.
- [x] Implement GGUF/Ollama benchmark runner or wrapper.
- [x] Run the same prompt and generation settings.
- [x] Save raw quantized GGUF result.
- [x] Compare GGUF result against baseline and AirLLM attempts.

## Phase 7: Analysis and Figures

- [x] Implement result loading and summary table generation.
- [x] Add result summary table from existing JSON files.
- [x] Generate latency comparison figure.
- [x] Generate throughput comparison figure.
- [x] Generate memory comparison figure.
- [x] Generate cost break-even figure.
- [x] Add qualitative output comparison table.

## Phase 8: Economic Analysis

- [x] Create economic analysis note.
- [x] Define hardware cost assumptions.
- [x] Define electricity price and power estimate.
- [x] Define API pricing assumptions with date.
- [x] Calculate local cost per request or token volume.
- [x] Calculate API cost for the same workload.
- [x] Identify break-even point.
- [x] Include prompt caching sensitivity if used as the original extension.

## Phase 9: Final Report

- [x] Refresh `docs/PLAN.md` to match the current repository.
- [ ] Write hardware documentation section.
- [ ] Write model choice justification.
- [x] Write baseline analysis.
- [x] Write AirLLM and quantization analysis.
- [ ] Explain Prefill versus Decode in relation to measurements.
- [ ] Explain compute-bound versus memory-bound behavior.
- [ ] Explain AirLLM, virtual memory, paging, and mmap.
- [x] Add economic conclusion.
- [ ] Add limitations and recommended usage scenarios.
- [x] Embed result summary table in `README.md`.
- [x] Embed Phase 7 figures in `README.md`.
- [ ] Embed screenshots in `README.md` if useful.

## Phase 10: Verification

- [ ] Run available tests.
- [ ] Run linting or formatting checks if configured.
- [ ] Verify README commands from a clean shell where practical.
- [ ] Check that no secrets or tokens are committed.
- [ ] Confirm required deliverables are present.
