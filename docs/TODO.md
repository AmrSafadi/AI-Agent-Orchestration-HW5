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
- [ ] Estimate memory requirements.
- [x] Justify why the model stresses the local machine.
- [ ] Verify installed backend availability without downloading model weights.
- [ ] Finalize backend-specific model IDs.
- [ ] Record selected model in config and README.

## Phase 5: Baseline Experiment

- [ ] Implement baseline runner.
- [ ] Run fixed prompt with fixed generation settings.
- [ ] Capture load failure, timeout, or successful output.
- [ ] Measure runtime, memory, and token statistics where possible.
- [ ] Save raw baseline results.
- [ ] Add baseline screenshots or terminal evidence if useful.

## Phase 6: AirLLM Experiment

- [ ] Install and verify AirLLM dependencies.
- [ ] Configure model shard/cache path.
- [ ] Implement AirLLM runner.
- [ ] Run the same prompt and generation settings.
- [ ] Test quantization settings supported by the environment.
- [ ] Save raw AirLLM results.

## Phase 7: Analysis and Figures

- [ ] Implement result loading and summary table generation.
- [ ] Generate latency comparison figure.
- [ ] Generate throughput comparison figure.
- [ ] Generate memory comparison figure.
- [ ] Generate cost break-even figure.
- [ ] Add qualitative output comparison table.

## Phase 8: Economic Analysis

- [ ] Define hardware cost assumptions.
- [ ] Define electricity price and power estimate.
- [ ] Define API pricing assumptions with date.
- [ ] Calculate local cost per request or token volume.
- [ ] Calculate API cost for the same workload.
- [ ] Identify break-even point.
- [ ] Include prompt caching sensitivity if used as the original extension.

## Phase 9: Final Report

- [ ] Write hardware documentation section.
- [ ] Write model choice justification.
- [ ] Write baseline analysis.
- [ ] Write AirLLM and quantization analysis.
- [ ] Explain Prefill versus Decode in relation to measurements.
- [ ] Explain compute-bound versus memory-bound behavior.
- [ ] Explain AirLLM, virtual memory, paging, and mmap.
- [ ] Add economic conclusion.
- [ ] Add limitations and recommended usage scenarios.
- [ ] Embed tables, figures, and screenshots in `README.md`.

## Phase 10: Verification

- [ ] Run available tests.
- [ ] Run linting or formatting checks if configured.
- [ ] Verify README commands from a clean shell where practical.
- [ ] Check that no secrets or tokens are committed.
- [ ] Confirm required deliverables are present.
