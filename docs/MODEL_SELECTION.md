# Model Selection Decision

## Purpose

This note documents the model-selection decision before downloading any model weights. The goal is to choose models that match the assignment requirements while respecting the limits of the local machine.

No large model weights should be downloaded until backend compatibility and disk/cache paths are checked.

## Hardware Summary

The hardware inspection in `results/hardware.json` reports:

| Component | Value |
| --- | --- |
| CPU | Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz |
| CPU cores | 4 physical cores, 8 logical cores |
| RAM | 15.8 GB |
| GPU | Intel(R) UHD Graphics |
| Reported VRAM | 1.0 GB |
| OS | Windows 11 |
| Free disk | About 316 GB on `C:\` |

## Hardware Interpretation

This is a laptop-class CPU system with limited integrated graphics rather than a workstation GPU. The CPU can run small inference jobs, but it is not designed for high-throughput transformer inference. The GPU is Intel integrated graphics with only about 1 GB of reported VRAM, so it is not useful for normal CUDA-based local LLM inference.

The available disk space is acceptable for experimentation, but model downloads and AirLLM layer shard/cache files must still be managed carefully. AirLLM can create many large files, so cache locations should be explicit and monitored.

## Why This Machine Is Memory-Limited

The main constraint is memory, not just raw compute. Local LLM inference requires memory for:

- Model weights.
- Tokenizer and runtime overhead.
- Intermediate activations.
- The KV cache created during generation.
- Backend framework overhead.

With only 15.8 GB of system RAM and no practical dedicated VRAM, the machine has little room for a full-precision multi-billion-parameter model. Once RAM pressure rises, Windows may page data to disk. Paging can make a run technically continue, but latency becomes very high because disk I/O is much slower than RAM.

This makes the machine a useful fit for the assignment: direct local inference is expected to expose memory bottlenecks, while AirLLM can be evaluated as a memory-saving strategy that trades speed for feasibility.

## Why A Full-Precision 7B Model Is Probably Too Heavy

A normal 7B model in full precision is likely too heavy for this machine. Approximate model-weight memory alone is:

- FP32: about 28 GB for 7B parameters.
- FP16/BF16: about 14 GB for 7B parameters.

The FP16 estimate already approaches the full physical RAM of the machine before counting runtime overhead, token buffers, Python process memory, framework allocations, and KV cache. In practice, a direct full-precision 7B load would likely fail, trigger heavy paging, or run with unusable latency.

Quantized formats reduce the weight memory significantly, but quantization does not remove all overhead. This is why a direct baseline with a 7B-class model is expected to be challenging, and why the experiment should be staged carefully.

## Safer Two-Model Strategy

The safer approach is to use two models with different roles.

### A. Smoke-Test Model

Candidate:

```text
microsoft/Phi-3-mini-4k-instruct
```

Role:

- Validate the benchmark pipeline before using a larger model.
- Check prompt handling, token counting, timing, result serialization, and memory logging.
- Reduce debugging time because the model is smaller than a 7B-class model.
- Confirm that the selected backend path works before spending time on large downloads or AirLLM cache generation.

This model is not the main stress-test evidence for the assignment. It is a practical engineering step to prove the scripts and measurements work.

### B. Main Stress-Test Model

Candidate:

```text
Qwen/Qwen2.5-7B-Instruct-GGUF
```

Role:

- Provide the main assignment stress test.
- Represent a 7B-class model that is plausibly too large for comfortable direct local execution on this hardware.
- Support the required analysis of memory pressure, latency, quantization, and local/on-prem tradeoffs.
- Give a realistic case where quantization and model format choices matter.

The GGUF format is especially relevant for local inference tooling such as Ollama or llama.cpp-style runners. However, AirLLM compatibility must be checked carefully because AirLLM workflows may expect Hugging Face/SafeTensors-style model layouts rather than GGUF files.

## Compatibility Caveat

The final model choice may still change after checking exact backend compatibility with:

- Ollama.
- Hugging Face Transformers.
- AirLLM.

The important decision at this stage is the strategy, not a final irreversible download. The project should first confirm which formats and model families are supported by each backend, then choose the smallest set of downloads that supports the baseline, AirLLM, quantization, and reporting requirements.

## Current Decision

Proceed with a two-model plan:

| Role | Candidate model | Purpose |
| --- | --- | --- |
| Smoke test | `microsoft/Phi-3-mini-4k-instruct` | Validate benchmark scripts and measurement pipeline |
| Main stress test | `Qwen/Qwen2.5-7B-Instruct-GGUF` | Demonstrate memory-limited local inference and quantization tradeoffs |

No model weights are downloaded as part of this decision note.

