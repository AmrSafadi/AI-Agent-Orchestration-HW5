# Model Selection Decision

## Purpose

This note documents the model-selection decision before downloading any model
weights. The goal is to choose a model that matches the assignment requirement:
large enough to stress the local machine, but not so large that there is no
realistic chance of running the experiment even with AirLLM and quantization.

No large model weights are downloaded as part of this decision note.

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

This is a laptop-class CPU system with limited integrated graphics rather than a
workstation GPU. The CPU can run small inference jobs, but it is not designed
for high-throughput transformer inference. The GPU is Intel integrated graphics
with only about 1 GB of reported VRAM, so it is not useful for normal CUDA-based
local LLM inference.

The main constraint is system RAM and memory bandwidth. Local LLM inference
requires memory for model weights, tokenizer and runtime overhead, temporary
activations, KV cache, backend buffers, and the operating system. Once memory
pressure rises, Windows may page data to disk, which can turn a technically
running experiment into an unusably slow one.

## Why 7B Is Deferred

A normal 7B model in full precision is too risky as the main target for this
machine. Approximate model-weight memory alone is:

| Format | Approximate weight memory |
| --- | --- |
| FP32 | About 28 GB for 7B parameters |
| FP16 / BF16 | About 14 GB for 7B parameters |

The FP16/BF16 estimate already approaches the full 15.8 GB physical RAM before
counting Python, framework allocations, token buffers, KV cache, and the
operating system. A direct full-load 7B run would likely fail, trigger heavy
paging, or run with unusable latency.

The assignment asks for a model that is large but not impossibly large. For this
machine, 7B is useful as a documented rejected/deferred boundary, but not as the
first large download.

## Selected Main Model

```text
Qwen/Qwen2.5-3B-Instruct
```

Role:

- Main Hugging Face/SafeTensors candidate for direct baseline and AirLLM.
- Large enough to stress 15.8 GB RAM through the Python/Transformers stack.
- More realistic than a full 7B direct-load attempt.
- Same family as a matching GGUF quantized repository.

The Qwen model card reports 3.09B parameters. BF16 weights are roughly 6.2 GB
before runtime overhead, KV cache, tokenizer files, and operating-system memory.
That should create meaningful pressure on this laptop without making the first
experiment an avoidable endurance test.

## Quantized Comparison Candidate

```text
Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M
```

Role:

- Optional quantized local comparison through Ollama or llama.cpp if approved.
- Demonstrates how quantization changes disk size, memory pressure, and
  latency.
- Keeps the experiment in the same model family as the direct HF/AirLLM target.

The GGUF model card lists Q4_K_M at about 2.1 GB. That is much more practical
for local CPU inference than a full-precision 7B model, though throughput is
still expected to be limited by the laptop CPU.

## Backup Candidate

```text
microsoft/Phi-3-mini-4k-instruct
```

Role:

- Backup HF/Transformers candidate if Qwen 3B has compatibility problems.
- Still a 3B-4B class model that can stress the local machine.

Phi-3 Mini is 3.8B parameters, so it is slightly larger than Qwen 3B. It remains
a strong backup, but Qwen 3B gives a cleaner path across HF/SafeTensors, AirLLM,
and matching GGUF quantization.

## Completed Pipeline Validation Model

```text
sshleifer/tiny-gpt2
```

Role:

- Validate Transformers loading, tokenization, generation, timing, memory
  sampling, and JSON result writing.
- Confirm the local benchmark plumbing before any large model download.

This model already ran successfully and is not the assignment model.

## Deferred / Rejected Candidate

```text
Qwen/Qwen2.5-7B-Instruct
```

Role:

- Documented as too risky for this laptop as the main direct-load target.
- May be discussed in the report as a model-size boundary that exceeds the
  practical comfort zone of the available hardware.

The 7B model is not selected for the first large download.

## Current Decision

Proceed with a Qwen 3B-centered plan:

| Role | Candidate model | Purpose |
| --- | --- | --- |
| Completed pipeline validation | `sshleifer/tiny-gpt2` | Prove the local Transformers measurement pipeline works |
| Main HF/AirLLM candidate | `Qwen/Qwen2.5-3B-Instruct` | Stress local RAM while staying plausible for AirLLM |
| Quantized GGUF candidate | `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` | Compare quantized local inference if a GGUF backend is approved |
| Backup HF candidate | `microsoft/Phi-3-mini-4k-instruct` | Alternative 3B-4B class model |
| Deferred as too large | `Qwen/Qwen2.5-7B-Instruct` | Documented but not selected for first download |

No model weights are downloaded as part of this decision note.

