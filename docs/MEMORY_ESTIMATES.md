# Model Memory Estimates

## Purpose

This note estimates memory requirements before downloading any larger model
weights. The estimates are planning numbers, not measured results. Actual usage
depends on backend implementation, context length, prompt length, generated token
count, quantization format, attention implementation, operating-system paging,
and framework overhead.

No model weights are downloaded as part of this note.

## Local Hardware Baseline

The hardware inspection in `results/hardware.json` reports:

| Component | Value |
| --- | --- |
| CPU | Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz |
| CPU cores | 4 physical cores, 8 logical cores |
| System RAM | 15.8 GB |
| GPU | Intel(R) UHD Graphics |
| Reported VRAM | 1.0 GB |
| Free disk | About 316 GB on `C:\` |

This machine should be treated as a CPU/RAM-limited inference environment. The
integrated GPU is not a practical CUDA target for these experiments.

## Estimation Method

Approximate model-weight memory:

| Numeric format | Rule of thumb |
| --- | --- |
| FP32 | 4 bytes per parameter |
| FP16 / BF16 | 2 bytes per parameter |
| INT8 | About 1 byte per parameter, plus metadata |
| 4-bit GGUF | About 0.5 bytes per parameter, plus quantization metadata |

Runtime memory is higher than weight memory because the process also needs:

- Tokenizer files and Python/backend overhead.
- Temporary tensors and activations during prefill.
- KV cache during decoding.
- Backend-specific buffers.
- Operating-system file cache and paging headroom.

For this project, "fits in RAM" should mean there is enough headroom to run
without heavy paging, not merely that the model-weight file is smaller than
15.8 GB.

## Candidate Model Estimates

| Model | Role | Approx params / file scale | Expected weight memory | Practical RAM expectation on this machine | Risk |
| --- | --- | --- | --- | --- | --- |
| `sshleifer/tiny-gpt2` | Pipeline validation only | Tiny test model, far below 1B params | Negligible relative to system RAM | Already ran successfully with peak process RAM around 358 MB in `results/baseline_tiny_gpt2.json` | Low |
| `microsoft/Phi-3-mini-4k-instruct` | Transformers smoke-test candidate | About 3.8B params | About 7.6 GB in FP16/BF16; about 15.2 GB in FP32 | Likely heavy but plausible as a carefully staged CPU/RAM smoke test if loaded in reduced precision; limited headroom for framework overhead and KV cache | Medium to high |
| `Qwen/Qwen2.5-7B-Instruct-GGUF` | GGUF quantized baseline/stress candidate | 7B-class model, quantized GGUF variants | Q4 variants are commonly around 4-5 GB on disk; higher-bit variants use more | Best candidate for a direct local quantized baseline through Ollama or llama.cpp-style tooling; still CPU-limited and may be slow | Medium |
| `Qwen/Qwen2.5-7B-Instruct` | AirLLM candidate | 7B-class HF/SafeTensors model, documented as BF16 in the compatibility note | Roughly 14-16 GB for BF16 weights before overhead | Direct full-load Transformers execution is likely too close to or above available RAM after overhead; AirLLM is the intended feasibility path | High without AirLLM; medium with AirLLM but slow |

## Detailed Notes

### `sshleifer/tiny-gpt2`

This model is only a plumbing check. The completed tiny baseline run already
validated model loading, tokenization, generation, timing, memory sampling, and
JSON result writing. It should not be used as evidence that the final assignment
model fits locally.

### `microsoft/Phi-3-mini-4k-instruct`

Phi-3 Mini is a reasonable next smoke test because it is much larger than
`sshleifer/tiny-gpt2` but smaller than the Qwen 7B candidates. Its expected
FP16/BF16 weight memory is still large for a 15.8 GB RAM machine. A successful
run would need careful generation settings and memory logging; a failure or
paging-heavy run would still be useful evidence for the report.

### `Qwen/Qwen2.5-7B-Instruct-GGUF`

The GGUF candidate is the most practical path for a quantized local baseline.
It should be tested through a GGUF-native backend such as Ollama or llama.cpp
only after the user explicitly approves installing the backend and downloading
the model. A Q4 variant is expected to fit in system RAM more comfortably than a
full-precision 7B model, but throughput will still be limited by the laptop CPU.

### `Qwen/Qwen2.5-7B-Instruct`

The Hugging Face/SafeTensors Qwen candidate is the more plausible AirLLM target
because AirLLM expects Hugging Face-style model layouts rather than GGUF. Direct
full-load execution is expected to be difficult on this machine: BF16 weights
alone can approach the full RAM capacity once runtime overhead is included.
This is the key reason AirLLM, layer-wise loading, and quantization are relevant
for the assignment.

## Current Decision

Proceed in stages:

1. Keep `sshleifer/tiny-gpt2` as completed pipeline validation evidence.
2. Use `microsoft/Phi-3-mini-4k-instruct` as the next optional Transformers
   smoke test only after explicit approval to download it.
3. Use `Qwen/Qwen2.5-7B-Instruct-GGUF` for a quantized GGUF baseline only after
   explicit approval to install/use a GGUF backend and download the selected
   quantized variant.
4. Use `Qwen/Qwen2.5-7B-Instruct` as the AirLLM candidate only after explicit
   approval to install AirLLM and download/cache the model.

