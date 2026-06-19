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
| `Qwen/Qwen2.5-3B-Instruct` | Main HF/AirLLM candidate | 3.09B params | About 6.2 GB in BF16; about 12.4 GB in FP32 | Good stress fit: likely heavy under Transformers on CPU/RAM, but not as unrealistic as 7B | Medium to high |
| `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` | Quantized GGUF comparison candidate | 3.09B params, quantized GGUF | Model card lists Q4_K_M around 2.1 GB | Plausible local CPU quantized run if Ollama or llama.cpp is approved; still expected to be slow | Medium |
| `microsoft/Phi-3-mini-4k-instruct` | Backup HF candidate | About 3.8B params | About 7.6 GB in FP16/BF16; about 15.2 GB in FP32 | Strong backup, but slightly larger than Qwen 3B and less clean for matching GGUF/AirLLM comparison | Medium to high |
| `Qwen/Qwen2.5-7B-Instruct` | Deferred too-large candidate | 7B-class HF/SafeTensors model | Roughly 14-16 GB for BF16 weights before overhead | Too close to total RAM for a first direct-load target; likely failure or severe paging | High |

## Detailed Notes

### `sshleifer/tiny-gpt2`

This model is only a plumbing check. The completed tiny baseline run already
validated model loading, tokenization, generation, timing, memory sampling, and
JSON result writing. It should not be used as evidence that the final assignment
model fits locally.

### `Qwen/Qwen2.5-3B-Instruct`

Qwen 3B is the selected main assignment model. It is large enough to stress the
15.8 GB RAM laptop when loaded through a Python/Transformers stack, but not as
unrealistic as a full 7B BF16 model. It also keeps the experiment aligned across
the direct HF baseline and the AirLLM candidate.

### `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M`

The matching GGUF Q4_K_M variant is the optional quantized comparison target.
The model card lists Q4_K_M around 2.1 GB, making it a practical way to show
how quantization changes the local inference memory profile. It should only be
used after explicit approval to install/use a GGUF backend and download the
model.

### `microsoft/Phi-3-mini-4k-instruct`

Phi-3 Mini remains a backup HF candidate because it is much larger than
`sshleifer/tiny-gpt2` and still within the 3B-4B stress-test range. Its expected
FP16/BF16 weight memory is large for a 15.8 GB RAM machine. A successful run
would need careful generation settings and memory logging; a failure or
paging-heavy run would still be useful evidence for the report.

### `Qwen/Qwen2.5-7B-Instruct`

The 7B Qwen candidate is deferred rather than selected. A full BF16 direct load
is likely too close to or above practical RAM limits once runtime overhead is
included. It is useful to mention as a rejected model-size boundary in the final
report, but it should not be the first large download.

### `Qwen/Qwen2.5-7B-Instruct-GGUF`

The 7B GGUF path is also deferred. A Q4 7B model might technically fit in RAM,
but the assignment asks for a model that is large enough without becoming an
unfocused endurance test. Qwen 3B gives a cleaner, safer experiment for this
hardware.

## Current Decision

Proceed in stages:

1. Keep `sshleifer/tiny-gpt2` as completed pipeline validation evidence.
2. Use `Qwen/Qwen2.5-3B-Instruct` as the main direct HF baseline and AirLLM
   candidate only after explicit approval to download it.
3. Use `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` as the optional quantized local
   comparison only after explicit approval to install/use a GGUF backend and
   download the selected variant.
4. Keep `microsoft/Phi-3-mini-4k-instruct` as a backup model.
5. Defer Qwen 7B models as too risky for the first experiment on this hardware.

