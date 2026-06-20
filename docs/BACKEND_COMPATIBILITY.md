# Backend Compatibility Plan

## Purpose

This document records backend compatibility checks and the final backend choices
used by the experiment. The assignment needs a direct baseline, an AirLLM run,
and quantization/performance analysis. The selected strategy uses one main model
family where possible, then adds a quantized GGUF comparison through Ollama.

## Sources Checked

- AirLLM GitHub README: https://github.com/lyogavin/airllm
- Ollama importing models documentation: https://ollama.readthedocs.io/en/import/
- Hugging Face model card for `Qwen/Qwen2.5-3B-Instruct`: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Hugging Face model card for `Qwen/Qwen2.5-3B-Instruct-GGUF`: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
- Hugging Face model card for `microsoft/Phi-3-mini-4k-instruct`: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Hugging Face model card for `Qwen/Qwen2.5-7B-Instruct`: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## Current Candidate Models

| Role | Candidate | Format / ecosystem | Status |
| --- | --- | --- | --- |
| Completed pipeline validation | `sshleifer/tiny-gpt2` | Hugging Face Transformers | Already ran successfully |
| Main HF/AirLLM candidate | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face Transformers, SafeTensors, BF16 | Selected main assignment model |
| Quantized local comparison | `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` | GGUF, Ollama | Completed successfully through Ollama |
| Backup HF candidate | `microsoft/Phi-3-mini-4k-instruct` | Hugging Face Transformers, SafeTensors | Backup if Qwen 3B has compatibility problems |
| Deferred too-large candidate | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face Transformers, SafeTensors, BF16 | Not selected for first download |

## Backend Compatibility Matrix

| Backend | `Qwen/Qwen2.5-3B-Instruct` | `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` | `microsoft/Phi-3-mini-4k-instruct` | Notes |
| --- | --- | --- | --- | --- |
| Hugging Face Transformers | Supported by model card examples | Not the natural path; GGUF is not the standard Transformers format | Supported by model card examples | Direct baseline path |
| Ollama | Possible through SafeTensors import, but not preferred | Model card provides `ollama run hf.co/...:Q4_K_M` | Possible through quantized variants, not primary path | Strong option for GGUF comparison |
| llama.cpp | Possible after conversion or via GGUF variant | Model card provides llama.cpp examples | Possible through GGUF conversion or variants | Strong option for quantized local inference |
| AirLLM | Plausible because AirLLM documents Qwen support through `AutoModel.from_pretrained(...)` | Risky; AirLLM docs do not present GGUF as the loading path | Unclear; AirLLM examples focus on other model families | Main AirLLM path is HF/SafeTensors Qwen 3B |

## Key Findings

### AirLLM

AirLLM's documented workflow initializes models with:

```python
from airllm import AutoModel
model = AutoModel.from_pretrained("...")
```

The README says a Hugging Face repo ID or local model path can be used. It also
documents `layer_shards_saving_path`, `compression`, `profiling_mode`,
`hf_token`, and `delete_original` configuration options.

Important AirLLM implications:

- AirLLM supports Qwen-style models through `AutoModel`.
- AirLLM documents SafeTensors support.
- AirLLM documents 4-bit and 8-bit compression through the `compression`
  argument.
- AirLLM splits the original model into layer-wise shards, which can consume
  significant disk space.
- The current AirLLM docs do not show GGUF as the normal loading path.

Decision:

- Treat `Qwen/Qwen2.5-3B-Instruct` as the main AirLLM candidate.
- Treat `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` as an optional GGUF quantized
  comparison candidate.
- The Qwen 3B AirLLM attempt was run after approval and is preserved as
  compatibility evidence.

### Ollama / llama.cpp

The `Qwen/Qwen2.5-3B-Instruct-GGUF` model card provides backend examples for
llama.cpp and Ollama, including:

```text
ollama run hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M
```

Decision:

- Ollama was used for the GGUF quantized comparison.
- The model was downloaded and benchmarked after approval.

### Hugging Face Transformers

`Qwen/Qwen2.5-3B-Instruct` is tagged for Transformers and SafeTensors. The model
card reports 3.09B parameters and BF16 tensor type.

`microsoft/Phi-3-mini-4k-instruct` is tagged for Transformers and SafeTensors
and has model-card examples using the Transformers pipeline. The model card
describes Phi-3 Mini 4K as a 3.8B-parameter model.

`Qwen/Qwen2.5-7B-Instruct` is tagged for Transformers and SafeTensors, but it is
deferred because direct full-load execution is too close to the machine's total
RAM before runtime overhead.

Decision:

- Transformers is the direct baseline path for `Qwen/Qwen2.5-3B-Instruct`.
- `microsoft/Phi-3-mini-4k-instruct` remains a backup if Qwen 3B fails for
  compatibility reasons.

## Recommended Experiment Path

The final staged experiment path was:

1. Keep `sshleifer/tiny-gpt2` as completed pipeline validation.
2. Prepare the baseline runner for configurable model IDs, prompts, generation
   settings, timeout handling, and JSON result output.
3. Use `Qwen/Qwen2.5-3B-Instruct` for the main direct Transformers baseline
   after explicit approval to download model weights.
4. Use `Qwen/Qwen2.5-3B-Instruct` for AirLLM after explicit approval to install
   AirLLM and create layer shards.
5. Use `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` for the Ollama quantized
   comparison with streaming output to measure TTFT.

## Current Decision

The backend evidence is complete for submission. Direct Transformers and AirLLM
provide negative local-deployment evidence, while Ollama/GGUF provides the
successful quantized local inference path.

