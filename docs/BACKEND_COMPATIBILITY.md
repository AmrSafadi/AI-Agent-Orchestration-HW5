# Backend Compatibility Plan

## Purpose

This document checks backend compatibility before implementing the baseline runner or downloading model weights. The assignment needs a direct baseline, an AirLLM run, and quantization/performance analysis. The safest implementation path is to avoid committing to a large download until the backend/model-format match is clear.

## Sources Checked

- AirLLM GitHub README: https://github.com/lyogavin/airllm
- Ollama importing models documentation: https://ollama.readthedocs.io/en/import/
- Hugging Face model card for `microsoft/Phi-3-mini-4k-instruct`: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Hugging Face model card for `Qwen/Qwen2.5-7B-Instruct-GGUF`: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
- Hugging Face model card for `Qwen/Qwen2.5-7B-Instruct`: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## Current Candidate Models

| Role | Candidate | Format / ecosystem | Status |
| --- | --- | --- | --- |
| Smoke test | `microsoft/Phi-3-mini-4k-instruct` | Hugging Face Transformers, SafeTensors | Good candidate for validating the Python benchmark pipeline |
| Main stress test for local baseline | `Qwen/Qwen2.5-7B-Instruct-GGUF` | GGUF, llama.cpp/Ollama | Good candidate for quantized local baseline stress testing |
| Main stress test for AirLLM | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face Transformers, SafeTensors, BF16 | Better candidate for AirLLM compatibility than GGUF |

## Backend Compatibility Matrix

| Backend | `microsoft/Phi-3-mini-4k-instruct` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | `Qwen/Qwen2.5-7B-Instruct` | Notes |
| --- | --- | --- | --- | --- |
| Hugging Face Transformers | Supported by model card examples | Not the natural path; GGUF is not the standard Transformers format | Supported by tags and model card | Useful for smoke test and possible direct full-precision stress baseline |
| Ollama | Possible through a quantized/GGUF variant, not this exact HF repo as the primary path | Explicitly supported by the model card with `ollama run hf.co/...:Q4_K_M` | Possible import path for SafeTensors models, but heavier and less convenient | Strong option for GGUF baseline |
| llama.cpp | Possible through GGUF conversion or existing quantized variants | Explicitly supported by the model card with `llama-cli` / `llama-server` examples | Possible after conversion to GGUF | Strong option for quantized local inference |
| AirLLM | Unclear; AirLLM examples focus on large model families and `AutoModel`, not Phi-3 specifically | Risky; AirLLM documentation does not present GGUF as its loading path | More plausible; AirLLM documents Qwen support through `AutoModel.from_pretrained(...)` | Main reason not to rely on GGUF for AirLLM |

## Key Findings

### AirLLM

AirLLM's documented workflow initializes models with:

```python
from airllm import AutoModel
model = AutoModel.from_pretrained("...")
```

The README says a Hugging Face repo ID or local model path can be used. It also documents `layer_shards_saving_path`, `compression`, `profiling_mode`, `hf_token`, and `delete_original` configuration options.

Important AirLLM implications:

- AirLLM supports Qwen-style models through `AutoModel`.
- AirLLM documents SafeTensors support.
- AirLLM documents 4-bit and 8-bit compression through the `compression` argument.
- AirLLM splits the original model into layer-wise shards, which can consume significant disk space.
- The current AirLLM docs do not show GGUF as the normal loading path.

Decision:

- Treat `Qwen/Qwen2.5-7B-Instruct-GGUF` as an Ollama/llama.cpp baseline candidate.
- Treat `Qwen/Qwen2.5-7B-Instruct` as the safer AirLLM candidate.
- Do not download either 7B model until a tiny import/availability check confirms the installed backend path.

### Ollama

Ollama supports importing GGUF files and can also import some SafeTensors models. The official importing documentation includes a `FROM /path/to/file.gguf` pattern for GGUF and a `FROM /path/to/safetensors/directory` pattern for SafeTensors.

The `Qwen/Qwen2.5-7B-Instruct-GGUF` model card explicitly provides an Ollama command:

```text
ollama run hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
```

Decision:

- Ollama is the cleanest baseline path for the GGUF stress-test model.
- Because this command downloads model weights, it must not be run yet.

### Hugging Face Transformers

`microsoft/Phi-3-mini-4k-instruct` is tagged for Transformers and SafeTensors and has model-card examples using the Transformers pipeline. The model card describes Phi-3 Mini 4K as a 3.8B-parameter model.

`Qwen/Qwen2.5-7B-Instruct` is tagged for Transformers and SafeTensors. Hugging Face reports the tensor type as BF16 and the model size as 8B params.

Decision:

- Transformers is a good smoke-test path for `microsoft/Phi-3-mini-4k-instruct`.
- Transformers may also be useful for a direct full-precision Qwen baseline, but this will likely be memory-heavy on this machine.

## Recommended Experiment Path

Use staged backend checks before any large model download:

1. Check whether local commands are available:
   - `ollama --version`
   - Python import check for `transformers`, `torch`, and `airllm`
2. Decide dependency installation only after seeing what is already installed.
3. Use `microsoft/Phi-3-mini-4k-instruct` for the first smoke test if Transformers is selected.
4. Use `Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` for the Ollama/llama.cpp quantized baseline if Ollama is available.
5. Use `Qwen/Qwen2.5-7B-Instruct` rather than the GGUF repo for AirLLM, unless AirLLM GGUF support is explicitly verified later.

## Current Decision

Do not implement the baseline runner yet.

Do not download model weights yet.

Next implementation checkpoint:

- Add a lightweight environment/backend availability check that does not download weights.
- Then decide the first executable baseline path and exact dependencies.

