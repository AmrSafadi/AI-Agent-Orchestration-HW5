# GGUF Quantization Plan

## Purpose

This plan covers the quantized local-inference comparison for Exercise 05. It
comes after the direct Transformers baseline and AirLLM attempts, and is meant
to test whether a 4-bit GGUF model is more practical on the available laptop
hardware.

The planned backend install, model download, and benchmark run were completed
through Ollama. The final raw result is stored in
`results/gguf_qwen2_5_3b_instruct_q4_k_m.json`.

## Current Evidence

| Path | Model | Status | Meaning |
| --- | --- | --- | --- |
| Tiny Transformers smoke test | `sshleifer/tiny-gpt2` | Succeeded | Benchmark plumbing works |
| Direct Transformers baseline | `Qwen/Qwen2.5-3B-Instruct` | Timed out after 900 seconds | BF16 direct local inference is impractical on this laptop |
| AirLLM selected model | `Qwen/Qwen2.5-3B-Instruct` | Failed after partial sharding | AirLLM hit a Qwen model-layout compatibility issue |
| AirLLM backup model | `microsoft/Phi-3-mini-4k-instruct` | Failed after sharding | AirLLM hit a BetterTransformer `phi3` support limitation |

This is enough AirLLM evidence for the report. The completed GGUF comparison
adds a quantized backend that uses a smaller on-disk and in-memory
representation.

## Recommended Quantized Target

| Field | Value |
| --- | --- |
| Model family | Qwen 2.5 |
| Quantized model | `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` |
| Format | GGUF |
| Quantization | Q4_K_M, about 4-bit mixed quantization |
| Expected model-file size | About 2.1 GB, based on the model-selection note |
| Expected hardware fit | Plausible in system RAM; still CPU-bound and likely slow |

This target keeps the same model family as the main HF/AirLLM candidate while
changing the storage and execution format. That makes the comparison easier to
explain than switching to a completely unrelated model.

## Backend Choice

| Backend | Pros | Cons | Recommendation |
| --- | --- | --- | --- |
| Ollama | Easiest GGUF user path; simple command-line run; can pull from Hugging Face GGUF refs | Less direct Python-level control; service/runtime behavior must be documented | Preferred first quantized attempt |
| llama.cpp CLI | Direct GGUF backend; transparent model arguments; strong fit for benchmarking | Requires installing or building llama.cpp binaries | Good fallback if Ollama is not approved or fails |
| llama-cpp-python | Python integration; easier to wrap in existing benchmark runner | More dependency/build risk on Windows | Defer unless Python integration is required |

The selected implementation path was Ollama, because it minimized new Python
dependency risk and provided a clear local quantized inference story.

## Reproduction Command

The benchmarked GGUF path uses the local Ollama service and streaming output so
TTFT can be measured.

```powershell
ollama pull hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M
uv run python experiments/run_ollama.py --config config/experiment.example.json --stream
```

The benchmark uses the same prompt and generation settings already used by the
baseline and AirLLM attempts:

```text
Prompt: Explain the difference between prefill and decode in local LLM inference.
Max new tokens: 32
Temperature: 0.2
```

## Planned Result File

```text
results/gguf_qwen2_5_3b_instruct_q4_k_m.json
```

The result records:

- Backend name and version.
- Model ID and quantization.
- Prompt and generation settings.
- Load or pull failure, timeout, or successful generation.
- Wall-clock runtime.
- Peak RAM if measurable.
- Output text if generation succeeds.
- Notes about CPU-only execution and any service startup behavior.

## Report Angle

The expected final comparison is:

- Direct BF16 Transformers: too slow or timed out.
- AirLLM: attempted layer-wise loading, but blocked by backend/model
  compatibility.
- Quantized GGUF Q4: the successful local path on this CPU/RAM-only laptop,
  though still slower and lower quality than an external API.

This gives the final report a clear engineering conclusion: local deployment is
not just about parameter count. Format, quantization, runtime backend,
operating-system behavior, and model-architecture support all determine whether
the experiment succeeds.
