# Prompt Log

## Purpose

This note records the prompt choices used in the benchmark so the experiment is
reproducible and the prompt is not treated as an implicit hidden variable.

## Fixed Benchmark Prompt

```text
Explain the difference between prefill and decode in local LLM inference.
```

## Generation Settings

| Field | Value |
| --- | --- |
| Max new tokens | 32 for the main Qwen/AirLLM/GGUF attempts |
| Temperature | 0.2 for the main Qwen/AirLLM/GGUF attempts |
| Tiny smoke-test max new tokens | 8 |
| Tiny smoke-test prompt | `Local LLM benchmarking checks` |

## Rationale

The main prompt was chosen because it is short, technical, and directly tied to
the lecture concepts required by the assignment. It exercises both prefill and
decode without requiring a long context window, external facts, or subjective
creative writing quality. That makes it suitable for comparing local inference
feasibility across direct Transformers, AirLLM attempts, and quantized GGUF.

The same main prompt and generation limit are reused across the direct Qwen
baseline, the AirLLM Qwen attempt, the AirLLM Phi-3 backup attempt, and the
Ollama GGUF Q4 run. The tiny GPT-2 prompt is deliberately separate because that
run is only a dependency and JSON-output smoke test, not evidence of model
capability.

## Prompt Changes Considered

Longer prompts were avoided because this laptop is CPU/RAM-limited and the
assignment goal is to expose local deployment bottlenecks without turning the
experiment into a long-context endurance test. Open-ended creative prompts were
also avoided because output quality would be harder to judge consistently.
