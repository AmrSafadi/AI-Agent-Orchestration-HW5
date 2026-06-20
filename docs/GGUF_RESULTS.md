# GGUF Quantization Results

## Purpose

This note records the quantized GGUF run used to compare against the direct
Transformers baseline and AirLLM attempts.

## Backend Setup

| Field | Value |
| --- | --- |
| Backend | Ollama |
| Version | `0.30.8` |
| CLI path used in this shell | `C:\Users\amrsa\AppData\Local\Programs\Ollama\ollama.exe` |
| Model | `hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` |
| Local model size reported by Ollama | 2.1 GB |
| Result file | `results/gguf_qwen2_5_3b_instruct_q4_k_m.json` |

Ollama installed successfully through `winget`. The current PowerShell session
did not pick up the new PATH, so commands were verified through the full
installed executable path.

## Benchmark Result

| Metric | Value |
| --- | --- |
| Status | `success` |
| Prompt | `Explain the difference between prefill and decode in local LLM inference.` |
| Max new tokens | 32 |
| Temperature | 0.2 |
| Input tokens | 23 |
| Output tokens | 32 |
| Time to first token | 6.3538 seconds |
| Total runtime | 13.4804 seconds |
| Tokens per second | 4.4649 |
| Time per output token | 0.224 seconds |
| Peak RAM | 2347.17 MB |
| Peak VRAM | null; CPU/local system-memory path |

The run used Ollama's local `/api/generate` endpoint with `stream=true`.
Streaming output made first-token latency measurable, so `ttft_seconds` is
recorded as 6.3538 seconds.

## Interpretation

This is the strongest positive local-inference result so far:

- Direct BF16 Transformers for Qwen 3B timed out after 900 seconds.
- AirLLM made useful progress but failed on model/backend compatibility issues.
- Q4 GGUF through Ollama completed the same prompt successfully and used about
  2.3 GB of RAM for the local runtime processes.

The result supports the main quantization conclusion for the report: on this
CPU/RAM-only laptop, changing model format and quantization level had a larger
practical impact than attempting to run the full Hugging Face model directly.
