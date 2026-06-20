# Result Summary

## Reproducible Table Command

This table is derived only from existing JSON result files. It does not rerun
model downloads, model loading, or generation.

```powershell
uv run python experiments/summarize_results.py
```

## Benchmark Summary

| Run | Backend | Model | Quantization | Status | Runtime (s) | Tokens/s | Peak RAM (MB) | Evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Tiny Transformers smoke test | transformers | sshleifer/tiny-gpt2 | n/a | success | 2.8321 | 112.6696 | 322.13 | Completed 8 output tokens |
| Direct Qwen Transformers | transformers | Qwen/Qwen2.5-3B-Instruct | n/a | timeout | 900 | n/a | n/a | Timed out after 900 seconds. |
| AirLLM Qwen | airllm | Qwen/Qwen2.5-3B-Instruct | n/a | failed | n/a | n/a | n/a | IndexError: list index out of range |
| AirLLM Phi-3 backup | airllm | microsoft/Phi-3-mini-4k-instruct | n/a | failed | n/a | n/a | n/a | NotImplementedError: The model type phi3 is not yet supported to be used with BetterTransformer |
| Ollama GGUF Q4 | ollama | hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M | Q4_K_M | success | 13.4804 | 4.4649 | 2347.17 | Completed 32 output tokens |

## Cross-Run Interpretation

The tiny Transformers run is a pipeline validation result. It proves that the
benchmark code can load a model, generate text, sample memory, and write JSON,
but it is too small to represent the assignment workload.

The direct Qwen Transformers run is the negative full-precision baseline. It
timed out after 900 seconds on the CPU/RAM-only laptop, which makes the direct
BF16 Hugging Face path impractical for this hardware.

The AirLLM Qwen run shows that AirLLM was installed and exercised, and that the
Windows cache/symlink issue was mitigated. The run then failed with a Qwen model
layout issue after partial sharding, so it is deployment-compatibility evidence
rather than a completed performance benchmark.

The AirLLM Phi-3 backup run confirms that the failure was not only a Qwen
download/cache problem. Phi-3 reached a different backend limitation:
BetterTransformer does not support model type `phi3` in this stack.

The Ollama GGUF Q4 run is the successful local LLM result. The same fixed prompt
completed in 13.4804 seconds with 6.3538 seconds TTFT, 4.4649 output tokens per
second, and about 2.3 GB peak RAM across the local Ollama runtime processes. For
this machine, quantized GGUF format was the practical path where direct
Transformers and the tested AirLLM combinations were not.

## Qualitative Comparison

| Run | Output quality / behavior | Report use |
| --- | --- | --- |
| Tiny Transformers smoke test | Completed quickly, but repeated the word `stairs`; useful only as a plumbing check. | Validates benchmark code, not model capability. |
| Direct Qwen Transformers | Produced no completed output because the worker timed out after 900 seconds. | Negative full-precision baseline. |
| AirLLM Qwen | Produced no generation; failed after partial sharding with `IndexError`. | AirLLM/model-layout compatibility evidence. |
| AirLLM Phi-3 backup | Produced no generation; failed because BetterTransformer does not support `phi3`. | Backup-model dependency compatibility evidence. |
| Ollama GGUF Q4 | Produced a relevant explanation beginning with the requested prefill/decode topic. | Best successful local inference result. |

## Figures

The Phase 7 figures are generated from the same saved JSON files:

```powershell
uv run python experiments/make_figures.py
```

- `figures/run_status_summary.svg`
- `figures/throughput_comparison.svg`
- `figures/decode_latency_comparison.svg`
- `figures/memory_comparison.svg`
