# Baseline Results

## Purpose

This note records direct baseline outcomes before introducing AirLLM or
quantized GGUF execution. Failed, slow, or timed-out runs are valid evidence for
the assignment because they show the practical limits of local LLM execution on
the available hardware.

## Direct Transformers Baseline: Qwen 3B

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Backend | Hugging Face Transformers |
| Prompt | `Explain the difference between prefill and decode in local LLM inference.` |
| Max new tokens | 32 |
| Temperature | 0.2 |
| Result file | `results/baseline_qwen_qwen2_5_3b_instruct.json` |
| Status | `timeout` |
| Timeout | 900 seconds |

The direct Qwen 3B Transformers baseline did not complete within the configured
900-second timeout. The worker was terminated and the result was saved as a
machine-readable JSON file.

Because the run timed out before a completed generation result, the benchmark
does not yet include input token count, output token count, TTFT, TPOT, tokens
per second, or peak RAM for this model. The result is still useful evidence:
direct loading/execution of the selected 3B BF16 model is not comfortable on the
current laptop setup.

## Interpretation

This baseline supports the hardware analysis:

- The integrated Intel GPU is not a practical CUDA inference target.
- The CPU/RAM path is slow enough that direct Transformers execution is not a
  comfortable local deployment mode for this model.
- The next experiment should test whether AirLLM changes feasibility by trading
  memory pressure for disk I/O and latency.
- A later quantized GGUF run can test whether Q4 quantization provides a more
  practical local baseline than BF16 Transformers.

