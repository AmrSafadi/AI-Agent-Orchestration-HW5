# AirLLM Experiment Plan

## Purpose

This plan defines the AirLLM experiment before running AirLLM model execution.
The direct Transformers baseline for
`Qwen/Qwen2.5-3B-Instruct` timed out after 900 seconds, so AirLLM is now the
main optimization intervention to test.

AirLLM has since been installed in the project environment. The install check is
recorded in `results/airllm_install_check.json`.

## Baseline To Improve

| Field | Value |
| --- | --- |
| Baseline model | `Qwen/Qwen2.5-3B-Instruct` |
| Baseline backend | Hugging Face Transformers |
| Baseline status | `timeout` |
| Timeout | 900 seconds |
| Raw result | `results/baseline_qwen_qwen2_5_3b_instruct.json` |

The baseline did not produce a completed generation. The AirLLM run should use
the same model, prompt, and generation limit so the comparison remains fair.

## Planned AirLLM Target

| Setting | Planned value |
| --- | --- |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Prompt | `Explain the difference between prefill and decode in local LLM inference.` |
| Max new tokens | 32 |
| Temperature | 0.2 |
| Layer/cache directory | `airllm_cache/qwen2_5_3b_instruct` |
| Initial compression | `null` / backend default |
| Optional compression | `4bit` or `8bit` only if supported in the installed AirLLM version |
| Timeout | 1800 seconds |
| Planned result file | `results/airllm_qwen_qwen2_5_3b_instruct.json` |

## Installed Dependency Notes

AirLLM installation required compatibility pins:

- `airllm==2.11.0`
- `optimum<2`, because AirLLM imports `optimum.bettertransformer`.
- `transformers>=4.41.2,<4.49`, because `optimum.bettertransformer` rejects
  newer Transformers versions.
- `sentencepiece`, because AirLLM imports the Baichuan tokenizer during package
  initialization.

The exact command `python -c "import airllm; print(airllm.__version__)"` is not a
valid version check for this package because `airllm` does not expose
`__version__`. The verified metadata version is `2.11.0`.

## Experiment Question

The core question is not whether AirLLM is fast. The question is whether AirLLM
changes the resource allocation enough to make the selected model more feasible
than the direct Transformers baseline.

The report should compare:

- Whether the run completes, fails, or times out.
- Total runtime and observed latency.
- Peak RAM where measurable.
- Output token count and output sample if generation completes.
- Disk/cache behavior during layer sharding and inference.

## Lecture Concepts To Connect

AirLLM should be analyzed through the lecture concepts:

- Virtual memory and paging: AirLLM-like layer movement is conceptually similar
  to keeping only part of a large working set active at a time.
- `mmap` and disk I/O: feasibility may improve by shifting pressure from RAM to
  storage, but each token can become slower because layers must be moved through
  memory/disk repeatedly.
- Prefill versus Decode: direct baseline timed out before useful token metrics,
  so AirLLM may reveal whether startup/prefill or repeated decode movement is
  the dominant cost.
- Memory-bound behavior: this laptop has limited RAM and no practical CUDA VRAM,
  so memory movement is expected to dominate once the model is larger than the
  comfortable local working set.
- Quantization: optional AirLLM compression should be treated as a separate
  intervention because it changes the memory/latency/quality tradeoff.

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| AirLLM install fails on Windows/Python 3.12 | Blocks AirLLM execution | Preserve install logs and document as compatibility evidence |
| AirLLM does not support this exact Qwen model/version | Blocks direct comparison | Try documented `AutoModel` path first; keep Phi-3 as backup only if needed |
| Layer sharding takes a long time | Long setup phase | Use explicit cache directory and timeout; document setup time separately |
| Disk I/O dominates inference | Very slow generation | Treat as expected AirLLM tradeoff and report it clearly |
| Compression argument differs by version | Runtime error | Start with default/no compression, then test supported values only |
| Cache consumes significant disk | Storage pressure | Use `airllm_cache/qwen2_5_3b_instruct` and keep it git-ignored |

## Next Implementation Checkpoint

After the first AirLLM failure is recorded:

1. Configure a Windows-safe Hugging Face cache/download strategy that avoids
   requiring symlink privileges.
2. Retry the same AirLLM prompt/settings.
3. Compare the AirLLM result with the direct Transformers timeout baseline.
