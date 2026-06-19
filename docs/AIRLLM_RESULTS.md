# AirLLM Results

## Purpose

This note records AirLLM execution outcomes. Failed setup or runtime attempts
are valid evidence when they expose compatibility, cache, disk, or memory
bottlenecks relevant to local LLM deployment.

## Attempt 1: Qwen 3B AirLLM Symlink Failure

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Backend | AirLLM |
| Prompt | `Explain the difference between prefill and decode in local LLM inference.` |
| Max new tokens | 32 |
| Temperature | 0.2 |
| Layer/cache path | `airllm_cache/qwen2_5_3b_instruct` |
| Historical result | Recorded in commit history before retry |
| Status | `failed` |

The first AirLLM attempt failed before producing a completed benchmark. The raw
result records:

```text
OSError: [WinError 1314] A required privilege is not held by the client
```

The error occurred while Hugging Face cache code attempted to create a symlink
inside:

```text
C:\Users\amrsa\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct
```

## Interpretation

This is a Windows cache/symlink compatibility failure, not an AirLLM performance
result yet. It still matters for the assignment because local/on-prem deployment
depends on operating-system behavior, filesystem permissions, and model-cache
layout, not only Python model code.

The next retry should avoid requiring symlink privileges. Preferred mitigations:

- Configure Hugging Face cache behavior/environment for Windows-safe downloads.
- Prefer a local model/cache path that avoids symlink creation if possible.
- Avoid requiring administrator privileges or Windows Developer Mode unless
  documented as an optional workaround.

## Mitigation Smoke Retry

A short 10-second retry was run after adding a project-local Hugging Face cache
and disabling symlink use inside the AirLLM worker on Windows. The retry reached
metadata/file fetching and layer loading, then timed out intentionally after 10
seconds with peak RAM recorded.

This confirms the symlink privilege failure was bypassed. The next run can use
the normal AirLLM timeout and save the real result file.

## Attempt 2: Qwen 3B AirLLM Local-Cache Retry

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Backend | AirLLM |
| Cache mitigation | Project-local Hugging Face cache with symlink support disabled on Windows |
| Layer/cache path | `airllm_cache/qwen2_5_3b_instruct` |
| Result file | `results/airllm_qwen_qwen2_5_3b_instruct.json` |
| Status | `failed` |
| Error | `IndexError: list index out of range` |

The retry bypassed the Windows symlink privilege error and made real progress:

- Hugging Face metadata and model shard downloads started through the project
  cache.
- AirLLM split and saved the embedding layer, transformer layers 0-35, and the
  model norm layer.
- The run failed before generation with `IndexError: list index out of range`.

This is an AirLLM/model-layout compatibility result rather than a local hardware
performance result. The partial shard output suggests AirLLM could process most
of Qwen 3B's SafeTensors layout, but failed near the final layer/head handling.

## Current Interpretation

The AirLLM path has now exposed two distinct deployment issues:

1. Windows/Hugging Face cache symlink privileges, mitigated by the local cache
   strategy.
2. AirLLM compatibility with the selected Qwen 2.5 3B model layout, currently
   blocking completed generation.

The next engineering decision is whether to patch around the Qwen layout issue,
try the backup HF model, or move to the planned quantized GGUF comparison for
the quantization part of the assignment.
