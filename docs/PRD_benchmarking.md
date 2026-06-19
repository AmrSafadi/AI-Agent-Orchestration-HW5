# Benchmarking Mechanism PRD

## Purpose

The benchmarking mechanism provides consistent measurement across direct baseline inference and AirLLM inference. It exists so the final report compares execution modes using the same prompt, generation settings, result schema, and analysis logic.

## Inputs

- Model identifier.
- Backend name, such as `baseline`, `ollama`, `transformers`, or `airllm`.
- Quantization setting, if applicable.
- Prompt text.
- Maximum output token count.
- Temperature and other generation parameters.
- Cache or shard path for AirLLM.
- Optional timeout.

## Outputs

Each benchmark run should produce one structured result record:

```json
{
  "run_id": "example-run-id",
  "backend": "airllm",
  "model": "model/name",
  "quantization": "q4",
  "status": "success",
  "prompt": "fixed prompt text",
  "input_tokens": 128,
  "output_tokens": 64,
  "ttft_seconds": 12.3,
  "tpot_seconds": 1.4,
  "tokens_per_second": 0.71,
  "total_runtime_seconds": 101.2,
  "peak_ram_mb": 24576,
  "peak_vram_mb": 0,
  "output_sample": "short generated output sample",
  "error": null,
  "notes": "optional observations"
}
```

If a run fails, the record should still be saved with:

- `status` set to `failed`, `timeout`, or `too_slow`.
- `error` containing the relevant message.
- Partial timing and memory data when available.

## Required Metrics

- Time to first token, `TTFT`.
- Time per output token, `TPOT`, or inter-token latency.
- Throughput in tokens per second.
- Total runtime.
- Peak RAM usage.
- Peak VRAM usage if GPU information is available.
- Input and output token counts.
- Qualitative output notes.

## Measurement Rules

- Use the same prompt for comparable baseline and AirLLM runs.
- Use the same maximum output token count for comparable runs.
- Record failed attempts rather than deleting them.
- Use low token limits for initial smoke tests.
- Separate model download/setup time from inference timing where possible.
- Record hardware and environment details before benchmark interpretation.

## Acceptance Criteria

- Baseline and AirLLM results can be loaded into one summary table.
- Failed runs are represented clearly enough to support technical analysis.
- Figures can be generated directly from saved result files.
- The final README can explain every reported number using saved raw data.

## Edge Cases

- Model load fails before generation starts.
- First token is never produced.
- Token counting is unavailable for a backend.
- GPU metrics are unavailable because no supported GPU exists.
- A run is manually stopped after becoming unusably slow.
- AirLLM creates large shard files and fills the selected disk.

## Notes for Implementation

- Prefer a shared result schema for all runners.
- Keep backend-specific code isolated from metric aggregation.
- Store timestamps and command/config metadata to make runs traceable.
- Avoid embedding secrets or Hugging Face tokens in result files.

