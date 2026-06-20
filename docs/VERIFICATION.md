# Verification

## Purpose

This note records the final non-expensive verification pass before submission.
It avoids rerunning large model downloads or expensive inference jobs.

## Commands Run

| Check | Command | Status |
| --- | --- | --- |
| Python syntax/import compilation | `uv run python -m compileall experiments src` | Passed |
| Result table generation | `uv run python experiments/summarize_results.py` | Passed |
| Figure regeneration | `uv run python experiments/make_figures.py` | Passed |
| Economic artifact regeneration | `uv run python experiments/run_economics.py` | Passed |
| Unit tests | `uv run python -m pytest` | Not available; `pytest` is not installed and no `tests/` directory exists |
| Linting | `uv run ruff check experiments src` | Not available; `ruff` is not installed in the environment |
| Secret scan | `rg -n "api[_-]?key|hf_token|huggingface.*token|OPENAI_API_KEY|ANTHROPIC|sk-|hf_[A-Za-z0-9]" .` | No committed secrets found; hits are config field names, package names, and documentation references |

## Deliverable Check

| Deliverable | Evidence |
| --- | --- |
| Main report | `README.md` |
| Planning docs | `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, `docs/PRD_benchmarking.md` |
| Hardware capture | `results/hardware.json` |
| Baseline results | `results/baseline_tiny_gpt2.json`, `results/baseline_qwen_qwen2_5_3b_instruct.json` |
| AirLLM results | `results/airllm_qwen_qwen2_5_3b_instruct.json`, `results/airllm_phi3_mini_instruct.json` |
| GGUF result | `results/gguf_qwen2_5_3b_instruct_q4_k_m.json` |
| Economic results | `results/economics_summary.json`, `results/economics_curve.csv` |
| Figures | `figures/*.svg` |
| Reproducible scripts | `experiments/*.py` |
| Source package | `src/airllm_benchmark/` |

## Notes

Screenshots are not included because the submission evidence is captured as raw
JSON result files, generated SVG figures, and reproducible commands. The README
embeds the figures directly.

The project intentionally does not rerun expensive model downloads during final
verification. Existing committed result files are the source of truth for the
final report.
