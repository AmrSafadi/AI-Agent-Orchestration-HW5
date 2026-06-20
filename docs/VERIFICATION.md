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
| Unit tests | `uv run python -m pytest` | Passed, 14 tests |
| Coverage report | `uv run python -m pytest --cov=src --cov=experiments --cov-report=term-missing` | Passed, 25% line coverage; no fail-under gate because tests focus on deterministic helpers and avoid model execution |
| Linting | `uv run ruff check experiments src tests` | Passed |
| Secret scan | `rg -n "api[_-]?key|hf_token|huggingface.*token|OPENAI_API_KEY|ANTHROPIC|sk-|hf_[A-Za-z0-9]" .` | No committed secrets found; hits are placeholders, config field names, package names, and documentation references |

## Deliverable Check

| Deliverable | Evidence |
| --- | --- |
| Main report | `README.md` |
| Planning docs | `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, `docs/PRD_benchmarking.md` |
| Prompt log | `docs/PROMPT_LOG.md` |
| Environment template | `.env.example` |
| Hardware capture | `results/hardware.json` |
| Baseline results | `results/baseline_tiny_gpt2.json`, `results/baseline_qwen_qwen2_5_3b_instruct.json` |
| AirLLM results | `results/airllm_qwen_qwen2_5_3b_instruct.json`, `results/airllm_phi3_mini_instruct.json` |
| GGUF result | `results/gguf_qwen2_5_3b_instruct_q4_k_m.json` |
| Economic results | `results/economics_summary.json`, `results/economics_curve.csv` |
| Figures | `figures/*.svg` |
| Reproducible scripts | `experiments/*.py` |
| Source package | `src/airllm_benchmark/` |
| Tests | `tests/*.py` |

## Notes

Screenshots are not included because the submission evidence is captured as raw
JSON result files, generated SVG figures, and reproducible commands. The README
embeds the figures directly.

The project intentionally does not rerun expensive model downloads during final
verification. Existing committed result files are the source of truth for the
final report.
