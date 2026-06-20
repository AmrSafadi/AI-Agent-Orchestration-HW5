import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_results", ROOT / "experiments" / "summarize_results.py"
)
assert SPEC is not None
summarize_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summarize_results)


def test_cell_escapes_markdown_pipes() -> None:
    assert summarize_results._cell("a|b") == "a\\|b"


def test_evidence_prefers_completed_token_count() -> None:
    row = {"status": "success", "output_tokens": 32, "error": None}

    assert summarize_results._evidence(row) == "Completed 32 output tokens"


def test_evidence_summarizes_error_sentence() -> None:
    row = {"status": "failed", "error": "IndexError: list index out of range. details"}

    assert summarize_results._evidence(row) == "IndexError: list index out of range"
