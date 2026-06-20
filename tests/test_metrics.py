from airllm_benchmark.metrics import compact_notes, error_text, safe_round


def test_safe_round_preserves_none_and_rounds_values() -> None:
    assert safe_round(None) is None
    assert safe_round(1.23456) == 1.2346
    assert safe_round(1.23456, digits=2) == 1.23


def test_compact_notes_skips_empty_parts() -> None:
    assert compact_notes(["one", None, "", "two"]) == "one two"


def test_error_text_includes_exception_type() -> None:
    assert error_text(ValueError("bad value")) == "ValueError: bad value"
