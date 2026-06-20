import json

import experiments.make_figures as figures
import experiments.run_economics as economics
import experiments.summarize_results as summary


def test_make_figures_helpers_and_main(tmp_path, monkeypatch) -> None:
    rows = [
        ("ok", {"status": "success", "tokens_per_second": 2.5, "output_tokens": 4}),
        ("bad", {"status": "failed", "tokens_per_second": None, "error": "IndexError: x"}),
    ]
    metric_rows = figures._metric_rows(rows, "tokens_per_second")

    assert metric_rows[0]["value"] == 2.5
    assert metric_rows[1]["value"] is None
    assert figures._evidence(rows[0][1]) == "4 output tokens"
    assert figures._evidence(rows[1][1]) == "layout issue"
    assert figures._fmt(1.2300) == "1.23"

    result_paths = []
    for index, (_label, row) in enumerate(rows):
        path = tmp_path / f"result_{index}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        result_paths.append(path)

    monkeypatch.setattr(figures, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(figures, "RESULTS", [("ok", result_paths[0]), ("bad", result_paths[1])])
    figures.main()

    assert (tmp_path / "figures" / "throughput_comparison.svg").exists()
    assert (tmp_path / "figures" / "run_status_summary.svg").exists()


def test_summary_helpers_escape_and_format_values(tmp_path, capsys, monkeypatch) -> None:
    row = {
        "backend": "ollama",
        "model": "a|b",
        "quantization": None,
        "status": "success",
        "total_runtime_seconds": 1.2300,
        "tokens_per_second": 4,
        "peak_ram_mb": None,
        "output_tokens": 3,
    }
    path = tmp_path / "baseline_tiny_gpt2.json"
    path.write_text(json.dumps(row), encoding="utf-8")

    assert summary._cell("a|b") == "a\\|b"
    assert summary._number(1.2300) == "1.23"
    assert summary._evidence(row) == "Completed 3 output tokens"

    monkeypatch.setattr(summary, "RESULTS", [path])
    summary.main()
    output = capsys.readouterr().out

    assert "Tiny Transformers smoke test" in output
    assert "a\\|b" in output


def test_economics_main_and_helpers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(economics, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(economics, "FIGURES", tmp_path / "figures")
    economics.main()

    summary_data = json.loads(
        (tmp_path / "results" / "economics_summary.json").read_text(encoding="utf-8")
    )
    curve_text = (tmp_path / "results" / "economics_curve.csv").read_text(encoding="utf-8")

    assert summary_data["workload"]["output_tokens"] == economics.OUTPUT_TOKENS
    assert summary_data["costs"]["break_even_requests_per_month"] is not None
    assert "requests_per_month" in curve_text
    assert (tmp_path / "figures" / "cost_break_even.svg").exists()
    assert economics._break_even(1.0, 2.0, 1.0) is None
