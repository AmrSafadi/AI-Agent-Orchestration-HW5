import importlib.util
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ollama_api", ROOT / "src" / "airllm_benchmark" / "runners" / "ollama_api.py"
)
assert SPEC is not None
ollama_api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ollama_api)


def test_read_streaming_response_combines_chunks_and_final_metrics() -> None:
    lines = [
        b'{"response":"hello","done":false}\n',
        b'{"response":" world","done":false}\n',
        b'{"response":"","done":true,"eval_count":2}\n',
    ]

    final, text, ttft = ollama_api.read_streaming_response(lines, time.perf_counter())

    assert final["done"] is True
    assert final["eval_count"] == 2
    assert text == "hello world"
    assert ttft is not None
