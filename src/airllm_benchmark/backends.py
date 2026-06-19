"""Backend availability checks that do not download model weights."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackendCheckReport:
    """Serializable report of locally available LLM backends."""

    collected_at_utc: str
    platform: dict[str, Any]
    python: dict[str, Any]
    commands: dict[str, dict[str, Any]]
    python_imports: dict[str, dict[str, Any]]


def collect_backend_check() -> BackendCheckReport:
    """Collect backend availability without downloading or loading model weights."""

    return BackendCheckReport(
        collected_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=_platform_info(),
        python=_python_info(),
        commands={
            "ollama": _command_check("ollama", ["ollama", "--version"]),
            "llama_cpp_cli": _command_check("llama-cli", ["llama-cli", "--version"]),
            "llama_cpp_server": _command_check("llama-server", ["llama-server", "--version"]),
            "nvidia_smi": _command_check("nvidia-smi", ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        },
        python_imports={
            "torch": _python_import_check("torch"),
            "transformers": _python_import_check("transformers"),
            "airllm": _python_import_check("airllm"),
            "psutil": _python_import_check("psutil"),
        },
    )


def write_backend_check(report: BackendCheckReport, output_path: Path) -> None:
    """Write a backend check report as pretty JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def _platform_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _python_info() -> dict[str, Any]:
    return {
        "version": sys.version,
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def _command_check(command_name: str, version_command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command_name)
    result: dict[str, Any] = {
        "available": executable is not None,
        "executable": executable,
        "version_command": version_command,
        "returncode": None,
        "stdout": None,
        "stderr": None,
    }
    if executable is None:
        return result

    completed = _run(version_command)
    if completed is None:
        result["stderr"] = "Command failed to start or timed out."
        return result

    result.update(
        {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip() or None,
            "stderr": completed.stderr.strip() or None,
        }
    )
    return result


def _python_import_check(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    result: dict[str, Any] = {
        "available": spec is not None,
        "origin": spec.origin if spec else None,
        "version": None,
        "error": None,
    }
    if spec is None:
        return result

    try:
        module = __import__(module_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        result["available"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["version"] = getattr(module, "__version__", None)
    return result


def _run(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, capture_output=True, check=False, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

