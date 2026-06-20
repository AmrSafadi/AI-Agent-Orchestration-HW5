import json
import subprocess
from types import SimpleNamespace

import experiments.check_backends as check_backends
import experiments.collect_hardware as collect_hardware
import src.airllm_benchmark.backends as backends
import src.airllm_benchmark.hardware as hardware
from airllm_benchmark.runners.ollama_memory import OllamaProcessMemorySampler


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(["cmd"], returncode, stdout, stderr)


def test_backend_command_and_import_checks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(backends.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(backends, "_run", lambda command: completed("version\n"))

    command = backends._command_check("tool", ["tool", "--version"])
    assert command["available"] is True
    assert command["stdout"] == "version"

    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda name: None)
    missing = backends._python_import_check("missing")
    assert missing["available"] is False

    report = backends.collect_backend_check()
    output = tmp_path / "backend.json"
    backends.write_backend_check(report, output)
    assert json.loads(output.read_text(encoding="utf-8"))["commands"]["ollama"]["available"]


def test_backend_ollama_windows_fallback(monkeypatch, tmp_path) -> None:
    ollama_dir = tmp_path / "Programs" / "Ollama"
    ollama_dir.mkdir(parents=True)
    exe = ollama_dir / "ollama.exe"
    exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(backends.shutil, "which", lambda _name: None)
    monkeypatch.setattr(backends.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(backends, "_run", lambda command: completed("ollama 1"))

    result = backends._ollama_command_check()
    assert result["available"] is True
    assert result["executable"].endswith("ollama.exe")


def test_hardware_helpers_and_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.platform, "release", lambda: "r")
    monkeypatch.setattr(hardware.platform, "version", lambda: "v")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "cpu")
    monkeypatch.setattr(hardware.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        hardware.shutil,
        "disk_usage",
        lambda _root: SimpleNamespace(total=10, used=4, free=6),
    )
    monkeypatch.setattr(hardware, "_run", lambda command: None)

    report = hardware.collect_hardware_report(tmp_path)
    output = tmp_path / "hardware.json"
    hardware.write_hardware_report(report, output)

    assert report.cpu["logical_cores"] == 8
    assert report.memory["total_bytes"] is None
    assert report.gpu == []
    assert json.loads(output.read_text(encoding="utf-8"))["platform"]["system"] == "Linux"


def test_hardware_gpu_parsers(monkeypatch) -> None:
    monkeypatch.setattr(
        hardware,
        "_run",
        lambda command: completed("GPU A, 4096, 555\nbad\nGPU B, nope, 777\n"),
    )
    assert hardware._nvidia_smi_gpus()[0]["vram_total_mb"] == 4096
    assert hardware._nvidia_smi_gpus()[1]["vram_total_mb"] is None

    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hardware,
        "_run",
        lambda command: completed(
            json.dumps({"Name": "Intel", "AdapterRAM": 1024, "DriverVersion": "1"})
        ),
    )
    controllers = hardware._windows_video_controllers()
    assert controllers[0]["name"] == "Intel"
    assert hardware._safe_int("bad") is None
    assert hardware._bytes_to_gb(1024**3) == 1.0


def test_experiment_entrypoints_write_reports(tmp_path, monkeypatch, capsys) -> None:
    backend_report = backends.BackendCheckReport("now", {}, {}, {}, {})
    hardware_report = hardware.HardwareReport("now", {}, {}, {}, {}, [], [])

    monkeypatch.setattr(check_backends, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(check_backends, "collect_backend_check", lambda: backend_report)
    monkeypatch.setattr(check_backends, "write_backend_check", backends.write_backend_check)
    check_backends.main()
    assert (tmp_path / "results" / "backend_check.json").exists()
    assert "Saved backend check report" in capsys.readouterr().out

    monkeypatch.setattr(collect_hardware, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collect_hardware, "collect_hardware_report", lambda _root: hardware_report)
    monkeypatch.setattr(collect_hardware, "write_hardware_report", hardware.write_hardware_report)
    collect_hardware.main()
    assert (tmp_path / "results" / "hardware.json").exists()
    assert "Saved hardware report" in capsys.readouterr().out


class FakeOllamaProcess:
    def __init__(self, name: str, rss: int) -> None:
        self.info = {"name": name}
        self._rss = rss

    def memory_info(self):
        return SimpleNamespace(rss=self._rss)


class PsutilErrorProcess(FakeOllamaProcess):
    def memory_info(self):
        raise __import__("psutil").Error()


def test_ollama_memory_sampler_records_matching_processes(monkeypatch) -> None:
    processes = [
        FakeOllamaProcess("ollama.exe", 1024**2),
        FakeOllamaProcess("llama-server", 2 * 1024**2),
        FakeOllamaProcess("python", 100 * 1024**2),
        PsutilErrorProcess("ollama", 0),
    ]
    monkeypatch.setattr(
        "airllm_benchmark.runners.ollama_memory.psutil.process_iter",
        lambda _attrs: processes,
    )

    sampler = OllamaProcessMemorySampler(interval_seconds=0.001)
    assert sampler.peak_ram_mb is None
    sampler._record_current_rss()
    assert sampler.peak_ram_mb == 3.0

    with OllamaProcessMemorySampler(interval_seconds=0.001) as running:
        pass
    assert running.peak_ram_mb == 3.0
