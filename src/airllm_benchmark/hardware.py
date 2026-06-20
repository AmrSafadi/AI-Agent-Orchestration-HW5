"""Hardware inspection utilities for the local LLM benchmark."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardwareReport:
    """Serializable hardware report for experiment reproducibility."""

    collected_at_utc: str
    platform: dict[str, Any]
    python: dict[str, Any]
    cpu: dict[str, Any]
    memory: dict[str, Any]
    gpu: list[dict[str, Any]]
    disks: list[dict[str, Any]]


def collect_hardware_report(root: Path | None = None) -> HardwareReport:
    """Collect hardware details using standard Python and Windows tools when available."""

    project_root = root or Path.cwd()
    return HardwareReport(
        collected_at_utc=datetime.now(UTC).isoformat(),
        platform=_platform_info(),
        python=_python_info(),
        cpu=_cpu_info(),
        memory=_memory_info(),
        gpu=_gpu_info(),
        disks=_disk_info(project_root),
    )


def write_hardware_report(report: HardwareReport, output_path: Path) -> None:
    """Write a hardware report as pretty JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def _platform_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _python_info() -> dict[str, Any]:
    return {
        "version": sys.version,
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def _cpu_info() -> dict[str, Any]:
    return {
        "processor": platform.processor(),
        "physical_cores": _powershell_int(
            "(Get-CimInstance Win32_Processor | "
            "Measure-Object -Property NumberOfCores -Sum).Sum"
        ),
        "logical_cores": os.cpu_count(),
        "name": _powershell_text("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"),
    }


def _memory_info() -> dict[str, Any]:
    total_bytes = _windows_total_ram_bytes()
    if total_bytes is None:
        total_bytes = _powershell_int(
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
        )
    return {
        "total_bytes": total_bytes,
        "total_gb": _bytes_to_gb(total_bytes),
    }


def _gpu_info() -> list[dict[str, Any]]:
    gpus = _nvidia_smi_gpus()
    if gpus:
        return gpus
    return _windows_video_controllers()


def _disk_info(project_root: Path) -> list[dict[str, Any]]:
    roots = {project_root.anchor or str(project_root.resolve().drive)}
    if platform.system() == "Windows":
        system_drive = os.environ.get("SYSTEMDRIVE")
        if system_drive:
            roots.add(f"{system_drive}\\")

    disks = []
    for root in sorted(roots):
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        disks.append(
            {
                "path": root,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "total_gb": _bytes_to_gb(usage.total),
                "free_gb": _bytes_to_gb(usage.free),
            }
        )
    return disks


def _windows_total_ram_bytes() -> int | None:
    if platform.system() != "Windows":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not success:
        return None
    return int(status.ullTotalPhys)


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    query = "name,memory.total,driver_version"
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    completed = _run(command)
    if completed is None or completed.returncode != 0:
        return []

    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, memory_total_mb, driver_version = parts
        gpus.append(
            {
                "name": name,
                "vram_total_mb": _safe_int(memory_total_mb),
                "driver_version": driver_version,
                "source": "nvidia-smi",
            }
        )
    return gpus


def _windows_video_controllers() -> list[dict[str, Any]]:
    if platform.system() != "Windows":
        return []

    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json"
    )
    completed = _run(["powershell", "-NoProfile", "-Command", command])
    if completed is None or completed.returncode != 0 or not completed.stdout.strip():
        return []

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []

    controllers = data if isinstance(data, list) else [data]
    gpus = []
    for controller in controllers:
        adapter_ram = controller.get("AdapterRAM")
        gpus.append(
            {
                "name": controller.get("Name"),
                "vram_total_bytes": adapter_ram,
                "vram_total_gb": _bytes_to_gb(adapter_ram),
                "driver_version": controller.get("DriverVersion"),
                "source": "Win32_VideoController",
            }
        )
    return gpus


def _powershell_text(command: str) -> str | None:
    completed = _run(["powershell", "-NoProfile", "-Command", command])
    if completed is None or completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    return text or None


def _powershell_int(command: str) -> int | None:
    text = _powershell_text(command)
    return _safe_int(text)


def _run(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, capture_output=True, check=False, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 2)
