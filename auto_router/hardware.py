"""Detect this machine's memory and GPU, and pick the local models that fit.

    python -m auto_router.hardware            # human summary
    python -m auto_router.hardware --json     # machine-readable, for installers
    python -m auto_router.hardware --json --with-bonsai

Two questions are answered from ``auto_router/data/local_models.json`` (sizes
read from the Hugging Face API, JevBench ranks and licences, measured local
runs, each with its source and date):

* does Bonsai 2 (Ternary-Bonsai-2-27B GGUF) fit, and which packing;
* which open, commercially licensed Jev-class decision model to run locally.

Detection only reads: ``nvidia-smi``, ``rocm-smi``, ``system_profiler`` /
``sysctl`` on macOS, PowerShell CIM on Windows, ``/proc/meminfo`` on Linux.
Every probe is optional and bounded by a short timeout; a missing tool is not
an error. Memory thresholds in the data file are assumptions (file size plus a
margin), and the output says so.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

DATA = Path(__file__).resolve().parent / "data" / "local_models.json"

Runner = Callable[[list[str]], "str | None"]


@dataclass
class GPU:
    vendor: str            # nvidia | amd | apple | intel | other
    name: str
    vram_gb: float | None  # None when unknown; for Apple, unified memory is used instead
    unified: bool = False


@dataclass
class Hardware:
    os: str                # linux | macos | windows
    wsl: bool
    arch: str
    ram_gb: float | None
    gpus: list[GPU] = field(default_factory=list)
    probes: dict[str, str] = field(default_factory=dict)

    def best_gpu(self) -> GPU | None:
        known = [g for g in self.gpus if g.vram_gb]
        return max(known, key=lambda g: g.vram_gb or 0) if known else None


def load_table(path: Path = DATA) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], timeout: float = 8.0) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _is_wsl(os_name: str) -> bool:
    if os_name != "linux":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _os_name() -> str:
    s = platform.system().lower()
    return {"darwin": "macos"}.get(s, s)


def parse_meminfo(text: str) -> float | None:
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.M)
    return round(int(m.group(1)) / 1024 / 1024, 1) if m else None


def parse_nvidia_smi(text: str) -> list[GPU]:
    """``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits``."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append(GPU("nvidia", parts[0], round(float(parts[1]) / 1024, 1)))
            except ValueError:
                gpus.append(GPU("nvidia", parts[0], None))
    return gpus


def parse_rocm_smi(text: str) -> list[GPU]:
    """``rocm-smi --showmeminfo vram --json``: bytes per card."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    gpus = []
    for card, info in data.items():
        if not isinstance(info, dict):
            continue
        total = next((v for k, v in info.items() if "total" in k.lower() and "used" not in k.lower()), None)
        try:
            gb = round(int(total) / 1024 ** 3, 1) if total is not None else None
        except (TypeError, ValueError):
            gb = None
        gpus.append(GPU("amd", str(card), gb))
    return gpus


def parse_system_profiler(text: str, ram_gb: float | None) -> list[GPU]:
    """``system_profiler SPDisplaysDataType -json``; Apple Silicon shares RAM with the GPU."""
    try:
        items = json.loads(text).get("SPDisplaysDataType") or []
    except (json.JSONDecodeError, AttributeError):
        return []
    gpus = []
    for it in items:
        name = it.get("sppci_model") or it.get("_name") or "gpu"
        vendor_raw = str(it.get("spdisplays_vendor") or it.get("sppci_vendor") or "").lower()
        if "apple" in name.lower() or "apple" in vendor_raw:
            gpus.append(GPU("apple", name, ram_gb, unified=True))
            continue
        vram = str(it.get("spdisplays_vram") or it.get("spdisplays_vram_shared") or "")
        m = re.search(r"(\d+)\s*(GB|MB)", vram)
        gb = (int(m.group(1)) / (1024 if m.group(2) == "MB" else 1)) if m else None
        vendor = "amd" if "amd" in vendor_raw or "radeon" in name.lower() else (
            "nvidia" if "nvidia" in vendor_raw else ("intel" if "intel" in vendor_raw else "other"))
        gpus.append(GPU(vendor, name, gb))
    return gpus


def parse_windows_cim(text: str) -> tuple[float | None, list[GPU]]:
    """JSON from the PowerShell probe in :func:`detect` (RAM bytes + video controllers).

    ``AdapterRAM`` is a 32-bit field and caps at 4 GB, so an NVIDIA card's size
    comes from ``nvidia-smi`` when it is present; this is only the fallback.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, []
    ram = data.get("ram")
    ram_gb = round(int(ram) / 1024 ** 3, 1) if ram else None
    ctrls = data.get("gpus") or []
    if isinstance(ctrls, dict):
        ctrls = [ctrls]
    gpus = []
    for c in ctrls:
        name = str(c.get("Name") or "gpu")
        low = name.lower()
        vendor = ("nvidia" if "nvidia" in low else "amd" if ("amd" in low or "radeon" in low)
                  else "intel" if "intel" in low else "other")
        raw = c.get("AdapterRAM")
        gb = round(int(raw) / 1024 ** 3, 1) if raw else None
        gpus.append(GPU(vendor, name, gb))
    return ram_gb, gpus


WIN_PROBE = ("$r=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;"
             "$g=Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM;"
             "@{ram=$r;gpus=$g} | ConvertTo-Json -Compress -Depth 3")


def detect(run: Runner | None = None, *, os_name: str | None = None,
           meminfo: str | None = None) -> Hardware:
    """Probe the machine. ``run`` and ``meminfo`` exist for tests."""
    run = run or _run
    os_name = os_name or _os_name()
    hw = Hardware(os=os_name, wsl=_is_wsl(os_name), arch=platform.machine().lower(), ram_gb=None)

    if os_name == "linux":
        if meminfo is None:
            try:
                meminfo = Path("/proc/meminfo").read_text()
            except OSError:
                meminfo = ""
        hw.ram_gb = parse_meminfo(meminfo)
        hw.probes["ram"] = "/proc/meminfo"
    elif os_name == "macos":
        out = run(["sysctl", "-n", "hw.memsize"])
        if out and out.strip().isdigit():
            hw.ram_gb = round(int(out.strip()) / 1024 ** 3, 1)
            hw.probes["ram"] = "sysctl hw.memsize"
        sp = run(["system_profiler", "SPDisplaysDataType", "-json"])
        if sp:
            hw.gpus.extend(parse_system_profiler(sp, hw.ram_gb))
            hw.probes["gpu"] = "system_profiler"
    elif os_name == "windows":
        ps = shutil.which("powershell") and "powershell" or "pwsh"
        out = run([ps, "-NoProfile", "-NonInteractive", "-Command", WIN_PROBE])
        if out:
            hw.ram_gb, gpus = parse_windows_cim(out)
            hw.gpus.extend(gpus)
            hw.probes["ram"] = hw.probes["gpu"] = "PowerShell CIM"

    nv = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if nv:
        nvidia = parse_nvidia_smi(nv)
        hw.gpus = [g for g in hw.gpus if g.vendor != "nvidia"] + nvidia
        hw.probes["gpu"] = "nvidia-smi"
    if os_name == "linux":
        rocm = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
        if rocm:
            hw.gpus.extend(parse_rocm_smi(rocm))
            hw.probes["gpu_amd"] = "rocm-smi"
    return hw


# ------------------------------------------------------------------ picking

def _gpu_budget(hw: Hardware, vendors: list[str] | None = None) -> tuple[str, float] | None:
    """(kind, GB usable for model weights) of the best accelerator, or None."""
    best: tuple[str, float] | None = None
    for g in hw.gpus:
        if vendors and g.vendor not in vendors:
            continue
        if g.unified and g.vram_gb:
            # macOS lets the GPU wire roughly two thirds to three quarters of RAM.
            cand = ("unified", round(g.vram_gb * 0.7, 1))
        elif g.vram_gb:
            cand = ("vram", g.vram_gb)
        else:
            continue
        if best is None or cand[1] > best[1]:
            best = cand
    return best


def pick_bonsai(hw: Hardware, table: dict) -> dict:
    b = table["bonsai"]
    files = {f["file"]: f for f in b["files"]}
    ptq, pq2 = files["Ternary-Bonsai-2-27B-PTQ1_0.gguf"], files["Ternary-Bonsai-2-27B-PQ2_0.gguf"]
    base = {"model": b["name"], "repo": b["repo"], "licence": b["licence"], "runtime": b["runtime"],
            "source": b["source"]}
    for g in hw.gpus:
        if g.vendor == "apple" and g.unified and (g.vram_gb or 0) >= pq2["min_unified_gb"]:
            return {**base, "fits": True, "file": pq2["file"], "gb": pq2["gb"], "device": "metal",
                    "reason": f"Apple Silicon with {g.vram_gb} GB unified memory (>= {pq2['min_unified_gb']} GB assumed)"}
    gpu = _gpu_budget(hw, ["nvidia"])
    if gpu and gpu[0] == "vram":
        if gpu[1] >= pq2["min_vram_gb"]:
            return {**base, "fits": True, "file": pq2["file"], "gb": pq2["gb"], "device": "cuda",
                    "reason": f"NVIDIA GPU with {gpu[1]} GB VRAM (>= {pq2['min_vram_gb']} GB assumed for PQ2_0)"}
        if gpu[1] >= ptq["min_vram_gb"]:
            return {**base, "fits": True, "file": ptq["file"], "gb": ptq["gb"], "device": "cuda",
                    "reason": f"NVIDIA GPU with {gpu[1]} GB VRAM (>= {ptq['min_vram_gb']} GB assumed for PTQ1_0)"}
    if (hw.ram_gb or 0) >= ptq["min_ram_gb_cpu"]:
        return {**base, "fits": True, "file": ptq["file"], "gb": ptq["gb"], "device": "cpu",
                "reason": f"no supported GPU with enough memory; {hw.ram_gb} GB RAM runs it on the CPU "
                          f"(slow; CPU speed not measured here)"}
    return {**base, "fits": False, "file": None, "gb": None, "device": None,
            "reason": f"needs an NVIDIA GPU with >= {ptq['min_vram_gb']} GB, Apple Silicon with >= "
                      f"{pq2['min_unified_gb']} GB, or >= {ptq['min_ram_gb_cpu']} GB RAM for CPU"}


def _fits(req: dict, hw: Hardware, gpu_reserved_gb: float, ram_reserved_gb: float) -> str | None:
    """Device the tier fits on, or None."""
    vendors = req.get("gpu_vendor")
    for g in hw.gpus:
        if vendors and g.vendor not in vendors:
            continue
        if g.unified and g.vram_gb and "min_unified_gb" in req:
            if g.vram_gb - gpu_reserved_gb >= req["min_unified_gb"]:
                return "metal"
        elif g.vram_gb and "min_vram_gb" in req and g.vendor in ("nvidia", "amd"):
            if g.vram_gb - gpu_reserved_gb >= req["min_vram_gb"]:
                return "cuda" if g.vendor == "nvidia" else "rocm/vulkan"
    if vendors:          # a GPU-only tier does not fall back to the CPU
        return None
    if "min_ram_gb_cpu" in req and (hw.ram_gb or 0) - ram_reserved_gb >= req["min_ram_gb_cpu"]:
        return "cpu"
    if not req:
        return "browser"
    return None


def pick_jev(hw: Hardware, table: dict, *, bonsai: dict | None = None,
             include_opt_in: bool = False) -> dict:
    """Best tier that fits, after reserving memory for Bonsai if it runs too."""
    gpu_res = ram_res = 0.0
    if bonsai and bonsai.get("fits"):
        if bonsai["device"] == "cpu":
            ram_res = float(bonsai["gb"]) + 2
        else:
            gpu_res = float(bonsai["gb"]) + 1.5
    alternatives: list[dict] = []
    # Accelerators first: a smaller quantisation on a GPU beats a larger one on
    # the CPU for a per-request decision (measured: JevK5 Q4 189 ms on an 8 GB
    # RTX 3070; CPU latency was not measured here).
    for allowed in ({"cuda", "metal", "rocm/vulkan"}, {"cpu", "browser"}):
        for tier in table["jev_tiers"]:
            if not tier.get("commercial_ok"):
                continue
            device = _fits(tier.get("requires") or {}, hw, gpu_res, ram_res)
            if device not in allowed:
                continue
            entry = {"id": tier["id"], "display": tier["display"], "device": device,
                     "repo": tier["repo"], "file": tier["file"], "gb": tier["gb"],
                     "licence": tier["licence"], "jevbench": tier["jevbench"],
                     "serving": tier["serving"], "openai_compatible": tier["openai_compatible"],
                     "measured": tier["measured"], "source": tier["source"]}
            if not tier.get("default_pick") and not include_opt_in:
                alternatives.append({**entry, "why_not_default": tier.get("note")})
                continue
            entry["alternatives"] = alternatives
            if tier.get("last_resort"):
                entry["note"] = tier["note"]
            return entry
    return {"id": None, "reason": "nothing fits", "alternatives": alternatives}


def recommend(hw: Hardware, table: dict | None = None, *, with_bonsai: bool = True) -> dict:
    table = table or load_table()
    bonsai = pick_bonsai(hw, table)
    jev = pick_jev(hw, table, bonsai=bonsai if with_bonsai else None)
    return {"hardware": {**asdict(hw)}, "bonsai": bonsai, "jev": jev,
            "not_picked_by_licence": table["not_default_licence"],
            "not_picked_note": table["not_default_note"],
            "data": {"file": "auto_router/data/local_models.json", "retrieved": table["retrieved"]},
            "assumptions": "memory thresholds are assumptions (file size plus a margin); "
                           "Bonsai 2 routing quality is not measured by this repository"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect hardware and pick local models.")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.add_argument("--with-bonsai", action="store_true",
                   help="reserve memory for Bonsai 2 when picking the Jev-class model")
    args = p.parse_args(argv)
    rec = recommend(detect(), with_bonsai=args.with_bonsai)
    if args.json:
        print(json.dumps(rec, indent=2))
        return 0
    hw = rec["hardware"]
    gpus = ", ".join(f"{g['name']} ({g['vram_gb']} GB{' unified' if g['unified'] else ''})"
                     for g in hw["gpus"]) or "none detected"
    print(f"OS: {hw['os']}{' (WSL)' if hw['wsl'] else ''}  arch: {hw['arch']}  RAM: {hw['ram_gb']} GB")
    print(f"GPU: {gpus}")
    b = rec["bonsai"]
    print(f"Bonsai 2: {'fits: ' + b['file'] + ' on ' + b['device'] if b['fits'] else 'does not fit'} - {b['reason']}")
    j = rec["jev"]
    print(f"Jev-class model: {j.get('display') or 'none'}" + (f" on {j['device']}" if j.get("device") else ""))
    print(f"(data retrieved {rec['data']['retrieved']}; {rec['assumptions']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
