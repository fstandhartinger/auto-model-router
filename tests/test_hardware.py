"""Hardware detection and the local-model picker, with mocked probes."""
import json

import pytest

from auto_router import hardware as hw
from auto_router.hardware import GPU, Hardware

TABLE = hw.load_table()


def machine(ram, gpus=(), os_name="linux"):
    return Hardware(os=os_name, wsl=False, arch="x86_64", ram_gb=ram, gpus=list(gpus))


def test_table_has_sources_date_and_real_sizes():
    assert TABLE["retrieved"] == "2026-09-25"
    sizes = {f["file"]: f["bytes"] for f in TABLE["bonsai"]["files"]}
    assert sizes["Ternary-Bonsai-2-27B-PTQ1_0.gguf"] == 5946648928
    assert sizes["Ternary-Bonsai-2-27B-PQ2_0.gguf"] == 7206168928
    for tier in TABLE["jev_tiers"]:
        assert tier["source"].startswith("https://")
        assert tier["licence"] and tier["jevbench"]["version"] == "v1.4.2"
    assert all(e["licence"] for e in TABLE["not_default_licence"])


def test_non_commercial_models_are_never_picked():
    ids = {e["id"] for e in TABLE["not_default_licence"]}
    assert {"malkuth-4b", "malkuth-2b"} <= ids
    for ram in (2, 8, 16, 64):
        for gpus in ([], [GPU("nvidia", "big", 48.0)]):
            pick = hw.pick_jev(machine(ram, gpus), TABLE)
            assert pick.get("id") not in ids
            assert all(a["id"] not in ids for a in pick.get("alternatives", []))


@pytest.mark.parametrize("gpus,ram,file,device", [
    ([GPU("nvidia", "RTX 4090", 24.0)], 32, "Ternary-Bonsai-2-27B-PQ2_0.gguf", "cuda"),
    ([GPU("nvidia", "RTX 3070", 8.0)], 16, "Ternary-Bonsai-2-27B-PTQ1_0.gguf", "cuda"),
    ([GPU("apple", "Apple M3", 18.0, unified=True)], 18, "Ternary-Bonsai-2-27B-PQ2_0.gguf", "metal"),
    ([GPU("nvidia", "GTX 1650", 4.0)], 32, "Ternary-Bonsai-2-27B-PTQ1_0.gguf", "cpu"),
    ([GPU("amd", "RX 7900", 24.0)], 16, "Ternary-Bonsai-2-27B-PTQ1_0.gguf", "cpu"),
])
def test_bonsai_fit(gpus, ram, file, device):
    b = hw.pick_bonsai(machine(ram, gpus), TABLE)
    assert b["fits"] and b["file"] == file and b["device"] == device


def test_bonsai_does_not_fit_small_machines():
    b = hw.pick_bonsai(machine(8, [GPU("apple", "M1", 8.0, unified=True)], "macos"), TABLE)
    assert not b["fits"] and "needs" in b["reason"]


def test_jev_tiers_by_memory():
    assert hw.pick_jev(machine(16, [GPU("nvidia", "4060", 8.0)]), TABLE)["id"] == "jevk5-v0.2-q8"
    small_gpu = hw.pick_jev(machine(16, [GPU("nvidia", "1650", 4.0)]), TABLE)
    assert (small_gpu["id"], small_gpu["device"]) == ("jevk5-v0.2-q4", "cuda")   # GPU before CPU
    assert hw.pick_jev(machine(16), TABLE)["id"] == "jevk5-v0.2-q8"
    assert hw.pick_jev(machine(16, [GPU("apple", "M1", 16.0, unified=True)], "macos"), TABLE)["device"] == "metal"
    assert hw.pick_jev(machine(8), TABLE)["id"] == "jevk5-v0.2-q4"
    assert hw.pick_jev(machine(6), TABLE)["id"] == "laya-native"
    last = hw.pick_jev(machine(2), TABLE)
    assert last["id"] == "laya-webgpu" and last["device"] == "browser" and "note" in last


def test_decider_is_only_an_opt_in_alternative_on_big_nvidia():
    pick = hw.pick_jev(machine(64, [GPU("nvidia", "RTX 4090", 24.0)]), TABLE)
    assert pick["id"] == "jevk5-v0.2-q8"
    assert [a["id"] for a in pick["alternatives"]] == ["decider-4b-v2"]
    assert hw.pick_jev(machine(64, [GPU("nvidia", "RTX 4090", 24.0)]), TABLE,
                       include_opt_in=True)["id"] == "decider-4b-v2"


def test_bonsai_memory_is_reserved_before_picking_jev():
    m = machine(16, [GPU("nvidia", "RTX 4060", 8.0)])
    b = hw.pick_bonsai(m, TABLE)
    assert b["device"] == "cuda"
    j = hw.pick_jev(m, TABLE, bonsai=b)
    assert j["device"] == "cpu"          # 8 GB VRAM is taken by Bonsai


def test_parsers():
    assert hw.parse_meminfo("MemTotal:       16318412 kB\n") == 15.6
    g = hw.parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564\nNVIDIA A10, n/a\n")
    assert g[0].vram_gb == 24.0 and g[1].vram_gb is None
    rocm = hw.parse_rocm_smi(json.dumps({"card0": {"VRAM Total Memory (B)": str(24 * 1024 ** 3),
                                                   "VRAM Total Used Memory (B)": "0"}}))
    assert rocm[0].vendor == "amd" and rocm[0].vram_gb == 24.0
    sp = json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M3 Pro", "spdisplays_vendor": "sppci_vendor_Apple"}]})
    assert hw.parse_system_profiler(sp, 36.0)[0].unified
    ram, gpus = hw.parse_windows_cim(json.dumps({"ram": 34359738368,
                                                 "gpus": {"Name": "NVIDIA GeForce RTX 4060 Laptop GPU",
                                                          "AdapterRAM": 4293918720}}))
    assert ram == 32.0 and gpus[0].vendor == "nvidia"


def fake_run(outputs):
    def run(cmd):
        return outputs.get(cmd[0])
    return run


def test_detect_linux_with_nvidia():
    h = hw.detect(fake_run({"nvidia-smi": "NVIDIA GeForce RTX 3070, 8192\n"}), os_name="linux",
                  meminfo="MemTotal: 33554432 kB\n")
    assert h.ram_gb == 32.0 and h.gpus[0].vram_gb == 8.0 and h.probes["gpu"] == "nvidia-smi"


def test_detect_macos_unified():
    sp = json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M2 Max"}]})
    h = hw.detect(fake_run({"sysctl": str(64 * 1024 ** 3), "system_profiler": sp}), os_name="macos")
    rec = hw.recommend(h, TABLE)
    assert h.ram_gb == 64.0 and rec["bonsai"]["device"] == "metal"


def test_detect_windows_prefers_nvidia_smi_over_32bit_adapter_ram():
    cim = json.dumps({"ram": 17179869184, "gpus": [{"Name": "NVIDIA GeForce RTX 4060 Laptop GPU",
                                                   "AdapterRAM": 4293918720}]})
    h = hw.detect(fake_run({"powershell": cim, "pwsh": cim, "nvidia-smi": "NVIDIA GeForce RTX 4060 Laptop GPU, 8188\n"}),
                  os_name="windows")
    assert [g.vram_gb for g in h.gpus] == [8.0]


def test_detect_without_any_tool():
    h = hw.detect(lambda cmd: None, os_name="linux", meminfo="")
    rec = hw.recommend(h, TABLE)
    assert h.ram_gb is None and rec["jev"]["id"] == "laya-webgpu"


def test_cli_json(capsys, monkeypatch):
    monkeypatch.setattr(hw, "detect", lambda: machine(16))
    assert hw.main(["--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["jev"]["id"] == "jevk5-v0.2-q8" and out["data"]["retrieved"] == "2026-09-25"
