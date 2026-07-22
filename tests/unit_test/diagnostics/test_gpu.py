# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import json
from types import ModuleType, SimpleNamespace

from typer.testing import CliRunner

import sglang_omni.diagnostics.gpu as gpu_diagnostics
from sglang_omni.cli import app


class _FakeCuda:
    def __init__(self) -> None:
        self.properties = [
            SimpleNamespace(
                name="GPU-A",
                major=12,
                minor=0,
                total_memory=32 * 1024**3,
                uuid="uuid-a",
            ),
            SimpleNamespace(
                name="GPU-B",
                major=12,
                minor=0,
                total_memory=32 * 1024**3,
                uuid="uuid-b",
            ),
        ]

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self.properties)

    def get_device_properties(self, index: int):
        return self.properties[index]

    def can_device_access_peer(self, source: int, target: int) -> bool:
        return False


class _FakeTorch:
    __version__ = "2.11.0+cu130"
    version = SimpleNamespace(cuda="13.0")

    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeNVML(ModuleType):
    NVML_P2P_STATUS_OK = 0
    NVML_P2P_CAPS_INDEX_READ = 0
    NVML_TOPOLOGY_NODE = 40

    def __init__(self) -> None:
        super().__init__("pynvml")
        self.shutdown_called = False

    def nvmlInit(self) -> None:
        pass

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        return f"handle:{index}"

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> SimpleNamespace:
        index = int(handle.split(":")[1])
        return SimpleNamespace(total=32 * 1024**3, free=(30 - index) * 1024**3)

    def nvmlDeviceGetPciInfo(self, handle: str) -> SimpleNamespace:
        index = int(handle.split(":")[1])
        return SimpleNamespace(busId=f"00000000:{index + 1:02x}:00.0")

    def nvmlDeviceGetCudaComputeCapability(
        self, handle: str
    ) -> tuple[int, int]:
        return (12, 0)

    def nvmlDeviceGetUUID(self, handle: str) -> str:
        index = int(handle.split(":")[1])
        return f"GPU-uuid-{'a' if index == 0 else 'b'}"

    def nvmlDeviceGetName(self, handle: str) -> str:
        index = int(handle.split(":")[1])
        return f"GPU-{'A' if index == 0 else 'B'}"

    def nvmlSystemGetDriverVersion(self) -> str:
        return "610.43.02"

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        return 13030

    def nvmlDeviceGetP2PStatus(
        self, source: str, target: str, index: int
    ) -> int:
        return 1

    def nvmlDeviceGetTopologyCommonAncestor(
        self, source: str, target: str
    ) -> int:
        return self.NVML_TOPOLOGY_NODE


def test_collect_gpu_diagnostics_preserves_reordered_visible_mapping(
    monkeypatch,
) -> None:
    fake_nvml = _FakeNVML()
    fake_torch = _FakeTorch()
    fake_torch.cuda.properties.reverse()
    monkeypatch.setattr(gpu_diagnostics, "_cuda_runtime_version", lambda: "13.3")
    monkeypatch.setattr(gpu_diagnostics, "_backend_inventory", lambda: [])

    report = gpu_diagnostics.collect_gpu_diagnostics(
        env={"CUDA_VISIBLE_DEVICES": "1,0"},
        torch_module=fake_torch,
        pynvml_module=fake_nvml,
    )

    assert [gpu["logical_index"] for gpu in report["gpus"]] == [0, 1]
    assert [gpu["physical_index"] for gpu in report["gpus"]] == [1, 0]
    assert [gpu["uuid"] for gpu in report["gpus"]] == [
        "GPU-uuid-b",
        "GPU-uuid-a",
    ]
    assert report["selection"]["attention_backend"] is None
    assert report["environment"]["cuda_visible_devices"] == "1,0"
    assert report["p2p"]["status"] == "unavailable"
    assert report["p2p"]["matrix"] == [[None, False], [False, None]]
    assert report["topology"]["matrix"] == [
        ["self", "node"],
        ["node", "self"],
    ]
    rendered = gpu_diagnostics.render_gpu_diagnostics(report)
    assert "logical 0 -> physical 1" in rendered
    assert fake_nvml.shutdown_called is True


def test_backend_inventory_reports_installed_but_unimportable(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_diagnostics,
        "_BACKENDS",
        (("communication", "nixl", "nixl-cu13", "nixl"),),
    )
    monkeypatch.setattr(gpu_diagnostics, "_package_version", lambda name: "1.3.1")
    monkeypatch.setattr(gpu_diagnostics, "_module_available", lambda name: False)

    backend = gpu_diagnostics._backend_inventory()[0]

    assert backend["installed"] is True
    assert backend["importable"] is False
    assert "not importable" in backend["reason"]


def test_check_gpu_json_does_not_load_a_model(monkeypatch) -> None:
    check_gpu_module = importlib.import_module("sglang_omni.cli.check_gpu")
    report = {
        "schema_version": 1,
        "environment": {"logical_device_count": 0},
        "gpus": [],
    }
    monkeypatch.setattr(
        check_gpu_module,
        "collect_gpu_diagnostics",
        lambda: report,
    )

    result = CliRunner().invoke(app, ["check-gpu", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == report
