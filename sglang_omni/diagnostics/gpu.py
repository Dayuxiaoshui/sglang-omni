# SPDX-License-Identifier: Apache-2.0
"""Model-free GPU, topology, and backend diagnostics."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Mapping
from typing import Any

from sglang_omni.utils.gpu_memory import (
    _decode_nvml_string,
    _shutdown_nvml,
    _try_import_pynvml,
    format_bytes_gib,
    parse_cuda_visible_devices,
)

_BACKENDS = (
    ("attention", "flash-attn-4", "flash-attn-4", "flash_attn.cute"),
    ("attention", "flashinfer", "flashinfer-python", "flashinfer"),
    ("attention", "triton", "triton", "triton"),
    ("attention", "torch-sdpa", "torch", "torch.nn.functional"),
    ("gemm", "sgl-deep-gemm", "sgl-deep-gemm", "deep_gemm"),
    ("gemm", "sglang-kernel", "sglang-kernel", "sgl_kernel"),
    ("moe", "quack-kernels", "quack-kernels", "quack"),
    ("quantization", "torchao", "torchao", "torchao"),
    (
        "quantization",
        "compressed-tensors",
        "compressed-tensors",
        "compressed_tensors",
    ),
    ("communication", "nixl", "nixl-cu13", "nixl_cu13"),
    (
        "communication",
        "mooncake",
        "mooncake-transfer-engine-cuda13",
        "mooncake",
    ),
)
_NO_MODEL_REASON = "No model or server configuration was loaded."


def _cuda_version(value: int | None) -> str | None:
    if not value:
        return None
    return f"{value // 1000}.{(value % 1000) // 10}"


def _normalize_uuid(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized.removeprefix("gpu-") or None


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_import_error(module: str) -> str | None:
    try:
        importlib.import_module(module)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _backend_inventory() -> list[dict[str, Any]]:
    backends = []
    for category, name, distribution, module in _BACKENDS:
        version = _package_version(distribution)
        import_error = (
            _module_import_error(module) if version is not None else None
        )
        importable = version is not None and import_error is None
        reason = None
        if version is None:
            reason = f"Distribution {distribution!r} is not installed."
        elif import_error is not None:
            reason = (
                f"Distribution {distribution!r} is installed, but module "
                f"{module!r} failed to import: {import_error}"
            )
        backends.append(
            {
                "category": category,
                "name": name,
                "distribution": distribution,
                "version": version,
                "module": module,
                "installed": version is not None,
                "importable": importable,
                "reason": reason,
            }
        )
    return backends


def _cuda_runtime_version() -> str | None:
    try:
        runtime = importlib.import_module("cuda.bindings.runtime")
        status, version = runtime.cudaRuntimeGetVersion()
        return _cuda_version(int(version)) if int(status) == 0 else None
    except Exception:
        return None


def _nvml_inventory(
    pynvml: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, str | None], list[str]]:
    system = {"driver_version": None, "cuda_driver_api_version": None}
    inventory: list[dict[str, Any]] = []
    warnings: list[str] = []
    if pynvml is None:
        return inventory, system, warnings

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        return inventory, system, [f"NVML initialization failed: {exc}"]

    try:
        system["driver_version"] = _decode_nvml_string(
            pynvml.nvmlSystemGetDriverVersion()
        )
    except Exception as exc:
        warnings.append(f"NVML driver query failed: {exc}")
    try:
        system["cuda_driver_api_version"] = _cuda_version(
            int(pynvml.nvmlSystemGetCudaDriverVersion_v2())
        )
    except Exception as exc:
        warnings.append(f"NVML CUDA driver query failed: {exc}")

    try:
        count = int(pynvml.nvmlDeviceGetCount())
        for physical_index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            inventory.append(
                {
                    "physical_index": physical_index,
                    "uuid": _decode_nvml_string(pynvml.nvmlDeviceGetUUID(handle)),
                    "pci_bus_id": _decode_nvml_string(pci.busId),
                    "name": _decode_nvml_string(pynvml.nvmlDeviceGetName(handle)),
                    "compute_capability": f"{int(major)}.{int(minor)}",
                    "total_memory_bytes": int(memory.total),
                    "free_memory_bytes": int(memory.free),
                    "_handle": handle,
                }
            )
    except Exception as exc:
        warnings.append(f"NVML device inventory failed: {exc}")
    return inventory, system, warnings


def _physical_device(
    logical_index: int,
    properties: Any | None,
    visible_devices: list[int | str],
    by_index: dict[int, dict[str, Any]],
    by_uuid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    torch_uuid = _normalize_uuid(getattr(properties, "uuid", None))
    if torch_uuid in by_uuid:
        return by_uuid[torch_uuid]

    if logical_index < len(visible_devices):
        visible = visible_devices[logical_index]
        if isinstance(visible, int):
            return by_index.get(visible, {})
        return by_uuid.get(_normalize_uuid(visible), {})

    return by_index.get(logical_index, {}) if not visible_devices else {}


def _logical_devices(
    torch: Any,
    visible_devices: list[int | str],
    inventory: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []

    by_index = {device["physical_index"]: device for device in inventory}
    by_uuid = {
        uuid: device
        for device in inventory
        if (uuid := _normalize_uuid(device.get("uuid"))) is not None
    }
    devices = []
    for logical_index in range(int(torch.cuda.device_count())):
        try:
            properties = torch.cuda.get_device_properties(logical_index)
        except Exception as exc:
            warnings.append(f"PyTorch GPU {logical_index} query failed: {exc}")
            properties = None
        visible_device = (
            visible_devices[logical_index]
            if logical_index < len(visible_devices)
            else logical_index
        )
        physical = _physical_device(
            logical_index, properties, visible_devices, by_index, by_uuid
        )
        if (
            isinstance(visible_device, str)
            and visible_device.upper().startswith("MIG-")
            and not physical
        ):
            warnings.append(
                f"CUDA_VISIBLE_DEVICES entry {visible_device!r} is a MIG device; "
                "physical GPU mapping, free memory, and topology are unsupported."
            )
        torch_cc = (
            f"{properties.major}.{properties.minor}"
            if properties is not None
            else None
        )
        devices.append(
            {
                "logical_index": logical_index,
                "visible_device": visible_device,
                "physical_index": physical.get("physical_index"),
                "uuid": physical.get("uuid")
                or str(getattr(properties, "uuid", "") or "") or None,
                "pci_bus_id": physical.get("pci_bus_id"),
                "name": physical.get("name") or getattr(properties, "name", None),
                "compute_capability": physical.get("compute_capability") or torch_cc,
                "total_memory_bytes": physical.get("total_memory_bytes")
                or getattr(properties, "total_memory", None),
                "free_memory_bytes": physical.get("free_memory_bytes"),
                "_handle": physical.get("_handle"),
            }
        )
    return devices


def _p2p_report(
    torch: Any, device_count: int
) -> tuple[list[list[bool | None]], str, str]:
    matrix = [[None for _ in range(device_count)] for _ in range(device_count)]
    if device_count < 2:
        return matrix, "not_applicable", "Fewer than two visible GPUs."

    try:
        for source in range(device_count):
            for target in range(device_count):
                if source != target:
                    matrix[source][target] = bool(
                        torch.cuda.can_device_access_peer(source, target)
                    )
    except Exception as exc:
        return matrix, "unknown", f"P2P query failed: {exc}"

    pairs = [
        matrix[source][target]
        for source in range(device_count)
        for target in range(device_count)
        if source != target
    ]
    if all(pairs):
        return matrix, "full", "All visible GPU pairs support direct peer access."
    return (
        matrix,
        "unavailable",
        "P2P is unavailable; keep custom all-reduce disabled and use "
        "NCCL/host-staged transport.",
    )


def _topology_name(pynvml: Any, value: Any) -> str:
    for name, constant in (
        ("internal", "NVML_TOPOLOGY_INTERNAL"),
        ("single", "NVML_TOPOLOGY_SINGLE"),
        ("multiple", "NVML_TOPOLOGY_MULTIPLE"),
        ("host_bridge", "NVML_TOPOLOGY_HOSTBRIDGE"),
        ("node", "NVML_TOPOLOGY_NODE"),
        ("system", "NVML_TOPOLOGY_SYSTEM"),
    ):
        if value == getattr(pynvml, constant, object()):
            return name
    return str(value)


def _topology_matrix(
    pynvml: Any | None,
    devices: list[dict[str, Any]],
    warnings: list[str],
) -> list[list[str | None]]:
    count = len(devices)
    matrix = [["self" if i == j else None for j in range(count)] for i in range(count)]
    if pynvml is None or any(device["_handle"] is None for device in devices):
        return matrix

    get_ancestor = getattr(pynvml, "nvmlDeviceGetTopologyCommonAncestor", None)
    if get_ancestor is None:
        return matrix
    for source in range(count):
        for target in range(count):
            if source == target:
                continue
            try:
                value = get_ancestor(
                    devices[source]["_handle"], devices[target]["_handle"]
                )
                matrix[source][target] = _topology_name(pynvml, value)
            except Exception as exc:
                warnings.append(
                    f"NVML topology query failed for logical GPUs "
                    f"{source}->{target}: {exc}"
                )
    return matrix


def collect_gpu_diagnostics(
    *,
    env: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
    pynvml_module: Any | None = None,
) -> dict[str, Any]:
    """Collect diagnostics without loading model configuration or weights."""

    source_env = os.environ if env is None else env
    visible_value = source_env.get("CUDA_VISIBLE_DEVICES")
    visible_devices = parse_cuda_visible_devices(visible_value)
    torch = torch_module or importlib.import_module("torch")
    pynvml = pynvml_module if pynvml_module is not None else _try_import_pynvml()

    inventory, system, warnings = _nvml_inventory(pynvml)
    try:
        devices = _logical_devices(torch, visible_devices, inventory, warnings)
        p2p_matrix, p2p_status, p2p_reason = _p2p_report(torch, len(devices))
        topology = _topology_matrix(pynvml, devices, warnings)
    finally:
        if pynvml is not None:
            _shutdown_nvml(pynvml)

    for device in devices:
        device.pop("_handle", None)

    backends = _backend_inventory()
    warnings.extend(
        backend["reason"]
        for backend in backends
        if backend["installed"] and not backend["importable"]
    )
    return {
        "schema_version": 1,
        "environment": {
            "cuda_visible_devices": visible_value,
            **system,
            "cuda_runtime_version": _cuda_runtime_version(),
            "pytorch_version": getattr(torch, "__version__", None),
            "pytorch_cuda_build": getattr(
                getattr(torch, "version", None), "cuda", None
            ),
            "cuda_available": bool(torch.cuda.is_available()),
            "logical_device_count": len(devices),
        },
        "gpus": devices,
        "topology": {"matrix": topology},
        "p2p": {
            "status": p2p_status,
            "matrix": p2p_matrix,
            "fallback_reason": p2p_reason,
        },
        "backends": backends,
        "selection": {
            "attention_backend": None,
            "gemm_backend": None,
            "moe_backend": None,
            "quantization_backend": None,
            "cuda_graph": None,
            "torch_compile": None,
            "reason": _NO_MODEL_REASON,
        },
        "warnings": warnings,
    }


def render_gpu_diagnostics(report: Mapping[str, Any]) -> str:
    """Render a compact diagnostic summary for terminal output."""

    environment = report["environment"]
    visible = environment["cuda_visible_devices"]
    lines = [
        "SGLang-Omni GPU diagnostics (no model loaded)",
        f"CUDA_VISIBLE_DEVICES: {visible if visible is not None else '<unset>'}",
        f"Driver: {environment['driver_version'] or 'unavailable'}",
        (
            "CUDA driver/runtime: "
            f"{environment['cuda_driver_api_version'] or 'unavailable'} / "
            f"{environment['cuda_runtime_version'] or 'unavailable'}"
        ),
        (
            "PyTorch/CUDA build: "
            f"{environment['pytorch_version'] or 'unavailable'} / "
            f"{environment['pytorch_cuda_build'] or 'unavailable'}"
        ),
        "GPUs:",
    ]
    if not report["gpus"]:
        lines.append("  No CUDA devices are visible to PyTorch.")
    for device in report["gpus"]:
        lines.append(
            f"  logical {device['logical_index']} -> physical "
            f"{device['physical_index']} visible={device['visible_device']} "
            f"name={device['name'] or 'unknown'} "
            f"cc={device['compute_capability'] or 'unknown'} "
            f"memory={format_bytes_gib(device['free_memory_bytes'])}/"
            f"{format_bytes_gib(device['total_memory_bytes'])} "
            f"uuid={device['uuid'] or 'unknown'} "
            f"pci={device['pci_bus_id'] or 'unknown'}"
        )

    lines.extend(
        [
            f"Topology: {report['topology']['matrix'] or 'unavailable'}",
            (
                f"P2P: {report['p2p']['status']} "
                f"matrix={report['p2p']['matrix'] or 'unavailable'}"
            ),
            f"  Fallback: {report['p2p']['fallback_reason']}",
            "Backends:",
        ]
    )
    for backend in report["backends"]:
        status = (
            "available"
            if backend["importable"]
            else "installed, module unavailable"
            if backend["installed"]
            else "not installed"
        )
        lines.append(
            f"  {backend['category']}/{backend['name']}: {status}; "
            f"version={backend['version'] or '-'}"
        )

    lines.extend(
        [
            f"Selection: not evaluated; {report['selection']['reason']}",
            "CUDA Graph/torch.compile: not evaluated",
        ]
    )
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  {warning}" for warning in report["warnings"])
    return "\n".join(lines)
