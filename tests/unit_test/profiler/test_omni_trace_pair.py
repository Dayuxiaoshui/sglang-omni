# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the omni-gpu-deep-dive steady-state gate.

The gate in ``.claude/skills/omni-gpu-deep-dive/scripts/omni_trace_pair.py`` is
what stops a profiling run from charging one-time cost -- Dynamo/Inductor
compilation, CUDA graph capture -- to a steady-state kernel. Its whole value is
in one distinction that is easy to break while editing the marker tuples:

* reject *compiling* and *capturing*
* accept *compiled* and *captured execution*, because ``cudaGraphLaunch`` is what
  a healthy formal trace is full of and ``is_torchdynamo_compiling`` is a
  predicate every HuggingFace forward calls

``.claude/`` is not on the pytest path, so the module is loaded by file path.
Nothing here needs a GPU: the profiler and ``torch.cuda.synchronize`` are
replaced by fakes so the capture flow itself can be exercised too.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT
    / ".claude"
    / "skills"
    / "omni-gpu-deep-dive"
    / "scripts"
    / "omni_trace_pair.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("omni_trace_pair", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trace_pair() -> ModuleType:
    if not _SCRIPT.exists():
        pytest.skip(f"{_SCRIPT} is not present in this checkout")
    return _load_module()


def _write_trace(path: Path, event_names: list[str]) -> Path:
    """Write the minimal gzipped chrome trace the gate reads."""
    events = [{"name": name, "ph": "X", "ts": 0, "dur": 1} for name in event_names]
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    return path


def test_gate_accepts_captured_and_compiled_execution(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """A healthy formal trace replays graphs and calls the is-compiling predicate."""
    trace = _write_trace(
        tmp_path / "formal.trace.json.gz",
        [
            "cudaGraphLaunch",
            "cudaLaunchKernel",
            "is_torchdynamo_compiling",
            "triton_poi_fused_add_0",
            "aten::mm",
        ],
    )
    trace_pair.assert_steady_state(trace, tag="formal")


@pytest.mark.parametrize(
    "event_name",
    [
        "torch/_inductor/codecache.py(1234): load",
        "torch/_dynamo/convert_frame.py(900): _compile",
        "cudaModuleLoad",
        "cuModuleLoad",
    ],
)
def test_gate_rejects_compilation(
    trace_pair: ModuleType, tmp_path: Path, event_name: str
) -> None:
    """Compilation inside the profiled window must fail loudly, not be subtracted."""
    trace = _write_trace(tmp_path / "mapping.trace.json.gz", ["aten::mm", event_name])
    with pytest.raises(RuntimeError, match="not steady state"):
        trace_pair.assert_steady_state(trace, tag="mapping")


@pytest.mark.parametrize(
    "event_name",
    ["cudaStreamBeginCapture", "cudaStreamEndCapture", "cudaGraphInstantiate"],
)
def test_gate_rejects_graph_capture(
    trace_pair: ModuleType, tmp_path: Path, event_name: str
) -> None:
    trace = _write_trace(tmp_path / "formal.trace.json.gz", ["aten::mm", event_name])
    with pytest.raises(RuntimeError, match="not steady state"):
        trace_pair.assert_steady_state(trace, tag="formal")


def test_allow_capture_permits_capture_but_still_rejects_compilation(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """``allow_capture`` is an escape hatch for capture only."""
    captured = _write_trace(
        tmp_path / "captured.trace.json.gz", ["cudaStreamBeginCapture", "aten::mm"]
    )
    trace_pair.assert_steady_state(captured, tag="formal", allow_capture=True)

    compiled = _write_trace(
        tmp_path / "compiled.trace.json.gz", ["cudaModuleLoad", "aten::mm"]
    )
    with pytest.raises(RuntimeError, match="not steady state"):
        trace_pair.assert_steady_state(compiled, tag="formal", allow_capture=True)


def test_await_compression_waits_for_the_json_source_to_vanish(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """``gzip -f`` creates the archive first and unlinks the source only on success.

    So an existing ``.gz`` next to a still-present ``.json`` means the background
    subprocess is mid-write, and the gate would read a truncated file.
    """
    gz_path = tmp_path / "trace.json.gz"
    json_path = tmp_path / "trace.json"
    gz_path.write_bytes(b"")
    json_path.write_bytes(b"{}")

    with pytest.raises(TimeoutError, match="background gzip did not finish"):
        trace_pair.await_compression(gz_path, timeout_s=0.5)

    json_path.unlink()
    trace_pair.await_compression(gz_path, timeout_s=0.5)


def test_capture_accepts_a_string_output_dir_and_gates_the_trace(
    trace_pair: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """argparse hands over a str, so the API boundary must coerce it.

    Also pins the rest of the capture contract on CPU: warmup runs before the
    profiler is armed, ``iters`` calls are recorded, and the with_stack env var
    is set before ``start`` -- which is where ``TorchProfiler`` reads it.
    """
    calls: list[str] = []
    seen_with_stack: list[str | None] = []

    class FakeProfiler:
        @staticmethod
        def start(trace_path_template: str, run_id: str | None = None) -> str:
            calls.append(f"start:{run_id}")
            import os

            seen_with_stack.append(os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK"))
            gz_path = Path(f"{trace_path_template}_rank0.trace.json.gz")
            _write_trace(gz_path, ["cudaLaunchKernel", "aten::mm"])
            return str(gz_path)

        @staticmethod
        def stop(*, run_id: str | None = None) -> None:
            calls.append(f"stop:{run_id}")

    monkeypatch.setattr(trace_pair, "TorchProfiler", FakeProfiler)
    monkeypatch.setattr(trace_pair.torch.cuda, "synchronize", lambda: None)

    run_dir = trace_pair.capture(
        output_dir=str(tmp_path / "run"),
        tag="mapping",
        body=lambda: calls.append("body"),
        iters=3,
        warmup=2,
        with_stack=True,
    )

    assert run_dir == tmp_path / "run" / "mapping"
    assert seen_with_stack == ["1"]
    assert calls == ["body", "body", "start:mapping"] + ["body"] * 3 + ["stop:mapping"]
