# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the omni-gpu-deep-dive steady-state gate.

The gate in ``.claude/skills/omni-gpu-deep-dive/scripts/omni_trace_pair.py`` is
what stops a profiling run from charging one-time cost -- Dynamo/Inductor
compilation, CUDA graph capture -- to a steady-state kernel. Its whole value is
in one distinction that is easy to break while editing the marker tuples:

* reject *compiling* and *capturing*
* accept *compiled* and *captured execution*, because ``cudaGraphLaunch`` is what
  a healthy formal trace is full of, ``is_torchdynamo_compiling`` is a predicate
  every HuggingFace forward calls, and with ``with_stack`` on, a trace of
  compiled code carries inductor frames long after compilation finished

``.claude/`` is not on the pytest path, so the module is loaded by file path.
Nothing here needs a GPU, and nothing here needs the serving runtime: only
``capture`` imports omni's profiler, and the tests replace it and
``torch.cuda.synchronize`` with fakes.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
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
_PROFILER_MODULE = "sglang_omni.profiler.torch_profiler"


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


def _write_trace(path: Path, event_names: list[str], *, cat: str = "cpu_op") -> Path:
    """Write the minimal gzipped chrome trace the gate reads.

    Timestamps increase per event so the samples in a failure message can be
    told apart, which is the point of reporting them.
    """
    events = [
        {"name": name, "cat": cat, "ph": "X", "ts": index, "dur": 1}
        for index, name in enumerate(event_names)
    ]
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    return path


def test_the_gate_imports_without_the_serving_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only ``capture`` needs omni's profiler; the gate is stdlib plus torch.

    A ``None`` entry in ``sys.modules`` makes that import raise, which stands in
    for the common case: a stage that is plain torch, on a box without the
    pinned CUDA stack that ``sglang`` pulls in. Such a stage can capture with
    ``torch.profiler`` itself, and it should still be able to gate the result.
    """
    monkeypatch.setitem(sys.modules, _PROFILER_MODULE, None)
    module = _load_module()
    trace = _write_trace(tmp_path / "formal.trace.json.gz", ["cudaGraphLaunch"])
    module.assert_steady_state(trace, tag="formal")
    with pytest.raises(ImportError):
        module._torch_profiler()


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
        "torch/_dynamo/convert_frame.py(900): _compile",
        "torch/_inductor/compile_fx.py(1500): compile_fx",
        "torch/_inductor/async_compile.py(300): triton",
        "torch/_inductor/codecache.py(1234): load",
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


def test_gate_accepts_inductor_frames_that_are_not_compilation(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """The false positive that made a real run bypass this gate.

    These four names were observed in a ``with_stack`` trace whose window opened
    *after* compilation finished: ``output_code.py`` is how an already-compiled
    graph is entered, and the ``compile_worker`` threads sit in a blocking read
    for the life of the process, so they land in every trace. A gate that
    rejects them fails clean runs, and a gate that fails clean runs gets
    bypassed -- which is worse than not having one.
    """
    trace = _write_trace(
        tmp_path / "mapping.trace.json.gz",
        [
            "torch/_inductor/output_code.py(581): __call__",
            "torch/_inductor/compile_worker/subproc_pool.py(195): _read_thread",
            "torch/_inductor/compile_worker/subproc_pool.py(61): _recv_msg",
            "torch/_inductor/runtime/autotune_cache.py(481): end_compile",
            "aten::mm",
        ],
        cat="python_function",
    )
    trace_pair.assert_steady_state(trace, tag="mapping")


def test_gate_failure_names_the_matched_events_with_category_and_timestamp(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """The marker substring alone cannot tell a stack frame from real work."""
    trace = _write_trace(
        tmp_path / "mapping.trace.json.gz",
        ["aten::mm", "torch/_inductor/compile_fx.py(1500): compile_fx"],
        cat="python_function",
    )
    with pytest.raises(RuntimeError) as excinfo:
        trace_pair.assert_steady_state(trace, tag="mapping")

    message = str(excinfo.value)
    assert "torch/_inductor/compile_fx.py(1500): compile_fx" in message
    assert "cat=python_function" in message
    assert "ts=1" in message


def test_violations_are_bounded_so_one_marker_cannot_flood_the_message(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """A real mapping trace runs to hundreds of MB; the report stays readable."""
    trace = _write_trace(
        tmp_path / "mapping.trace.json.gz",
        ["torch/_inductor/compile_fx.py(1500): compile_fx"] * 50 + ["aten::mm"],
        cat="python_function",
    )
    hits = trace_pair.steady_state_violations(trace, samples=3)
    assert list(hits) == ["torch/_inductor/compile_fx"]
    assert len(hits["torch/_inductor/compile_fx"]) == 3


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

    monkeypatch.setattr(trace_pair, "_torch_profiler", lambda: FakeProfiler)
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
