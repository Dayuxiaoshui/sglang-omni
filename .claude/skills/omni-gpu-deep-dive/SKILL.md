---
name: omni-gpu-deep-dive
description: "Attribute GPU time in one sglang-omni stage to specific lines of `sglang_omni/` python source. Drives the `llm-torch-profiler-analysis` backend with a mapping/formal trace pair plus steady-state gates, and carries the rules for reading its three tables on an omni stage. Use when a stage is slower than expected and the question is which python code is the bottleneck, not which kernel is hot."
---

# Omni GPU Deep Dive

## Overview

The kernel, overlap, and fuse analysis is not in this repo. It lives in the
`llm-torch-profiler-analysis` skill:

- https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS

Source attribution for `sglang_omni/` and `sglang_omni_router/` is part of that
skill as of its PR #96, so nothing here patches it. This skill owns three things:

- the backend pointer and its capability check
- the mapping/formal trace pair and its steady-state gates,
  [scripts/omni_trace_pair.py](scripts/omni_trace_pair.py)
- the workload contract and the omni reading rules, this file

Omni is not a separate framework to the backend. It serves through the SGLang
runtime, so always pass `--framework sglang`. The backend's SGLang live-capture
path does not work against an omni server: omni's `/start_profile` takes
`run_id` and `trace_path_template` and stops on `/stop_profile`, while the
backend sends `output_dir` and `num_steps` and waits for a step count. Capture
locally, then analyze the finished traces.

## Backend

```bash
export OMNI_PROFILER_BACKEND=/path/to/AI-Infra-Auto-Driven-SKILLS/skills/llm-torch-profiler-analysis
grep -q sglang_omni "$OMNI_PROFILER_BACKEND/scripts/profile_common.py" \
  || { echo "backend predates BBuf #96; omni frames will not be attributed"; exit 1; }
```

The `grep` is the whole compatibility contract. An older backend still runs and
still prints three tables, but its path allowlist knows `python/sglang/` and
`vllm/` and not `sglang_omni/`, so omni frames lose to torch frames and the
kernel table names `torch/nn/modules/linear.py` instead of the omni line that
called it - the one column this skill exists to read.

## The One Rule

**Two traces, two questions. Never conclude from one.**

| | CUDA graph | `with_stack` | Answers |
| --- | --- | --- | --- |
| `mapping` | off | on | *where* - which python line owns this kernel |
| `formal` | on (real serving config) | off | *how much* - the time a user pays |

A python stack exists only when the launch is a real python call, so graph-on can
never name your code, and graph-off timings are not the ones you ship. Take
location from `mapping`, every number from `formal`, and conclude only when a
kernel is heavy in `formal` *and* attributed in `mapping`.

## Real H200 Validation

An "iter" is one call of the workload body, so each music3 row is one 200-frame
chunk of 30 DiT steps:

| trace | kernels/iter | GPU ms/iter | wall ms/iter | busy % |
| --- | ---: | ---: | ---: | ---: |
| code2wav `mapping` (eager) | 836 | 5.77 | 19.96 | 28.9 |
| code2wav `formal` (graph on) | 906 | 5.94 | **6.43** | **92.3** |
| whisper encoder `mapping` (eager) | 399 | 29.07 | 30.17 | 96.3 |
| whisper encoder `formal` (graph on) | 399 | 29.03 | 29.74 | 97.6 |
| music3 DiT `mapping` (eager) | 27844 | 524.87 | 875.80 | 59.9 |
| music3 DiT `formal` (graph on) | 29044 | 527.08 | **553.35** | **95.3** |

Three workloads, three regimes, and a single trace misleads in two of them:

- code2wav's `mapping` trace says "launch-bound, add CUDA graphs"; production
  has had them for months, 20 -> 6.4 ms. `formal` says 906 kernels per window at
  median 2.3 us and 92% busy, so the work left is fusion, not launches.
- whisper's two traces agree within 1% because its kernels are large. Luck, not
  a rule.
- music3's GPU time is identical either way, yet graphs still buy 322 ms of wall
  per chunk: a 30-step serial loop pays launch overhead *around* heavy work, so
  "busy 60%" is a gap problem and the top kernel rows are a compute problem at
  the same time.

## Main Flows

### 1. Capture the pair

Run from the repo root, in the environment that serves omni, where `sglang_omni`
and `sglang` are both importable:

```bash
python3 my_workload.py --output-dir .profiling-runs/<run>/
```

`my_workload.py` is one throwaway script per stage, roughly 100 lines with
argparse, not committed. It builds the module, builds one realistic input, and
hands `capture_pair` two callables running *the same work* eager and in the real
serving config:

```python
from pathlib import Path
import sys

sys.path.insert(0, ".claude/skills/omni-gpu-deep-dive/scripts")
from omni_trace_pair import capture_pair

capture_pair(
    output_dir=Path(args.output_dir),
    mapping_body=lambda: encoder(features),      # eager, no graph, no compile
    formal_body=lambda: runner.run(features),    # what production runs
    iters=args.iters,
    warmup=args.warmup,
)
```

`capture_pair` writes `<output-dir>/mapping` and `<output-dir>/formal`, sets
`SGLANG_TORCH_PROFILER_WITH_STACK` per side, waits for `TorchProfiler`'s
background gzip, and gates each trace. It calls `torch.cuda.synchronize()` and
expects gzipped traces, so it is CUDA-only; `TorchNPUProfiler` returns an
uncompressed path.

Rules for the body:

- **Random weights are fine.** Attribution and kernel shapes follow the module
  graph and the input shape, not the values.
- **Capture and compile *before* `capture_pair`.** One-time work inside the
  profiled window is what the gate exists to reject.
- **One realistic shape per run.** Omni stages are bucketed; a shape nobody
  serves produces a kernel mix nobody pays for.
- **Same work on both sides.** If `formal_body` covers less than `mapping_body`,
  the shares are not comparable.

To capture from a running omni server instead, the mapping-side equivalent of
eager is `<stage>.engine.disable_cuda_graph = true` in the stage config plus
`SGLANG_TORCH_PROFILER_WITH_STACK=1`, and the gates below are then yours to
enforce.

### 2. Analyze the pair

```bash
python3 "$OMNI_PROFILER_BACKEND/scripts/analyze_llm_torch_profile.py" \
  --framework sglang \
  --mapping-input .profiling-runs/<run>/mapping \
  --formal-input  .profiling-runs/<run>/formal \
  --output-dir    .profiling-runs/<run>/report
```

Every backend flag applies, including `--kernel-table-limit`, `--pid-substring`,
`--merge-profiles`, and single-trace `--input`. Nothing under `.profiling-runs/`
is committed.

## Gates

**Before** - `capture_pair` warms up, then refuses any trace containing a Dynamo
or Inductor compile or a CUDA graph capture (`assert_steady_state`). One-time
cost charged to a steady-state kernel is the most common way a run reaches a
confident wrong answer. Warm every shape bucket until the gate passes; never
subtract the cost afterwards.

The gate rejects compiling and capturing, not *compiled* or *captured
execution*: `cudaGraphLaunch` is what a healthy formal trace is full of, and
`is_torchdynamo_compiling` is a predicate every HF forward calls. If you widen
`_COMPILE_MARKERS`, keep that distinction or the gate will block clean runs.

**After** - accept a change on the `formal` config only. Mapping-trace deltas
prove nothing about serving, because the graph replaces the launch path the
mapping trace measured.

- Correctness before speed: fix the input and measure the *base's own*
  run-to-run output variance first, or you cannot tell a regression from
  nondeterminism the base already had.
- Then performance, graph-on: **A/A** first, base against base, for the noise
  floor, then a **paired A/B in both orders**. If the orders disagree, you
  measured drift - warm-up, clocks, other tenants - not your change.

## Reading The Report

- **Kernel table, "Python location"** is the answer to "which code". Trust it.
  Multiple sites with shares mean one kernel shape is reached from several call
  sites, such as a GEMM used by `fc1`/`fc2` and by `qkv_proj`. That is
  information, not noise.
- **`transformers/models/...` there is a real answer, not a failure.** Where a
  stage's compute lives in a vendored HF module and omni only wraps it, as in
  Code2Wav where omni contributes the graph runner and scheduler and
  transformers every kernel, the transformers line owns the kernel. Same for
  `torchaudio/`. What you cannot act on is a *torch runtime* frame such as
  `torch/nn/modules/linear.py` or `torch/nn/functional.py`: the mapping trace
  lost the caller, which is a gap and not a finding.
- **Overlap table's "Python scope" is a majority vote**, not the kernel table's
  top site: it attributes each launch by time window and reports the most common
  site, so a kernel split `:123` 77% / `:93` 23% can show `:93`. For the line to
  edit, use the kernel table.
- **Fuse table's "Candidate fused Python path"** cites an LLM-oriented catalog of
  upstream `python/sglang/srt/...` paths. The pattern match can be real; the
  destination usually is not omni's.

## Omni-Specific Reality

The LLM north star, "tensor cores never idle", is wrong for most omni stages:

- Feature extraction and vocoder stages are **memory-bound**: conv1d, STFT,
  depthwise, resample. The fix is fusion or fewer passes, not occupancy.
- Streaming vocoder windows are **launch-bound**; the kernel table looks flat and
  cheap while wall clock is launch count and gaps. Read the overlap table's gaps,
  and consider CUDA graphs or coalescing.
- A stage is a **pipeline process**, preprocess / encoder / LLM / vocoder / post,
  not prefill/decode. Profile the slow stage in isolation first; for cross-stage
  handoff use `sglang_omni/profiler/views.py`, not this skill.
- **After `torch.compile`** kernels become `triton_poi_fused_*` with stacks in
  generated code. For the mapping trace, disable compile as well as graphs.

## Where This Sits

Stage 2 of three, and worth little without the other two.

1. **Triage** - find the *stage* from the request event timeline, before any
   profiler. Read `py-spy` **per thread**, not aggregated: an omni stage's cost
   is usually one thread's and the aggregate hides which. For TTS, measure cold
   reference and hot reference separately - different workloads, and a mean over
   both describes neither.
2. **GPU deep dive** - this skill: one stage, one shape, the trace pair, the
   kernel table's python location. Escalate to `nsys` for SM headroom only when
   the report leaves no clear change point.
3. **Validation** - the **After** gate above.

`model-profiling` owns stage 1 and the record of the outcome: it plans the run,
gets human confirmation, and tracks findings. Use it first, and go back to it
with the answer.

## Output Contract

Return:

- the mapping and formal trace paths, and the backend path used
- kernel table, overlap-opportunity table, fuse-pattern table
- the `sglang_omni/...:<line>` locations the kernel table attributed, or an
  explicit note that attribution landed on torch runtime frames
- one short summary of what dominates the stage, with the number from `formal`
- which regime the stage is in: compute-bound, memory-bound, or launch-bound
