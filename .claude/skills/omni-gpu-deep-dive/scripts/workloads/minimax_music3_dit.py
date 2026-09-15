# SPDX-License-Identifier: Apache-2.0
"""MiniMax-Music 3 flow-matching DiT workload -- a diffusion-loop stage.

One body is one full chunk generation: omni's own ``MiniMaxMusic3DIT.forward``
runs ``--steps`` sequential denoising steps over a fixed mel window, so the
per-iteration numbers in the report are per chunk, not per step. Divide by
``--steps`` to reason about a single step.

The real serving config is graph-on and compile-off: ``acoustic.py`` enables
compiled blocks only when neither cache-dit nor the breakable CUDA graph is
requested, so the formal side here is the breakable graph and nothing else.

    python workloads/minimax_music3_dit.py --output-dir .profiling-runs/music3-dit
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omni_trace_pair import capture_pair  # noqa: E402

from sglang_omni.models.minimax_music3.constants import (  # noqa: E402
    AR_CHUNK_FRAMES,
    DEFAULT_DIT_CFG_SCALE,
    DEFAULT_DIT_STEPS,
)
from sglang_omni.models.minimax_music3.dit import MiniMaxMusic3DIT  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def build_dit(*, device: str, dtype: torch.dtype, attention_backend: str) -> MiniMaxMusic3DIT:
    with torch.device("meta"):
        dit = MiniMaxMusic3DIT(compute_dtype=dtype, attention_backend=attention_backend)
    dit = dit.to_empty(device=device).to(dtype)
    with torch.no_grad():
        for param in dit.parameters():
            param.normal_(0.0, 0.02)
    return dit.eval()


@contextmanager
def eager_steps(dit: MiniMaxMusic3DIT):
    """Run the loop through its eager branch while the graph stays captured.

    ``forward`` picks eager or graph per step off ``_bcg_runner``, so the mapping
    body has to hide the captured runner rather than avoid capturing it: capture
    inside the profiled window would trip the steady-state gate.
    """
    captured = dit._bcg_runner
    dit._bcg_runner = None
    try:
        yield
    finally:
        dit._bcg_runner = captured


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--ar-frames",
        type=int,
        default=AR_CHUNK_FRAMES,
        help="AR frames per chunk; the mel window is derived from it",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_DIT_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_DIT_CFG_SCALE)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--attention-backend", default="torch_sdpa")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        parser.error("this workload needs a CUDA device")
    torch.cuda.set_device(args.device)
    dtype = DTYPES[args.dtype]
    dit = build_dit(
        device=args.device, dtype=dtype, attention_backend=args.attention_backend
    )

    mel_len = dit.aligned_mel_length(args.ar_frames)
    align = torch.randn(1, 2048, mel_len, device=args.device, dtype=dtype)
    generator = torch.Generator(device=args.device).manual_seed(0)

    if not dit.enable_breakable_cuda_graph(
        mel_len=mel_len, min_free_gb=args.min_free_gb
    ):
        raise RuntimeError(
            "breakable CUDA graph capture failed; the formal trace would be eager "
            "and the pairing meaningless"
        )

    def solve() -> torch.Tensor:
        return dit(
            align.clone(),
            generator=generator,
            num_steps=args.steps,
            cfg_scale=args.cfg_scale,
        )

    def mapping_body() -> torch.Tensor:
        with eager_steps(dit):
            return solve()

    capture_pair(
        output_dir=args.output_dir,
        mapping_body=mapping_body,
        formal_body=solve,
        iters=args.iters,
        warmup=args.warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
