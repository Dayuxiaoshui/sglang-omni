# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni Code2Wav workload -- the launch-bound counterpart to whisper.

One streaming vocoder window is tiny ([1, 16, 35] codes by default: chunk 10 +
left context 25), so wall clock is dominated by launch count and gaps rather
than by any one kernel. That is the shape production actually replays, so it is
the shape to profile.

    python workloads/qwen3_omni_code2wav.py --output-dir .profiling-runs/code2wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omni_trace_pair import capture_pair  # noqa: E402

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (  # noqa: E402
    Code2WavCudaGraphRunner,
    GraphKey,
)

# Code2WavScheduler defaults; steady window = left_context_size + stream_chunk_size.
STREAM_CHUNK_SIZE = 10
LEFT_CONTEXT_SIZE = 25

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def build_code2wav(*, device: str, dtype: torch.dtype):
    config = Qwen3OmniMoeConfig().code2wav_config
    model = Qwen3OmniMoeCode2Wav._from_config(config)
    return model.to(device=device, dtype=dtype).eval(), config


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--frames", type=int, default=LEFT_CONTEXT_SIZE + STREAM_CHUNK_SIZE
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.3)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        parser.error("this workload needs a CUDA device")
    torch.cuda.set_device(args.device)
    model, config = build_code2wav(device=args.device, dtype=DTYPES[args.dtype])
    num_quantizers = int(config.num_quantizers)
    codes = torch.randint(
        0,
        int(config.codebook_size),
        (args.batch, num_quantizers, args.frames),
        device=args.device,
        dtype=torch.long,
    )

    runner = Code2WavCudaGraphRunner.build(
        model,
        device=args.device,
        num_quantizers=num_quantizers,
        total_gpu_memory_fraction=args.gpu_memory_fraction,
        graph_keys=(GraphKey(batch_size=args.batch, frames=args.frames),),
    )
    probe = runner.run(codes)
    if probe.execution_mode == "eager":
        raise RuntimeError(
            f"Code2Wav graph did not take this shape ({probe.fallback_reason}); "
            "the formal trace would be eager and the pairing meaningless"
        )

    capture_pair(
        output_dir=args.output_dir,
        mapping_body=lambda: model(codes),
        formal_body=lambda: runner.run(codes).output,
        iters=args.iters,
        warmup=args.warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
