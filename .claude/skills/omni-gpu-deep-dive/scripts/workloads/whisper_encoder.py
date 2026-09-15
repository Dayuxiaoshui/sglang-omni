# SPDX-License-Identifier: Apache-2.0
"""whisper-large-v3 encoder workload -- the reference for writing a new one.

A workload's whole job is to build the module, build one realistic input, and
hand ``capture_pair`` two callables that run the same work eager and in the real
serving config. Weights are randomly initialised: attribution and kernel shapes
follow the module graph and the input shape, not the values.

    python workloads/whisper_encoder.py --output-dir .profiling-runs/whisper-enc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import WhisperConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omni_trace_pair import capture_pair  # noqa: E402

from sglang_omni.models.whisper_asr.encoder_cuda_graph import (  # noqa: E402
    WhisperEncoderCudaGraphRunner,
)
from sglang_omni.models.whisper_asr.sglang_model import WhisperEncoder  # noqa: E402

# whisper-large-v3 encoder geometry (config.json of openai/whisper-large-v3).
LARGE_V3 = dict(
    num_mel_bins=128,
    d_model=1280,
    encoder_layers=32,
    encoder_attention_heads=20,
    encoder_ffn_dim=5120,
    max_source_positions=1500,
    activation_function="gelu",
)
# WhisperFeatureExtractor.nb_max_frames for one 30s window.
INPUT_FEATURE_LEN = 3000

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def build_encoder(*, layers: int, device: str, dtype: torch.dtype) -> WhisperEncoder:
    config = WhisperConfig(**{**LARGE_V3, "encoder_layers": layers})
    with torch.device("meta"):
        encoder = WhisperEncoder(config)
    encoder = encoder.to_empty(device=device).to(dtype)
    with torch.no_grad():
        for param in encoder.parameters():
            param.normal_(0.0, 0.02)
    return encoder.eval()


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=4, help="request batch (30s windows)")
    parser.add_argument("--frames", type=int, default=INPUT_FEATURE_LEN)
    parser.add_argument("--layers", type=int, default=LARGE_V3["encoder_layers"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        parser.error("this workload needs a CUDA device")
    torch.cuda.set_device(args.device)
    dtype = DTYPES[args.dtype]
    encoder = build_encoder(layers=args.layers, device=args.device, dtype=dtype)
    features = torch.randn(
        args.batch, LARGE_V3["num_mel_bins"], args.frames, device=args.device, dtype=dtype
    )

    # Capture before profiling: a capture inside the profiled window trips the
    # steady-state gate, and rightly so.
    runner = WhisperEncoderCudaGraphRunner(
        encoder,
        num_mel_bins=LARGE_V3["num_mel_bins"],
        input_feature_len=args.frames,
    )
    runner.capture([args.batch])
    if not runner.captured_buckets:
        raise RuntimeError(
            "CUDA graph capture produced no buckets; the formal trace would be eager "
            "and the pairing meaningless"
        )

    capture_pair(
        output_dir=args.output_dir,
        mapping_body=lambda: encoder(features),
        formal_body=lambda: runner.run(features),
        iters=args.iters,
        warmup=args.warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
