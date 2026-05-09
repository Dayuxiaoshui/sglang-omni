# SPDX-License-Identifier: Apache-2.0
"""Encoder backends, adapters, and TP-aware scheduler.

Public surface:

- ``EncoderBackend`` / ``LocalEncoderBackend`` / ``SGLangEncoderBackend`` —
  unified ``forward(model_inputs) -> dict`` interface that hides whether
  the underlying encoder is a local HF module or one of sglang main's
  TP-aware multimodal towers.
- ``EncoderAdapter`` — payload <-> model conversion contract registered
  per encoder.
- ``EncoderScheduler`` — :class:`sglang_omni.engines.base.Engine`
  implementation that owns batching/cache plus, when ``tp_size>1``,
  leader/follower coordination over the TP process group.
- ``build_encoder_executor`` — constructs the
  :class:`sglang_omni.executors.EngineExecutor` that pipeline stages plug
  into.
- ``register_encoder`` / ``get_encoder_spec`` — registry of named
  encoders. Models register their adapter and the default backend spec
  here; pipeline factories look the encoder up by name.
"""

from sglang_omni.encoders.adapter import EncoderAdapter
from sglang_omni.encoders.backend import (
    EncoderBackend,
    LocalEncoderBackend,
    SGLangEncoderBackend,
    SGLangEncoderSpec,
)
from sglang_omni.encoders.factory import build_encoder_executor
from sglang_omni.encoders.registry import (
    EncoderSpec,
    get_encoder_spec,
    list_encoder_names,
    register_encoder,
)
from sglang_omni.encoders.scheduler import EncoderScheduler

__all__ = [
    "EncoderAdapter",
    "EncoderBackend",
    "EncoderScheduler",
    "EncoderSpec",
    "LocalEncoderBackend",
    "SGLangEncoderBackend",
    "SGLangEncoderSpec",
    "build_encoder_executor",
    "get_encoder_spec",
    "list_encoder_names",
    "register_encoder",
]
