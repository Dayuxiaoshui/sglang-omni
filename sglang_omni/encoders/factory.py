# SPDX-License-Identifier: Apache-2.0
"""Factory: turn a registered encoder + pipeline args into an EngineExecutor."""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

from sglang_omni.engines.omni.types import SchedulerRequest
from sglang_omni.executors import EngineExecutor

from .backend import EncoderBackend, LocalEncoderBackend, SGLangEncoderBackend
from .registry import EncoderSpec, get_encoder_spec
from .scheduler import EncoderScheduler

logger = logging.getLogger(__name__)


def build_encoder_executor(
    encoder_name: str,
    *,
    stage_name: str,
    model_path: str,
    local_module_factory: Callable[[], nn.Module],
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    gpu_id: int | None = None,
    nccl_port: int | None = None,
    max_batch_size: int = 32,
    use_cache: bool = False,
    cache_size: int | None = None,
    cache_max_bytes: int | None = None,
    cache_device: torch.device | str | None = None,
    request_cost_fn: Callable[[SchedulerRequest], int] | None = None,
    max_batch_cost: int | None = None,
    backend_override: EncoderBackend | None = None,
) -> EngineExecutor:
    """Build an :class:`EngineExecutor` for a registered encoder.

    Backend selection:

    - ``tp_size == 1`` and no ``backend_override``: use
      :class:`LocalEncoderBackend` wrapping ``local_module_factory()``.
      This preserves the legacy execution path bit-for-bit.
    - ``tp_size > 1``: use :class:`SGLangEncoderBackend`. The encoder
      must have a ``sglang_spec`` registered or this raises.
    - ``backend_override`` (tests / advanced wiring): use as-is, ignoring
      the rules above.

    Args:
        encoder_name: Name registered via :func:`register_encoder`.
        stage_name: Pipeline stage name; used to bind the adapter to the
            right ``state.encoder_inputs`` / ``state.encoder_outs`` slot.
        model_path: HF model path; passed to the local module factory and
            to :class:`SGLangEncoderBackend`.
        local_module_factory: Zero-argument callable returning the
            single-process ``nn.Module`` for the local backend.
        device: GPU device the leader rank executes on.
        dtype: Optional torch dtype passed through to the backend.
        tp_size: TP world size for this encoder stage.
        max_batch_size, use_cache, cache_size, cache_max_bytes,
        cache_device, request_cost_fn, max_batch_cost: forwarded to
        :class:`EncoderScheduler`.
        backend_override: For tests — supply a pre-built backend and skip
        the selection rules above.
    """
    spec = get_encoder_spec(encoder_name)
    backend = backend_override or _select_backend(
        spec=spec,
        model_path=model_path,
        local_module_factory=local_module_factory,
        device=device,
        dtype=dtype,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )

    scheduler = EncoderScheduler(
        backend=backend,
        device=device,
        tp_size=tp_size,
        tp_rank=tp_rank,
        gpu_id=gpu_id,
        nccl_port=nccl_port,
        max_batch_size=max_batch_size,
        use_cache=use_cache,
        cache_size=cache_size,
        cache_max_bytes=cache_max_bytes,
        cache_device=cache_device,
        request_cost_fn=request_cost_fn,
        max_batch_cost=max_batch_cost,
    )
    adapter = spec.adapter_factory(stage_name)

    return EngineExecutor(
        engine=scheduler,
        request_builder=adapter.build_request,
        result_builder=adapter.apply_result,
    )


def _select_backend(
    *,
    spec: EncoderSpec,
    model_path: str,
    local_module_factory: Callable[[], nn.Module],
    device: str | torch.device,
    dtype: torch.dtype | None,
    tp_size: int,
    tp_rank: int,
) -> EncoderBackend:
    if tp_size <= 1:
        module = local_module_factory()
        return LocalEncoderBackend(module)

    if spec.sglang_spec is None:
        raise ValueError(
            f"encoder {spec.name!r} has no sglang_spec registered, "
            f"so tp_size={tp_size} (>1) is not supported. Either set "
            "tp_size=1 to use the local backend, or register an "
            "SGLangEncoderSpec for this encoder."
        )

    return SGLangEncoderBackend(
        spec.sglang_spec,
        model_path,
        device=device,
        dtype=dtype,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
