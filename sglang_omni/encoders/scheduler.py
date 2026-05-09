# SPDX-License-Identifier: Apache-2.0
"""Encoder scheduler — :class:`Engine` that drives an encoder backend.

Encapsulates two responsibilities the existing single-pass encoder path
left implicit:

1. **Backend ownership.** The scheduler holds the
   :class:`EncoderBackend` and feeds it through an :class:`OmniEngine`
   configured with the standard ``EncoderBatchPlanner`` /
   ``EncoderInputPreparer`` / ``EncoderOutputProcessor`` triple. This
   keeps batching + caching identical to the legacy
   ``create_single_pass_engine`` path.

2. **TP coordination.** When ``tp_size > 1`` the scheduler is the only
   place that knows about leader vs. follower ranks. ``start()`` brings
   up the parallel state, ``stop()`` tears it down, and the actual
   broadcast/gather of ``model_inputs`` is still TODO and will land in a
   follow-up PR (see GitHub issue #375). The leader-rank ``tp_size==1``
   path is fully functional today and should be a drop-in replacement
   for the existing encoder path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn

from sglang_omni.engines.base import Engine
from sglang_omni.engines.omni import create_single_pass_engine
from sglang_omni.engines.omni.types import SchedulerRequest

from .backend import EncoderBackend, SGLangEncoderBackend

logger = logging.getLogger(__name__)


class EncoderScheduler(Engine):
    """:class:`Engine` that drives an :class:`EncoderBackend`.

    For ``tp_size == 1`` this is a thin wrapper around an
    :class:`OmniEngine` configured for single-pass encoding. For
    ``tp_size > 1`` the scheduler additionally owns the TP process
    group's leader/follower coordination — currently scaffolded; see
    ``_ensure_tp_initialized`` and ``stop`` for the seams.

    Args:
        backend: The encoder backend. Must be an :class:`nn.Module`
            because :class:`OmniEngine.model_runner` calls it with
            ``**model_inputs``.
        device: Device the leader rank executes on.
        tp_size: World size of the TP group for this encoder (1 = no TP).
        max_batch_size: Forwarded to ``EncoderBatchPlanner``.
        use_cache: Forwarded to ``OmniEngine`` (encoder output cache).
        cache_size: Forwarded to ``OmniEngine``.
        cache_max_bytes: Forwarded to ``OmniEngine``.
        cache_device: Forwarded to ``OmniEngine``.
        request_cost_fn: Forwarded to ``EncoderBatchPlanner``.
        max_batch_cost: Forwarded to ``EncoderBatchPlanner``.
    """

    def __init__(
        self,
        *,
        backend: EncoderBackend,
        device: str | torch.device = "cuda",
        tp_size: int = 1,
        max_batch_size: int = 32,
        use_cache: bool = False,
        cache_size: int | None = None,
        cache_max_bytes: int | None = None,
        cache_device: torch.device | str | None = None,
        request_cost_fn: Callable[[SchedulerRequest], int] | None = None,
        max_batch_cost: int | None = None,
    ) -> None:
        if not isinstance(backend, nn.Module):
            raise TypeError(
                "EncoderBackend must be an nn.Module so it slots into "
                "OmniEngine.model_runner.model"
            )
        self._backend = backend
        self._device = torch.device(device)
        self._tp_size = int(tp_size)
        if self._tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if self._tp_size > 1:
            # Track #375 — leader/follower forward broadcast is the next
            # PR. Failing fast here avoids silent rank-divergence bugs.
            raise NotImplementedError(
                "EncoderScheduler tp_size>1 forward coordination is not "
                "implemented yet — track sglang-project/sglang-omni#375. "
                "Use tp_size=1 (LocalEncoderBackend) until the broadcast "
                "path lands."
            )

        self._engine = create_single_pass_engine(
            self._backend,
            device=str(self._device),
            use_cache=use_cache,
            cache_size=cache_size,
            cache_max_bytes=cache_max_bytes,
            cache_device=cache_device,
            max_batch_size=max_batch_size,
            request_cost_fn=request_cost_fn,
            max_batch_cost=max_batch_cost,
        )
        self._started = False

    # ------------------------------------------------------------------
    # Engine ABC
    # ------------------------------------------------------------------

    async def add_request(self, request_id: str, data: Any) -> None:
        await self._engine.add_request(request_id, data)

    async def get_result(self, request_id: str) -> Any:
        return await self._engine.get_result(request_id)

    async def abort(self, request_id: str) -> None:
        await self._engine.abort(request_id)

    async def start(self) -> None:
        if self._started:
            return
        self._ensure_tp_initialized()
        if isinstance(self._backend, SGLangEncoderBackend):
            self._backend.load()
        await self._engine.start()
        self._started = True
        logger.info(
            "EncoderScheduler started (tp_size=%d, backend=%s)",
            self._tp_size,
            type(self._backend).__name__,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        await self._engine.stop()
        self._started = False
        logger.info("EncoderScheduler stopped")

    # ------------------------------------------------------------------
    # TP setup (leader rank only)
    # ------------------------------------------------------------------

    def _ensure_tp_initialized(self) -> None:
        """Initialise sglang's parallel state for this rank.

        For ``tp_size == 1`` this is currently a no-op because both
        :class:`LocalEncoderBackend` and :class:`SGLangEncoderBackend`
        leader-rank load paths can run without ``initialize_model_parallel``
        when the underlying module does not depend on the parallel group
        — but follow-up work needs to flip that on once
        :class:`SGLangEncoderBackend` is exercised against real TP layers.
        Centralising it here keeps the policy in one place.
        """
        if self._tp_size <= 1:
            return
        # Reachable only once the ``tp_size > 1`` guard in __init__ is
        # relaxed. Left as a single-source-of-truth seam.
        raise NotImplementedError(
            "TP parallel state initialization for encoders is not yet "
            "wired (issue #375)"
        )
