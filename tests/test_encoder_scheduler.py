# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``sglang_omni.encoders.scheduler``."""

from __future__ import annotations

import asyncio

import pytest
import torch
import torch.nn as nn

from sglang_omni.encoders.backend import LocalEncoderBackend
from sglang_omni.encoders.scheduler import EncoderScheduler
from sglang_omni.engines.omni.runtime import EncoderRequestData


class _IdentityEncoder(nn.Module):
    """CPU-friendly module — returns the input under ``embeds``."""

    def forward(self, *, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"embeds": features.clone()}


def test_tp_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tp_size must be"):
        EncoderScheduler(
            backend=LocalEncoderBackend(_IdentityEncoder()),
            device="cpu",
            tp_size=0,
        )


def test_tp_size_gt_one_explicitly_unsupported() -> None:
    """Until the broadcast/gather path lands, tp_size>1 must fail loudly."""
    with pytest.raises(NotImplementedError, match="tp_size>1"):
        EncoderScheduler(
            backend=LocalEncoderBackend(_IdentityEncoder()),
            device="cpu",
            tp_size=2,
        )


def test_backend_must_be_nn_module() -> None:
    class NotAModule:
        pass

    with pytest.raises(TypeError, match="EncoderBackend must be an nn.Module"):
        EncoderScheduler(backend=NotAModule(), device="cpu")  # type: ignore[arg-type]


def test_tp1_round_trip_via_engine_abc() -> None:
    """Drive a single request through the Engine ABC on CPU."""

    async def _run() -> None:
        scheduler = EncoderScheduler(
            backend=LocalEncoderBackend(_IdentityEncoder()),
            device="cpu",
            tp_size=1,
        )
        await scheduler.start()
        try:
            features = torch.arange(6, dtype=torch.float32).reshape(1, 6)
            data = EncoderRequestData(input_dict={"features": features})
            await scheduler.add_request("req-1", data)
            result = await scheduler.get_result("req-1")
            assert torch.equal(result.output_dict["embeds"], features)
        finally:
            await scheduler.stop()

    asyncio.run(_run())
