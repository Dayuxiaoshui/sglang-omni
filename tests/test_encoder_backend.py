# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``sglang_omni.encoders.backend``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sglang_omni.encoders.backend import (
    LocalEncoderBackend,
    SGLangEncoderBackend,
    SGLangEncoderSpec,
)


class _DummyEncoder(nn.Module):
    """Tiny ``nn.Module`` that mirrors the dict-returning encoder contract."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 2)

    def forward(self, *, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"embeds": self.proj(features)}


class _BadEncoder(nn.Module):
    def forward(self, *, features: torch.Tensor) -> torch.Tensor:
        return features


def test_local_backend_dispatches_kwargs() -> None:
    backend = LocalEncoderBackend(_DummyEncoder())
    out = backend.forward(features=torch.zeros(1, 4))
    assert isinstance(out, dict)
    assert out["embeds"].shape == (1, 2)


def test_local_backend_rejects_non_dict_modules() -> None:
    backend = LocalEncoderBackend(_BadEncoder())
    with pytest.raises(TypeError, match="expected the wrapped module"):
        backend.forward(features=torch.zeros(1, 4))


def test_sglang_backend_forward_before_load_is_explicit() -> None:
    spec = SGLangEncoderSpec(
        arch_name="Stub",
        config_loader=lambda _path: object(),
        module_factory=lambda _cfg: _DummyEncoder(),
    )
    backend = SGLangEncoderBackend(spec, model_path="ignored", device="cpu", tp_size=1)
    with pytest.raises(RuntimeError, match="forward called before load"):
        backend.forward(features=torch.zeros(1, 4))


def test_sglang_backend_metadata_visible() -> None:
    spec = SGLangEncoderSpec(
        arch_name="Stub",
        config_loader=lambda _path: object(),
        module_factory=lambda _cfg: _DummyEncoder(),
    )
    backend = SGLangEncoderBackend(
        spec, model_path="ignored", device="cpu", tp_size=4, tp_rank=2
    )
    assert backend.arch_name == "Stub"
    assert backend.tp_size == 4
    assert backend.tp_rank == 2
