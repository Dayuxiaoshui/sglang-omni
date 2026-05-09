# SPDX-License-Identifier: Apache-2.0
"""Encoder backends.

A backend wraps the underlying encoder model and exposes a single contract
to the rest of the pipeline:

    forward(model_inputs: dict[str, Any]) -> dict[str, Any]

Two backends are supported:

- ``LocalEncoderBackend`` wraps an arbitrary ``torch.nn.Module`` (e.g. the
  HF-derived ``Qwen3OmniAudioEncoder`` / ``Qwen3OmniImageEncoder``).
  Single-process, no TP.

- ``SGLangEncoderBackend`` loads an encoder class from sglang main
  (e.g. ``sglang.srt.models.qwen3_omni_moe.Qwen3OmniMoeAudioEncoder``).
  Inherits TP via sglang's ``ColumnParallelLinear`` / ``RowParallelLinear``
  layers and the ``parallel_state`` group it sets up at process init.

Backends are ``torch.nn.Module`` subclasses so they slot directly into the
existing ``OmniEngine.model_runner.model`` position; the adapter remains
responsible for shaping inputs and consuming outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EncoderBackend(nn.Module):
    """Common interface for encoder execution backends."""

    def forward(self, **model_inputs: Any) -> dict[str, Any]:  # noqa: D401
        """Run encoder forward and return a dict of named outputs."""
        raise NotImplementedError


class LocalEncoderBackend(EncoderBackend):
    """Backend that wraps an in-process ``nn.Module`` — no TP."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module

    @property
    def module(self) -> nn.Module:
        return self._module

    def forward(self, **model_inputs: Any) -> dict[str, Any]:
        out = self._module(**model_inputs)
        if not isinstance(out, dict):
            raise TypeError(
                f"LocalEncoderBackend expected the wrapped module "
                f"{type(self._module).__name__!r} to return a dict, "
                f"got {type(out).__name__}"
            )
        return out


@dataclass(frozen=True)
class SGLangEncoderSpec:
    """Pointer into sglang main's model registry for an encoder.

    Resolution intentionally goes through callables rather than dotted
    import strings so unit tests can stub the loader without touching the
    real sglang package.
    """

    arch_name: str
    """Architecture identifier, e.g. ``"Qwen3OmniMoeAudioEncoder"``."""

    config_loader: Callable[[str], Any]
    """``(model_path) -> hf_config_for_this_encoder``."""

    module_factory: Callable[[Any], nn.Module]
    """``(encoder_config) -> nn.Module`` instance (TP-aware when sglang
    parallel state is initialized)."""

    weight_prefix: tuple[str, ...] = ()
    """Prefix(es) to strip from checkpoint weight names when loading."""


class SGLangEncoderBackend(EncoderBackend):
    """Backend that loads an encoder class from sglang main.

    The TP process-group bring-up + actual broadcast/gather of
    ``model_inputs`` across ranks is intentionally **not** handled inside
    the backend — it lives in :class:`EncoderScheduler` so the existing
    sglang-omni TP follower convention (one follower process per
    ``tp_rank > 0``) stays in one place.

    The backend itself is responsible only for:

    1. Constructing the sglang main module on the local rank.
    2. Loading weights with the registered prefix.
    3. Running the local rank's portion of forward when called.

    For ``tp_size > 1`` the underlying sglang layers require
    ``initialize_model_parallel`` to have been called on this process —
    that is the scheduler's job before it ever calls ``forward``.
    """

    def __init__(
        self,
        spec: SGLangEncoderSpec,
        model_path: str,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self._spec = spec
        self._model_path = model_path
        self._device = torch.device(device)
        self._dtype = dtype
        self._tp_rank = int(tp_rank)
        self._tp_size = int(tp_size)
        self._module: nn.Module | None = None

    @property
    def tp_size(self) -> int:
        return self._tp_size

    @property
    def tp_rank(self) -> int:
        return self._tp_rank

    @property
    def arch_name(self) -> str:
        return self._spec.arch_name

    def load(self) -> None:
        """Construct the sglang module and load weights.

        Caller must ensure ``initialize_model_parallel`` has run for
        ``tp_size > 1`` before this is invoked.
        """
        if self._module is not None:
            return

        config = self._spec.config_loader(self._model_path)
        module = self._spec.module_factory(config)
        if self._dtype is not None:
            module = module.to(dtype=self._dtype)
        module = module.to(self._device)
        module.eval()

        from sglang_omni.models.weight_loader import load_module

        load_module(
            module,
            self._model_path,
            prefix=self._spec.weight_prefix,
            dtype=self._dtype,
            device=str(self._device),
            strict=True,
        )
        self._module = module
        logger.info(
            "SGLangEncoderBackend loaded %s (tp_rank=%d, tp_size=%d)",
            self._spec.arch_name,
            self._tp_rank,
            self._tp_size,
        )

    def forward(self, **model_inputs: Any) -> dict[str, Any]:
        if self._module is None:
            raise RuntimeError(
                "SGLangEncoderBackend.forward called before load(); "
                "the EncoderScheduler should call load() during start()"
            )
        out = self._module(**model_inputs)
        if not isinstance(out, dict):
            raise TypeError(
                f"sglang encoder {self._spec.arch_name!r} expected to return a dict, "
                f"got {type(out).__name__}"
            )
        return out
