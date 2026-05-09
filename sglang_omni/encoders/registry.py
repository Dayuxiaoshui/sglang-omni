# SPDX-License-Identifier: Apache-2.0
"""Registry of named encoders.

Each entry pairs an :class:`EncoderAdapter` factory with the optional
:class:`SGLangEncoderSpec` describing how to load the encoder via sglang
main when ``tp_size > 1`` is requested.

A model registers its encoders at import time (see
``sglang_omni/encoders/qwen3_omni.py``); pipeline factories then look the
encoder up by name without any model-specific imports leaking into
generic code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sglang_omni.encoders.adapter import EncoderAdapter
from sglang_omni.encoders.backend import SGLangEncoderSpec


@dataclass(frozen=True)
class EncoderSpec:
    """Registry entry for a named encoder."""

    name: str
    """Stable identifier used by pipeline configs and factories."""

    adapter_factory: Callable[[str], EncoderAdapter]
    """``(stage_name) -> EncoderAdapter`` — bound at construction time."""

    sglang_spec: SGLangEncoderSpec | None = None
    """Pointer into sglang main, ``None`` if the encoder has no upstream
    counterpart yet (i.e. ``tp_size > 1`` is not currently supported)."""


_REGISTRY: dict[str, EncoderSpec] = {}


def register_encoder(spec: EncoderSpec) -> None:
    """Register ``spec`` under ``spec.name``.

    Re-registering the same name is a programming error rather than a
    silent no-op so accidental double-registration is caught at import
    time.
    """
    if spec.name in _REGISTRY:
        existing = _REGISTRY[spec.name]
        if existing is spec:
            return
        raise ValueError(
            f"encoder {spec.name!r} already registered "
            f"(existing={existing!r}, new={spec!r})"
        )
    _REGISTRY[spec.name] = spec


def get_encoder_spec(name: str) -> EncoderSpec:
    """Return the registered :class:`EncoderSpec` for ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = sorted(_REGISTRY)
        raise KeyError(
            f"unknown encoder {name!r}; registered encoders: {known}"
        ) from exc


def list_encoder_names() -> list[str]:
    """Return the sorted list of registered encoder names."""
    return sorted(_REGISTRY)
