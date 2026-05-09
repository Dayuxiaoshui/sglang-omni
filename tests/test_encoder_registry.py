# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``sglang_omni.encoders.registry``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sglang_omni.encoders.registry import (
    EncoderSpec,
    _REGISTRY,
    get_encoder_spec,
    list_encoder_names,
    register_encoder,
)


@dataclass
class _StubAdapter:
    stage_name: str

    def build_request(self, payload):  # pragma: no cover - never called
        return None

    def apply_result(self, payload, result):  # pragma: no cover - never called
        return payload


@pytest.fixture
def isolated_registry():
    """Snapshot+restore the global registry around each test."""
    snapshot = dict(_REGISTRY)
    try:
        _REGISTRY.clear()
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


def _make_spec(name: str) -> EncoderSpec:
    return EncoderSpec(name=name, adapter_factory=_StubAdapter)


def test_register_then_lookup(isolated_registry) -> None:
    spec = _make_spec("encoder-a")
    register_encoder(spec)
    assert get_encoder_spec("encoder-a") is spec
    assert "encoder-a" in list_encoder_names()


def test_double_register_with_same_spec_is_noop(isolated_registry) -> None:
    spec = _make_spec("encoder-a")
    register_encoder(spec)
    register_encoder(spec)  # should not raise


def test_register_conflict_raises(isolated_registry) -> None:
    register_encoder(_make_spec("encoder-a"))
    with pytest.raises(ValueError, match="already registered"):
        register_encoder(_make_spec("encoder-a"))


def test_unknown_encoder_lookup_raises(isolated_registry) -> None:
    with pytest.raises(KeyError, match="unknown encoder"):
        get_encoder_spec("nope")
