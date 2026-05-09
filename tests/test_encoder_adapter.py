# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3-Omni encoder adapter — ensure the new abstraction
preserves the legacy state I/O contract."""

from __future__ import annotations

import torch

from sglang_omni.engines.omni.runtime import EncoderRequestData
from sglang_omni.models.qwen3_omni.encoder_adapter import (
    QWEN3_OMNI_AUDIO_ENCODER,
    QWEN3_OMNI_IMAGE_ENCODER,
    Qwen3OmniEncoderAdapter,
)
from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.qwen3_omni.pipeline.next_stage import AUDIO_STAGE, IMAGE_STAGE
from sglang_omni.models.qwen3_omni.pipeline.state_io import store_state
from sglang_omni.proto import OmniRequest, StagePayload


def _make_payload(state: PipelineState) -> StagePayload:
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=None, params={}),
        data=None,
    )
    return store_state(payload, state)


def test_adapter_audio_round_trip() -> None:
    state = PipelineState()
    state.encoder_inputs[AUDIO_STAGE] = {
        "input_features": torch.zeros(1, 80, 4),
        "feature_attention_mask": torch.ones(1, 4),
    }
    payload = _make_payload(state)

    adapter = Qwen3OmniEncoderAdapter(stage_name=AUDIO_STAGE)
    request = adapter.build_request(payload)
    assert isinstance(request, EncoderRequestData)
    assert request.input_dict is state.encoder_inputs[AUDIO_STAGE]

    out = {"audio_embeds": torch.zeros(1, 4, 8)}
    payload_after = adapter.apply_result(payload, out)
    # apply_result returns a payload whose state has the result written
    from sglang_omni.models.qwen3_omni.pipeline.state_io import load_state

    state_after = load_state(payload_after)
    assert state_after.encoder_outs[AUDIO_STAGE] is out
    assert state_after.engine_outputs[AUDIO_STAGE] is out


def test_adapter_image_skip_path() -> None:
    """When the upstream marks ``_skip``, the adapter should preserve the
    pre-computed result so the encoder is bypassed."""
    state = PipelineState()
    cached = {"image_embeds": torch.zeros(1, 8)}
    state.encoder_inputs[IMAGE_STAGE] = {"_skip": True, "_result": cached}
    payload = _make_payload(state)

    adapter = Qwen3OmniEncoderAdapter(stage_name=IMAGE_STAGE)
    request = adapter.build_request(payload)
    assert request.output_dict is cached


def test_registered_names_round_trip() -> None:
    from sglang_omni.encoders.registry import get_encoder_spec

    audio_spec = get_encoder_spec(QWEN3_OMNI_AUDIO_ENCODER)
    image_spec = get_encoder_spec(QWEN3_OMNI_IMAGE_ENCODER)
    assert audio_spec.sglang_spec is not None
    assert image_spec.sglang_spec is not None
    assert audio_spec.sglang_spec.arch_name == "Qwen3OmniMoeAudioEncoder"
    assert image_spec.sglang_spec.arch_name == "Qwen3OmniMoeVisionEncoder"
