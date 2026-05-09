# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni encoder adapters and registry entries.

Two encoders are registered here:

- ``"qwen3_omni_audio"`` — bound to the audio stage; delegates payload
  conversion to the existing ``engine_io.build_encoder_request`` /
  ``apply_encoder_result`` so behaviour matches the legacy path
  bit-for-bit.
- ``"qwen3_omni_image"`` — same shape, bound to the image/vision stage.

Both adapters share the same logic (the existing helpers already key on
``stage_name``); the two registry entries differ only in their
``sglang_spec`` so each can be loaded from sglang main independently
when ``tp_size > 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.encoders.adapter import EncoderAdapter
from sglang_omni.encoders.backend import SGLangEncoderSpec
from sglang_omni.encoders.registry import EncoderSpec, register_encoder
from sglang_omni.engines.omni.runtime import EncoderRequestData
from sglang_omni.models.qwen3_omni.pipeline.engine_io import (
    apply_encoder_result,
    build_encoder_request,
)
from sglang_omni.models.qwen3_omni.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload

QWEN3_OMNI_AUDIO_ENCODER = "qwen3_omni_audio"
QWEN3_OMNI_IMAGE_ENCODER = "qwen3_omni_image"


@dataclass
class Qwen3OmniEncoderAdapter:
    """Adapter for both Qwen3-Omni audio and vision stages.

    The two encoders share the request/result shape — they differ only
    in which slot of ``state.encoder_inputs`` / ``state.encoder_outs``
    they read and write, which is keyed by ``stage_name``.
    """

    stage_name: str

    def build_request(self, payload: StagePayload) -> EncoderRequestData:
        state = load_state(payload)
        return build_encoder_request(state, stage_name=self.stage_name)

    def apply_result(self, payload: StagePayload, result: object) -> StagePayload:
        state = load_state(payload)
        apply_encoder_result(state, stage_name=self.stage_name, result=result)
        return store_state(payload, state)


def _audio_config_loader(model_path: str):
    from sglang_omni.models.qwen3_omni.components.common import load_thinker_config

    thinker_cfg = load_thinker_config(model_path)
    return thinker_cfg.audio_config


def _vision_config_loader(model_path: str):
    from sglang_omni.models.qwen3_omni.components.common import load_thinker_config

    thinker_cfg = load_thinker_config(model_path)
    return thinker_cfg.vision_config


def _audio_module_factory(config):
    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    return Qwen3OmniMoeAudioEncoder(config)


def _vision_module_factory(config):
    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

    return Qwen3OmniMoeVisionEncoder(config)


_AUDIO_SGLANG_SPEC = SGLangEncoderSpec(
    arch_name="Qwen3OmniMoeAudioEncoder",
    config_loader=_audio_config_loader,
    module_factory=_audio_module_factory,
    weight_prefix=("thinker.audio_tower.", "audio_tower."),
)


_VISION_SGLANG_SPEC = SGLangEncoderSpec(
    arch_name="Qwen3OmniMoeVisionEncoder",
    config_loader=_vision_config_loader,
    module_factory=_vision_module_factory,
    weight_prefix=("thinker.visual.", "visual."),
)


def _build_audio_adapter(stage_name: str) -> EncoderAdapter:
    return Qwen3OmniEncoderAdapter(stage_name=stage_name)


def _build_image_adapter(stage_name: str) -> EncoderAdapter:
    return Qwen3OmniEncoderAdapter(stage_name=stage_name)


def register() -> None:
    """Register Qwen3-Omni encoders.

    Idempotent — safe to call from multiple import sites.
    """
    register_encoder(
        EncoderSpec(
            name=QWEN3_OMNI_AUDIO_ENCODER,
            adapter_factory=_build_audio_adapter,
            sglang_spec=_AUDIO_SGLANG_SPEC,
        )
    )
    register_encoder(
        EncoderSpec(
            name=QWEN3_OMNI_IMAGE_ENCODER,
            adapter_factory=_build_image_adapter,
            sglang_spec=_VISION_SGLANG_SPEC,
        )
    )


# Register at import time so the adapter package can be used purely via
# string lookups by pipeline configs.
register()
