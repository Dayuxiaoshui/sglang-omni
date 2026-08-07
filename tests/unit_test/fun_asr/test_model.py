# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from sglang_omni.models.fun_asr.sglang_model import (
    FunAsrNanoAdaptor,
    FunAsrNanoAudioEncoder,
    FunAsrNanoForConditionalGeneration,
    FunAsrNanoFSMN,
    MultiHeadedAttentionSANM,
    _SHARED_QKV_BIAS,
    _SHARED_QKV_WEIGHT,
    _enable_shared_qkv,
    _fused_qkv_project,
    _sanm_mask_from_lengths,
)
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)


def test_fun_asr_audio_modules_match_current_checkpoint_parameter_names() -> None:
    encoder = FunAsrNanoAudioEncoder(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
    )
    encoder_names = set(dict(encoder.named_parameters()))

    assert "stem.self_attn.q_proj.weight" in encoder_names
    assert "stem.self_attn.k_proj.weight" in encoder_names
    assert "stem.self_attn.v_proj.weight" in encoder_names
    assert "stem.self_attn.out_proj.weight" in encoder_names
    assert "stem.fsmn.conv.weight" in encoder_names
    assert "stem.fc1.weight" in encoder_names
    assert "layers.0.self_attn_layer_norm.weight" in encoder_names
    assert "layers.0.final_layer_norm.weight" in encoder_names
    assert "layer_norm.weight" in encoder_names
    assert "timestamp_prediction_layers.0.fc2.weight" in encoder_names
    assert "timestamp_prediction_layer_norm.weight" in encoder_names

    projector = FunAsrNanoAdaptor(
        encoder_dim=8,
        llm_dim=8,
        ffn_dim=16,
        num_layers=1,
        attention_heads=2,
    )
    projector_names = set(dict(projector.named_parameters()))

    assert "linear_1.weight" in projector_names
    assert "linear_2.weight" in projector_names
    assert "blocks.0.self_attn.q_proj.weight" in projector_names
    assert "blocks.0.self_attn_layer_norm.weight" in projector_names
    assert "blocks.0.fc1.weight" in projector_names
    assert "blocks.0.final_layer_norm.weight" in projector_names


def _weight_loader_target() -> FunAsrNanoForConditionalGeneration:
    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=False)
    )
    model.audio_tower = nn.Module()
    model.audio_tower.layer_norm = nn.LayerNorm(2)
    model.multi_modal_projector = nn.Module()
    model.multi_modal_projector.linear_1 = nn.Linear(2, 2)
    return model


def test_fun_asr_weight_loader_loads_current_audio_prefixes() -> None:
    model = _weight_loader_target()
    expected = torch.tensor([2.0, 3.0])

    model.load_weights([("model.audio_tower.layer_norm.weight", expected.clone())])

    assert torch.equal(model.audio_tower.layer_norm.weight, expected)


def test_fun_asr_weight_loader_rejects_unknown_audio_weights() -> None:
    model = _weight_loader_target()

    with pytest.raises(ValueError, match=r"model\.audio_tower\.missing\.weight"):
        model.load_weights([("model.audio_tower.missing.weight", torch.ones(2))])


def test_fun_asr_audio_feature_shape() -> None:
    class _IdentityTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(
            self, value: torch.Tensor, mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            return value

    class _IdentityProjector(nn.Module):
        def forward(
            self, value: torch.Tensor, mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            return value

    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.audio_tower = _IdentityTower()
    model.multi_modal_projector = _IdentityProjector()
    item = SimpleNamespace(
        feature=torch.arange(68, dtype=torch.float32).reshape(1, 4, 17),
        feature_attention_mask=torch.ones(1, 17, dtype=torch.long),
    )

    embedding = model.get_audio_feature([item])

    assert embedding.shape == (3, 4)


def _tiny_audio_mm_model() -> FunAsrNanoForConditionalGeneration:
    """Encoder+adaptor only (no LLM) for get_audio_feature unit tests."""
    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.audio_tower = FunAsrNanoAudioEncoder(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        activation_dropout_rate=0.0,
    )
    model.multi_modal_projector = FunAsrNanoAdaptor(
        encoder_dim=8,
        llm_dim=8,
        ffn_dim=16,
        num_layers=1,
        attention_heads=2,
        dropout_rate=0.0,
    )
    model.eval()
    return model


def _audio_item(
    feature: torch.Tensor, length: int, *, hash_id: int
) -> MultimodalDataItem:
    # feature: [1, D, T]; mask marks the first ``length`` frames valid.
    t = feature.shape[-1]
    mask = torch.zeros((1, t), dtype=torch.long)
    mask[0, :length] = 1
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=hash_id,
        feature=feature,
        model_specific_data={"feature_attention_mask": mask},
    )


def test_sanm_attention_mask_blocks_pad_keys() -> None:
    torch.manual_seed(0)
    attn = MultiHeadedAttentionSANM(n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0)
    attn.eval()
    x = torch.randn(1, 4, 8).clone()
    # note (guozhihao): corrupt the pad frame so missed key-masking would diverge.
    x[0, 3] = 100.0
    mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]])
    x_valid = x[:, :3].contiguous()

    with torch.no_grad():
        out_masked, v_masked = attn(x, mask)
        out_valid, v_valid = attn(x_valid, mask=None)

    assert torch.allclose(out_masked[:, :3], out_valid, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v_masked[:, :3], v_valid, atol=1e-5, rtol=1e-5)


def test_sanm_fused_qkv_matches_separate_projections() -> None:
    torch.manual_seed(4)
    attn = MultiHeadedAttentionSANM(n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0)
    attn.eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        out, v = attn(x, mask=None)
        v_ref = attn.v_proj(x)
        q_ref = attn.q_proj(x)
        k_ref = attn.k_proj(x)
    assert torch.allclose(v, v_ref, atol=1e-5, rtol=1e-5)
    # Fused path must still produce a usable attention output.
    assert out.shape == x.shape
    assert q_ref.shape == k_ref.shape == v_ref.shape


def test_shared_qkv_preserves_checkpoint_names_and_storage() -> None:
    torch.manual_seed(6)
    attn = MultiHeadedAttentionSANM(n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0)
    qkv_parameter_names = {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_proj.bias",
        "k_proj.bias",
        "v_proj.bias",
    }
    original = {
        name: parameter.detach().clone()
        for name, parameter in attn.named_parameters()
        if name in qkv_parameter_names
    }

    assert _enable_shared_qkv(attn) == 1
    assert qkv_parameter_names <= set(dict(attn.named_parameters()))
    state_keys = set(attn.state_dict())
    assert _SHARED_QKV_WEIGHT not in state_keys
    assert _SHARED_QKV_BIAS not in state_keys

    packed_weight = getattr(attn.q_proj, _SHARED_QKV_WEIGHT)
    packed_bias = getattr(attn.q_proj, _SHARED_QKV_BIAS)
    assert packed_weight.data_ptr() == attn.q_proj.weight.data_ptr()
    projection_bytes = (
        attn.q_proj.out_features
        * attn.q_proj.in_features
        * packed_weight.element_size()
    )
    assert packed_weight.data_ptr() + projection_bytes == attn.k_proj.weight.data_ptr()
    assert (
        packed_weight.data_ptr() + 2 * projection_bytes == attn.v_proj.weight.data_ptr()
    )
    assert torch.equal(attn.q_proj.weight, original["q_proj.weight"])
    assert torch.equal(attn.k_proj.weight, original["k_proj.weight"])
    assert torch.equal(attn.v_proj.weight, original["v_proj.weight"])
    assert packed_bias.data_ptr() == attn.q_proj.bias.data_ptr()


def test_shared_qkv_inference_projection_and_weight_updates() -> None:
    torch.manual_seed(7)
    attn = MultiHeadedAttentionSANM(n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0)
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        expected = tuple(
            projection(x) for projection in (attn.q_proj, attn.k_proj, attn.v_proj)
        )
        _enable_shared_qkv(attn)
        actual = _fused_qkv_project(x, attn.q_proj, attn.k_proj, attn.v_proj)
        assert all(
            torch.equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )

        packed_weight = getattr(attn.q_proj, _SHARED_QKV_WEIGHT)
        packed_bias = getattr(attn.q_proj, _SHARED_QKV_BIAS)
        packed_ptr = packed_weight.data_ptr()
        attn.q_proj.weight.copy_(torch.full_like(attn.q_proj.weight, 3.0))
        attn.k_proj.bias.copy_(torch.full_like(attn.k_proj.bias, 4.0))

    assert packed_weight.data_ptr() == packed_ptr
    assert torch.equal(packed_weight[: attn.q_proj.out_features], attn.q_proj.weight)
    assert torch.equal(
        packed_bias[attn.q_proj.out_features : 2 * attn.q_proj.out_features],
        attn.k_proj.bias,
    )


def test_shared_qkv_project_load_weights_updates_packed_storage() -> None:
    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=False)
    )
    model.audio_tower = nn.Module()
    model.audio_tower.attention = MultiHeadedAttentionSANM(
        n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0
    )
    _enable_shared_qkv(model.audio_tower)
    model.multi_modal_projector = nn.Module()

    replacement = torch.full_like(model.audio_tower.attention.q_proj.weight, 5.0)
    model.load_weights(
        [
            (
                "model.audio_tower.attention.q_proj.weight",
                replacement,
            )
        ]
    )

    packed_weight = getattr(model.audio_tower.attention.q_proj, _SHARED_QKV_WEIGHT)
    assert torch.equal(
        packed_weight[: model.audio_tower.attention.q_proj.out_features], replacement
    )


def test_shared_qkv_load_state_dict_assign_rebuilds_aliases() -> None:
    attn = MultiHeadedAttentionSANM(n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0)
    _enable_shared_qkv(attn)
    state_dict = {
        name: tensor.detach().clone() for name, tensor in attn.state_dict().items()
    }
    state_dict["q_proj.weight"].fill_(6.0)

    attn.load_state_dict(state_dict, assign=True)

    packed_weight = getattr(attn.q_proj, _SHARED_QKV_WEIGHT)
    assert packed_weight.data_ptr() == attn.q_proj.weight.data_ptr()
    assert torch.equal(
        packed_weight[: attn.q_proj.out_features], state_dict["q_proj.weight"]
    )
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        actual = _fused_qkv_project(x, attn.q_proj, attn.k_proj, attn.v_proj)
        expected = tuple(
            projection(x) for projection in (attn.q_proj, attn.k_proj, attn.v_proj)
        )
    assert all(
        torch.equal(left, right) for left, right in zip(actual, expected, strict=True)
    )


def test_shared_qkv_deepcopy_rebuilds_aliases() -> None:
    attn = MultiHeadedAttentionSANM(n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0)
    _enable_shared_qkv(attn)

    cloned = copy.deepcopy(attn)

    packed_weight = getattr(cloned.q_proj, _SHARED_QKV_WEIGHT)
    assert packed_weight.data_ptr() == cloned.q_proj.weight.data_ptr()
    with torch.no_grad():
        cloned.k_proj.weight.fill_(7.0)
    start = cloned.q_proj.out_features
    assert torch.equal(
        packed_weight[start : start + cloned.k_proj.out_features],
        cloned.k_proj.weight,
    )


def test_shared_qkv_keeps_grad_enabled_path() -> None:
    torch.manual_seed(8)
    attn = MultiHeadedAttentionSANM(n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0)
    _enable_shared_qkv(attn)
    x = torch.randn(2, 5, 8, requires_grad=True)

    q, k, v = _fused_qkv_project(x, attn.q_proj, attn.k_proj, attn.v_proj)
    (q.square().mean() + k.square().mean() + v.square().mean()).backward()

    assert attn.q_proj.weight.grad is not None
    assert attn.k_proj.weight.grad is not None
    assert attn.v_proj.weight.grad is not None
    assert all(
        torch.count_nonzero(parameter.grad).item() > 0
        for parameter in (
            attn.q_proj.weight,
            attn.k_proj.weight,
            attn.v_proj.weight,
        )
    )


def test_shared_qkv_rebuilds_aliases_after_dtype_conversion() -> None:
    attn = MultiHeadedAttentionSANM(n_head=2, in_feat=8, n_feat=8, dropout_rate=0.0)
    _enable_shared_qkv(attn)
    with torch.inference_mode():
        attn.to(dtype=torch.float64)

    packed_weight = getattr(attn.q_proj, _SHARED_QKV_WEIGHT)
    assert packed_weight.dtype == torch.float64
    assert not torch.is_inference(packed_weight)
    assert not torch.is_inference(attn.q_proj.weight)
    assert packed_weight.data_ptr() == attn.q_proj.weight.data_ptr()
    projection_bytes = (
        attn.q_proj.out_features
        * attn.q_proj.in_features
        * packed_weight.element_size()
    )
    assert packed_weight.data_ptr() + projection_bytes == attn.k_proj.weight.data_ptr()

    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    q, k, v = _fused_qkv_project(x, attn.q_proj, attn.k_proj, attn.v_proj)
    (q.square().mean() + k.square().mean() + v.square().mean()).backward()
    assert attn.q_proj.weight.grad is not None
    assert attn.k_proj.weight.grad is not None
    assert attn.v_proj.weight.grad is not None


def test_fun_asr_shared_qkv_config_gate(monkeypatch) -> None:
    import sglang_omni.models.fun_asr.sglang_model as fun_asr_model

    class _FakeAudioTower(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.attention = MultiHeadedAttentionSANM(2, 8, 8, 0.0)

    class _FakeProjector(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.attention = fun_asr_model.MultiHeadedAttention(2, 8, 0.0)

    class _FakeLanguageModel(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    monkeypatch.setattr(fun_asr_model, "FunAsrNanoAudioEncoder", _FakeAudioTower)
    monkeypatch.setattr(fun_asr_model, "FunAsrNanoAdaptor", _FakeProjector)
    monkeypatch.setattr(fun_asr_model, "Qwen3ForCausalLM", _FakeLanguageModel)

    config = SimpleNamespace(
        encoder_config=SimpleNamespace(
            input_size=8,
            d_model=8,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            encoder_layers=1,
            num_timestamp_prediction_blocks=0,
            kernel_size=3,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            activation_function="relu",
        ),
        text_config=SimpleNamespace(hidden_size=8),
        adaptor_intermediate_size=16,
        adaptor_num_hidden_layers=1,
        adaptor_num_attention_heads=2,
        activation_function="relu",
    )

    disabled = FunAsrNanoForConditionalGeneration(config)
    assert not hasattr(disabled.audio_tower.attention.q_proj, _SHARED_QKV_WEIGHT)
    assert not hasattr(
        disabled.multi_modal_projector.attention.q_proj, _SHARED_QKV_WEIGHT
    )

    config.enable_fun_asr_shared_qkv = True
    enabled = FunAsrNanoForConditionalGeneration(config)
    assert hasattr(enabled.audio_tower.attention.q_proj, _SHARED_QKV_WEIGHT)
    assert hasattr(enabled.multi_modal_projector.attention.q_proj, _SHARED_QKV_WEIGHT)


def test_encoder_layer_runs_attention_once() -> None:
    torch.manual_seed(5)
    layer = FunAsrNanoAudioEncoder(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=1,
        tp_blocks=0,
        kernel_size=3,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        activation_dropout_rate=0.0,
    ).stem
    layer.eval()
    calls = {"attn": 0}
    original = layer.self_attn.forward

    def _counting_attn(x, mask=None):
        calls["attn"] += 1
        return original(x, mask)

    layer.self_attn.forward = _counting_attn  # type: ignore[method-assign]
    with torch.no_grad():
        layer(torch.randn(1, 6, 8), mask=None)
    assert calls["attn"] == 1


def test_fsmn_mask_zeros_pad_and_matches_unpadded() -> None:
    torch.manual_seed(1)
    fsmn = FunAsrNanoFSMN(size=4, kernel_size=3, dropout_rate=0.0)
    fsmn.eval()
    valid = torch.randn(1, 5, 4)
    padded = torch.zeros(1, 8, 4)
    padded[:, :5] = valid
    padded[:, 5:] = 7.0
    mask = _sanm_mask_from_lengths(
        torch.tensor([5]), 8, dtype=valid.dtype, device=valid.device
    )

    with torch.no_grad():
        out_serial = fsmn(valid, mask=None)
        out_batched = fsmn(padded, mask=mask)

    assert torch.allclose(out_serial, out_batched[:, :5], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_batched[:, 5:], torch.zeros_like(out_batched[:, 5:]))


def test_get_audio_feature_batched_matches_serial() -> None:
    torch.manual_seed(2)
    model = _tiny_audio_mm_model()

    lengths = [5, 12, 8]
    items = []
    for i, length in enumerate(lengths):
        feat = torch.randn(1, 8, length)
        items.append(_audio_item(feat, length, hash_id=i + 1))

    with torch.no_grad():
        batched = model.get_audio_feature(items)
        serial_parts = [model.get_audio_feature([item]) for item in items]
        serial = torch.cat(serial_parts, dim=0)

    expected_tokens = sum(
        max(fun_asr_low_frame_rate_length(length), 1) for length in lengths
    )
    assert batched.shape == (expected_tokens, 8)
    assert serial.shape == batched.shape
    assert torch.allclose(batched, serial, atol=1e-5, rtol=1e-5)


def test_get_audio_feature_batched_matches_serial_with_pre_padded_features() -> None:
    """Right-padded features must use the mask length, not T."""
    torch.manual_seed(3)
    model = _tiny_audio_mm_model()

    lengths = [6, 10]
    t_max = 10
    items = []
    for i, length in enumerate(lengths):
        feat = torch.zeros(1, 8, t_max)
        feat[:, :, :length] = torch.randn(1, 8, length)
        # note (guozhihao): garbage in the pad region catches a missing mask.
        if length < t_max:
            feat[:, :, length:] = 50.0
        items.append(_audio_item(feat, length, hash_id=100 + i))

    with torch.no_grad():
        batched = model.get_audio_feature(items)
        serial = torch.cat([model.get_audio_feature([item]) for item in items], dim=0)

    assert batched.shape == serial.shape
    assert torch.allclose(batched, serial, atol=1e-5, rtol=1e-5)


def test_get_audio_feature_single_item_output_length() -> None:
    model = _tiny_audio_mm_model()
    length = 16
    item = _audio_item(torch.randn(1, 8, length), length, hash_id=7)

    with torch.no_grad():
        out = model.get_audio_feature([item])

    assert out.shape == (fun_asr_low_frame_rate_length(length), 8)


def test_get_audio_feature_rejects_empty_items() -> None:
    model = _tiny_audio_mm_model()
    with pytest.raises(ValueError, match="at least one audio item"):
        model.get_audio_feature([])


def test_get_audio_feature_rejects_missing_feature() -> None:
    model = _tiny_audio_mm_model()
    item = MultimodalDataItem(modality=Modality.AUDIO, hash=1, feature=None)
    with pytest.raises(ValueError, match="missing feature"):
        model.get_audio_feature([item])


def test_get_audio_feature_rejects_non_singleton_feature_batch() -> None:
    model = _tiny_audio_mm_model()
    item = MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=2,
        feature=torch.randn(2, 8, 4),
        model_specific_data={
            "feature_attention_mask": torch.ones((2, 4), dtype=torch.long),
        },
    )
    with pytest.raises(ValueError, match=r"\[1, input_size, T\]"):
        model.get_audio_feature([item])
