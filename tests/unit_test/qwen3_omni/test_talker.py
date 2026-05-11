# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from sglang_omni_v1.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni_v1.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni_v1.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni_v1.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler


def test_qwen_talker_feedback_fifo_and_stream_done_contract() -> None:
    """Preserves Talker FIFO feedback consumption and prefetched stream-done state."""
    sched_req = SimpleNamespace(
        data=SimpleNamespace(
            pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
            pending_text_queue=deque(),
            tts_pad_embed=torch.tensor([7.0, 8.0]),
            thinker_chunks_done=False,
        )
    )

    assert (
        QwenTalkerModelRunner._take_next_decode_input_embed(
            sched_req=sched_req,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is None
    )
    sched_req.data.pending_text_queue.append(torch.tensor([20.0, 20.0]))
    assert torch.equal(
        QwenTalkerModelRunner._take_next_decode_input_embed(
            sched_req=sched_req,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        torch.tensor([21.0, 22.0]),
    )

    scheduler = object.__new__(QwenTalkerScheduler)
    req_data = SimpleNamespace(
        pending_text_queue=deque([torch.tensor([11.0, 12.0])]),
        thinker_chunks_done=True,
    )
    payload = SimpleNamespace(
        prefetched_chunks=[SimpleNamespace(data=torch.tensor([20.0, 20.0]))],
        prefetched_stream_done=True,
    )
    assert scheduler._is_request_build_ready(payload, pending_stream_done=True)
    scheduler._initialize_request_stream_state(req_data, payload)
    assert len(req_data.pending_text_queue) == 1
    assert torch.equal(req_data.pending_text_queue[0], torch.tensor([11.0, 12.0]))


def test_qwen_model_runner_and_code_predictor_tensor_contracts() -> None:
    """Preserves multimodal embed injection and code-predictor token shape."""

    class RecordingEmbed:
        num_embeddings = 10

        def __init__(self) -> None:
            self.seen: torch.Tensor | None = None

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.seen = input_ids.clone()
            return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)

    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner._embed_tokens = RecordingEmbed()
    runner._image_token_id = 5
    runner._video_token_id = 6
    runner._audio_token_id = 7
    req = SimpleNamespace(
        omni_model_inputs={
            "audio_embeds": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "pad_values": {"audio": 999},
        },
        _omni_consumed=None,
        is_chunked=0,
    )
    input_embeds, _, _ = runner._inject_multimodal_embeds(
        SimpleNamespace(input_ids=torch.tensor([1, 999, 2]), extend_seq_lens_cpu=[3]),
        SimpleNamespace(reqs=[req]),
    )

    assert (
        int(runner._embed_tokens.seen.max().item())
        < runner._embed_tokens.num_embeddings
    )
    assert torch.equal(input_embeds[1], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    logits = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 1.0, 0.0]]])
    sampled = Qwen3OmniTalker._sample_code_predictor_token(logits)
    assert sampled.shape == (2, 1)
    assert sampled[:, 0].tolist() == [2, 0]
