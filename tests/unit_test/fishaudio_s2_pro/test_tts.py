# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni_v1.models.fishaudio_s2_pro.fish_scheduler import (
    FishIterationController,
    FishScheduler,
)
from sglang_omni_v1.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerRequest,
)
from tests.unit_test.fixtures.fish_fakes import (
    FakeFishModel,
    FakeFishReq,
    make_s2pro_payload,
)


def test_fish_model_runner_vq_injection_and_code_collection_contracts() -> None:
    """Preserves VQ prompt embedding injection and semantic code collection."""
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = FakeFishModel()
    runner._semantic_begin_id = 200
    runner._semantic_end_id = 295
    runner._im_end_token_id = 99
    prefill_request = SchedulerRequest(
        request_id="prefill",
        data=SimpleNamespace(
            req=FakeFishReq(extend_input_len=3),
            vq_mask_tokens=torch.tensor([True, False, True]),
            vq_parts=[torch.tensor([[7, 8], [9, 10]], dtype=torch.long)],
        ),
    )
    embeds = runner._build_prefill_input_embeds(
        SimpleNamespace(input_ids=torch.tensor([10, 11, 12])),
        [prefill_request],
    )
    assert torch.equal(embeds[0], torch.tensor([1007.0, 1009.0]))
    assert torch.equal(embeds[1], torch.tensor([11.0, 11.0]))

    active = SchedulerRequest(
        request_id="active",
        data=SimpleNamespace(
            req=FakeFishReq(is_chunked=0),
            output_codes=[],
            previous_semantic_tokens=[],
            last_codebook_values=None,
        ),
    )
    runner._collect_step_outputs(SimpleNamespace(next_token_ids=None), [active])
    assert len(active.data.output_codes) == 1
    assert torch.equal(active.data.last_codebook_values, torch.tensor([1, 2]))
    assert active.data.previous_semantic_tokens == [201]


class _FakePlanner:
    def __init__(self) -> None:
        self.recorded = None

    def select_requests(self, waiting, running):
        del running
        return list(waiting)

    def build_batch(self, requests):
        return SimpleNamespace(request_ids=[request.request_id for request in requests])

    def record_last_batch(self, batch_data) -> None:
        self.recorded = batch_data


class _FakeResourceManager:
    def __init__(self) -> None:
        self.freed: list[str] = []

    def free(self, request) -> None:
        self.freed.append(request.request_id)


def make_fish_scheduler() -> FishScheduler:
    def request_builder(payload):
        return SimpleNamespace(
            req=FakeFishReq(rid=payload.request_id),
            output_codes=[torch.tensor([[100], [1], [2]], dtype=torch.long)],
            previous_semantic_tokens=[],
            last_codebook_values=None,
            max_new_tokens=4,
            input_ids=[1, 2, 3],
        )

    def result_adapter(data):
        payload = make_s2pro_payload(request_id=data.req.rid)
        payload.data = {"output_ids": list(data.req.output_ids)}
        return payload

    scheduler = FishScheduler(
        tree_cache=SimpleNamespace(cache_unfinished_req=lambda req: None),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(),
        prefill_manager=SimpleNamespace(),
        decode_manager=SimpleNamespace(),
        server_args=SimpleNamespace(),
        model_runner=SimpleNamespace(),
        request_builder=request_builder,
        result_adapter=result_adapter,
        im_end_token_id=99,
        max_new_tokens=4,
    )
    scheduler.batch_planner = _FakePlanner()
    scheduler.resource_manager = _FakeResourceManager()
    return scheduler


def test_fish_scheduler_lifecycle_abort_and_iteration_contracts() -> None:
    """Preserves chunked iteration state, finished emission, and abort cleanup."""
    request = SchedulerRequest(
        request_id="chunked",
        data=SimpleNamespace(
            req=FakeFishReq(is_chunked=2),
            output_codes=[],
            previous_semantic_tokens=[],
        ),
    )
    controller = FishIterationController(
        tree_cache=SimpleNamespace(cache_unfinished_req=lambda req: None),
        im_end_token_id=99,
        max_new_tokens=4,
    )
    controller.update_request(request, 10)
    assert request.data.req.is_chunked == 1
    assert request.data.req.output_ids == []

    scheduler = make_fish_scheduler()
    scheduler.process_input_requests([make_s2pro_payload(request_id="req-1")])
    batch = scheduler.schedule()
    finished = scheduler.update(
        batch,
        ModelRunnerOutput(outputs={"req-1": RequestOutput("req-1", data=99)}),
    )
    scheduler.emit_finished(finished)
    message = scheduler.outbox.get_nowait()
    assert batch.request_ids == ["req-1"]
    assert scheduler.resource_manager.freed == ["req-1"]
    assert message.type == "result"
    assert message.data.data["output_ids"] == [99]

    scheduler.process_input_requests([make_s2pro_payload(request_id="req-2")])
    scheduler.abort("req-2")
    scheduler.inbox.put(
        IncomingMessage("req-2", "new_request", make_s2pro_payload(request_id="req-2"))
    )
    assert scheduler.recv_requests() == []
    assert "req-2" not in scheduler._requests
