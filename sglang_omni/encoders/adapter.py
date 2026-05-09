# SPDX-License-Identifier: Apache-2.0
"""Encoder adapter contract.

The adapter owns the conversion between a pipeline ``StagePayload`` and
the encoder's request/result types. The :class:`EncoderScheduler` is
intentionally model-agnostic and asks the adapter to do all
model-specific work — including state load/store — so that a generic
encoder factory does not need to import any specific model module.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sglang_omni.engines.omni.runtime import EncoderRequestData
from sglang_omni.proto import StagePayload


@runtime_checkable
class EncoderAdapter(Protocol):
    """Bidirectional bridge between a stage payload and encoder I/O.

    Implementations are expected to be small pure-Python objects.

    Attributes:
        stage_name: Stage identifier the adapter is bound to (the same
            string the pipeline uses for its stage). Stored for
            traceability / logging; the adapter is also free to use it
            internally to key into model-specific state.
    """

    stage_name: str

    def build_request(self, payload: StagePayload) -> EncoderRequestData:
        """Materialise the encoder request from a stage payload."""
        ...

    def apply_result(
        self, payload: StagePayload, result: object
    ) -> StagePayload:
        """Apply the encoder result back into the payload and return it."""
        ...
