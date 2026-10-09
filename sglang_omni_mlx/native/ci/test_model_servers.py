# SPDX-License-Identifier: Apache-2.0
"""API tests for the native model servers beside qwen3_asr_server, run against
the real models.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_model_servers.py
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import subprocess
import time
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_server_api import (
    DATA_ROOT,
    RUNTIME_BIN,
    Server,
    clip,
    long_wav,
    pcm16,
    sse_events,
)
from websockets.exceptions import InvalidHandshake
from websockets.sync.client import connect

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
MOSS_REPO = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


class ModelServer(Server):
    """Another model's server binary, started and spoken to as qwen3_asr_server is."""

    def __init__(self, binary: str, model_kind: str, repo: str) -> None:
        super().__init__(
            binary, model_kind, Path(DATA_ROOT) / "models" / repo.replace("/", "_")
        )


@pytest.fixture(scope="module")
def whisper_server() -> Iterator[ModelServer]:
    running = ModelServer("whisper_server", "whisper", WHISPER_REPO)
    yield running
    running.stop()


def test_whisper_ready_event_names_a_loopback_endpoint(
    whisper_server: ModelServer,
) -> None:
    assert whisper_server.ready["event"] == "ready"
    assert whisper_server.ready["host"] == "127.0.0.1"
    assert whisper_server.ready["server_pid"] == whisper_server.process.pid
    assert whisper_server.ready["model_name"].startswith("voxt-whisper-")
    status, body = whisper_server.request("GET", "/health")
    assert (status, json.loads(body)) == (
        200,
        {"status": "healthy", "running": True, "request_states": {}},
    )


def test_whisper_final_request_streams_text_and_generation_metadata(
    whisper_server: ModelServer,
) -> None:
    status, body = whisper_server.post_form(
        {
            "stream": "true",
            "language": "en",
            "max_new_tokens": "1024",
            "temperature": "0.0",
            "include_generation_metadata": "true",
        },
        clip("0006_en_short"),
    )
    assert status == 200
    assert sse_events(body) == [
        {
            "type": "transcript.text.done",
            "text": "Surely you are not thinking of going off there.",
            "generation_metadata": {
                "generated_token_count": 10,
                "language": "en",
                "finish_reason": "stop",
            },
        },
        "[DONE]",
    ]


def test_whisper_plain_request_returns_json_text(whisper_server: ModelServer) -> None:
    status, body = whisper_server.post_form({"language": "zh"}, clip("0152_zh_short"))
    assert (status, json.loads(body)) == (
        200,
        {"text": "互联网结合了大众传播和人际传播的要素"},
    )


@pytest.mark.parametrize(
    ("fields", "wav"),
    [
        ({}, None),
        ({}, b"not audio"),
        ({"include_generation_metadata": "true"}, "wav"),
        ({"max_new_tokens": "many"}, "wav"),
        ({"temperature": "warm"}, "wav"),
        ({"temperature": "nan"}, "wav"),
        ({"temperature": "inf"}, "wav"),
        ({"temperature": "-1"}, "wav"),
        ({"max_new_tokens": "-1"}, "wav"),
    ],
)
def test_whisper_invalid_requests_are_rejected(
    whisper_server: ModelServer, fields: dict[str, str], wav: bytes | str | None
) -> None:
    status, body = whisper_server.post_form(
        fields, clip("0006_en_short") if wav == "wav" else wav
    )
    assert status == 400
    assert "detail" in json.loads(body)


def test_whisper_bad_number_error_names_its_field(whisper_server: ModelServer) -> None:
    status, body = whisper_server.post_form(
        {"max_new_tokens": "many"}, clip("0006_en_short")
    )
    assert status == 400
    assert "max_new_tokens" in json.loads(body)["detail"]


def test_whisper_language_with_a_truncated_utf8_sequence_is_unknown(
    whisper_server: ModelServer,
) -> None:
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n'.encode()
        + b"en\xf0\r\n"
        + f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'.encode()
        + clip("0006_en_short")
        + f"\r\n--{boundary}--\r\n".encode()
    )
    status, response = whisper_server.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    # Note (khazic): an unknown language sends no language token, as Swift does; the
    # truncated sequence must not be read past.
    assert status == 200
    assert "text" in json.loads(response)
    assert whisper_server.request("GET", "/health")[0] == 200


def test_whisper_disconnected_stream_stops_its_decode(
    whisper_server: ModelServer,
) -> None:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\ntrue\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        ).encode()
        + long_wav(600)
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection = http.client.HTTPConnection(
        "127.0.0.1", whisper_server.port, timeout=30
    )
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    assert response.status == 200
    time.sleep(2.0)
    assert json.loads(whisper_server.request("GET", "/health")[1])[
        "request_states"
    ] == {"running": 1}
    closed_at = time.monotonic()
    # Note (khazic): http.client keeps the socket open while the response object is.
    response.close()
    connection.close()
    while (
        json.loads(whisper_server.request("GET", "/health")[1])["request_states"] != {}
    ):
        assert (
            time.monotonic() - closed_at < 2.0
        ), "the decode kept running after its client left"
        time.sleep(0.05)


def test_whisper_serves_no_realtime_api(whisper_server: ModelServer) -> None:
    with pytest.raises(InvalidHandshake):
        connect(f"ws://127.0.0.1:{whisper_server.port}/v1/realtime")


def test_whisper_shutdown_reports_stopped() -> None:
    running = ModelServer("whisper_server", "whisper", WHISPER_REPO)
    running.process.stdin.write('{"command": "shutdown"}\n')
    running.process.stdin.flush()
    assert json.loads(running.process.stdout.readline()) == {"event": "stopped"}
    assert running.process.wait(timeout=10) == 0


def test_whisper_server_serves_only_whisper() -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "whisper_server"),
            "--supervised",
            "--model-kind",
            "qwen3_asr",
            "--model-directory",
            "x",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.fixture(scope="module")
def moss_server() -> Iterator[ModelServer]:
    running = ModelServer(
        "moss_transcribe_diarize_server", "moss_transcribe_diarize", MOSS_REPO
    )
    yield running
    running.stop()


def two_speaker_wav() -> bytes:
    """Two LibriSpeech speakers half a second apart."""
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(
            pcm16("0000_en_short") + bytes(16000) + pcm16("0001_en_short")
        )
    return out.getvalue()


def test_moss_request_returns_speaker_segments(moss_server: ModelServer) -> None:
    assert moss_server.ready["model_name"].startswith("voxt-moss_transcribe_diarize-")
    status, body = moss_server.post_form({}, two_speaker_wav())
    assert status == 200
    result = json.loads(body)
    # Greedy decoding follows the chip's arithmetic, so the exact text and
    # timestamps are a golden-file concern: assert the structure here.
    segments = result["segments"]
    assert [segment["speaker"] for segment in segments] == ["S01", "S02"]
    assert [segment["start"] for segment in segments] == sorted(
        segment["start"] for segment in segments
    )
    assert all(segment["end"] >= segment["start"] for segment in segments)
    assert "Socrates begins the timaeus" in segments[0]["text"]
    assert "no signs here" in segments[1]["text"]
    assert "[S01]" in result["text"] and "[S02]" in result["text"]


def test_moss_streamed_request_ends_with_segments(moss_server: ModelServer) -> None:
    status, body = moss_server.post_form(
        {"stream": "true", "include_generation_metadata": "true"},
        clip("0006_en_short"),
    )
    assert status == 200
    done, end = sse_events(body)
    assert end == "[DONE]"
    assert done["type"] == "transcript.text.done"
    assert done["segments"] and done["generation_metadata"]["language"] is None


@pytest.mark.parametrize(
    "fields",
    [
        {"prompt": "<|audio_pad|> and <|audio_pad|>"},
        {"max_new_tokens": "-1"},
        {"max_new_tokens": "many"},
    ],
)
def test_moss_invalid_requests_are_rejected(
    moss_server: ModelServer, fields: dict[str, str]
) -> None:
    status, body = moss_server.post_form(fields, clip("0006_en_short"))
    assert status == 400
    assert "detail" in json.loads(body)


def test_moss_realtime_rejects_a_prompt_it_cannot_render(
    moss_server: ModelServer,
) -> None:
    with connect(f"ws://127.0.0.1:{moss_server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "prompt": "<|audio_pad|><|audio_pad|>",
                    },
                }
            )
        )
        error = json.loads(socket.recv())
        assert error["type"] == "error"
        assert error["error"]["type"] == "invalid_request_error"
        assert error["error"]["code"] == "invalid_prompt"
        socket.send(
            json.dumps({"type": "session.update", "session": {"turn_detection": None}})
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"


def test_moss_realtime_finalizes_windows_with_segments(
    moss_server: ModelServer,
) -> None:
    pcm = pcm16("0344_en_long")[: 10 * 32000]
    with connect(f"ws://127.0.0.1:{moss_server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps({"type": "session.update", "session": {"turn_detection": None}})
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"
        for start in range(0, len(pcm), 3200):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[start : start + 3200]).decode(),
                    }
                )
            )
            # Twice real time: a window decodes well within its 2 s of sending.
            time.sleep(0.05)
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = []
        while not events or events[-1]["type"] != "transcription.completed":
            events.append(json.loads(socket.recv()))
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    # Windows are finalized as they fill; one that fills during a decode waits
    # for it, and may be joined by the tail at the commit.
    assert len(finals) >= 2
    assert [event["segment_id"] for event in finals] == list(range(len(finals)))
    completed = events[-1]
    assert completed["text"] == "\n".join(event["text"] for event in finals)
    starts = [segment["start"] for segment in completed["segments"]]
    assert starts == sorted(starts)
    assert finals[1]["segments"][0]["start"] >= 4.0


def test_moss_server_serves_only_moss() -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "moss_transcribe_diarize_server"),
            "--supervised",
            "--model-kind",
            "qwen3_asr",
            "--model-directory",
            "x",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
