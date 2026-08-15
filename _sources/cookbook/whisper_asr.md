# Whisper ASR

Whisper ASR checkpoints can be started through the OpenAI-compatible `/v1/audio/transcriptions` endpoint, but this path is experimental in the current SGLang-Omni tree. Prefer [Qwen3-ASR](qwen3_asr.md) for validated ASR serving.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download a Whisper checkpoint:

```bash
hf download openai/whisper-large-v3
```

## Server Configuration

Whisper ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

## Encoder CUDA Graph

The encoder CUDA Graph is enabled by default for the pipeline. The final bucket set is resolved from the serving prefill budget and the checkpoint's encoder prefix length; with the default 6,144-token budget and 1,500-token Whisper encoder prefix, batches 1, 2, and 4 are captured. To use eager encoder execution, override the pipeline configuration:

```yaml
config_cls: WhisperASRPipelineConfig
name: whisper
model_path: openai/whisper-large-v3-turbo

runtime_overrides:
  asr:
    enable_encoder_cuda_graph: false
```

The graph is captured after SGLang's generation graphs. Raise `max_prefill_tokens` before configuring larger buckets. Each request uses the smallest captured bucket that fits its batch. Requests larger than every captured bucket, with a different feature shape, or without a successful capture run eagerly. Startup and first-replay logs identify the captured and executed buckets.

## Prefill Coalescing

Whisper builds up to two requests concurrently and coalesces prefill at the serving-reachable batch size of two. A partial batch waits for at most 6 ms only while another request build is pending; a single request and a partial batch with no remaining build work are released immediately. This allows concurrent traffic to replay the encoder batch-2 graph without adding a fixed wait to the idle c=1 path.

`request_build_max_pending` bounds submitted request-build futures, not the request backlog. When `max_queued_requests` is unset, requests beyond that pending-build limit remain queued for later construction. Setting `max_queued_requests` retains the configured finite-queue rejection behavior.

Use `prefill_coalesce_requests` and `prefill_coalesce_wait_ms` to tune the gate. Set `prefill_coalesce_requests: 0` to disable only coalescing, or also set `request_build_max_workers: 1` to restore the pre-optimization request-build path:

```yaml
runtime_overrides:
  asr:
    request_build_max_workers: 1
    prefill_coalesce_requests: 0
```

## Async Decode

Whisper enables the shared one-step-lookahead decode path at batch size 2 and above. It overlaps the current decode step's GPU work with the previous step's host-side result processing, while batch size 1 remains on the synchronous path. The default running-request limit is 32. Use the shared decode-mode option to compare against synchronous decode or diagnose a request lifecycle issue:

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --decode-mode sync \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "openai/whisper-large-v3",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional language hint |
| `prompt` | string | unset | Optional text used as Whisper prev-context conditioning |
| `response_format` | string | `json` | Use `json` for the current Whisper path |
| `temperature` | float | `0.0` | Sampling temperature; defaults to greedy decoding |

The request builder also supports `task` (`transcribe` by default) and
`max_new_tokens`, but the public transcription endpoint currently exposes only
the fields above. The route uses the ASR stage default unless the pipeline is
configured another way. For smoke tests, keep the request minimal and use
`response_format=json`.

## Benchmarking

Use the shared SeedTTS benchmark for end-to-end concurrency, WER, latency, and throughput:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-base \
  --max-samples 128 --concurrencies 1,2,4,8,16,32 \
  --repeats 5 --warmup --output whisper_concurrency.json
```

To reproduce the async-decode comparison below, resolve the pinned checkpoint and start each mode separately on the same GPU:

```bash
MODEL_REVISION=e37978b90ca9030d5170a5c07aadb050351a65bb
MODEL_PATH=$(hf download openai/whisper-base --revision "$MODEL_REVISION")

CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --mem-fraction-static 0.20 \
  --port 8000

# Replace the command above with this one for the synchronous baseline.
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --mem-fraction-static 0.20 \
  --decode-mode sync \
  --port 8000
```

Run the same client command once per mode, changing only the output filename:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --model-path openai/whisper-base \
  --model-revision e37978b90ca9030d5170a5c07aadb050351a65bb \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --max-samples 128 \
  --concurrencies 1,2,4,8,16,32 \
  --repeats 5 \
  --warmup \
  --dtype float16 \
  --cuda-graph \
  --torch-compile \
  --max-running-requests 32 \
  --mem-fraction-static 0.20 \
  --fingerprint \
  --output whisper_async.json
```

## Benchmark Results

The following W-PR1 results used the 20-sample SeedTTS EN subset on a single H200 with `openai/whisper-base` in FP16. Each mode ran one discarded warmup and three measured repeats per concurrency.

| Concurrency | Eager req/s | CUDA Graph req/s | Throughput gain | Eager mean latency (s) | CUDA Graph mean latency (s) | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.57 | 20.29 | 3.7% | 0.051 | 0.049 | 0.0415 |
| 2 | 28.41 | 30.87 | 8.7% | 0.070 | 0.065 | 0.0415 |
| 4 | 37.90 | 41.70 | 10.0% | 0.104 | 0.094 | 0.0415 |
| 8 | 42.10 | 49.00 | 16.4% | 0.185 | 0.158 | 0.0415 |

All 480 W-PR1 measured requests completed successfully. Corpus WER was unchanged across eager and CUDA Graph modes at every concurrency.

The following W-PR2 results were measured separately on the same H200 and 20-sample subset with five measured repeats plus one discarded warmup per concurrency. The baseline used one request-build worker with coalescing disabled; the attribution run used two workers with coalescing disabled; the optimized run used two workers, a batch target of two, and a pending-build-aware 6 ms deadline.

| Concurrency | Baseline req/s | Two workers req/s | Coalesced req/s | Total gain | Gate gain | Baseline latency (s) | Coalesced latency (s) | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.04 | 22.51 | 22.46 | 6.8% | -0.3% | 0.047 | 0.044 | 0.0415 |
| 2 | 30.45 | 36.68 | 41.96 | 37.8% | 14.4% | 0.066 | 0.047 | 0.0415 |
| 4 | 40.24 | 55.62 | 62.83 | 56.2% | 13.0% | 0.097 | 0.063 | 0.0415 |
| 8 | 48.03 | 75.93 | 82.15 | 71.0% | 8.2% | 0.161 | 0.092 | 0.0415 |

All 1,200 measured requests completed successfully. Corpus WER remained 0.0415 in all three modes and at every concurrency. Logs from the optimized run showed `Replaying Whisper encoder CUDA graph batch=2 request_batch=2` and prefill batches with two sequences and 3,008 new tokens.

The async-decode comparison used the 128-sample SeedTTS EN subset on the same H200 with `openai/whisper-base` in FP16, one discarded warmup, and five measured repeats per concurrency. The baseline disabled async decode; all other serving settings were identical.

| Concurrency | Sync req/s | Async req/s | Throughput change | Sync P95 (s) | Async P95 (s) | P95 change | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.21 | 20.47 | +1.3% | 0.060 | 0.060 | -1.2% | 0.0329 |
| 2 | 36.39 | 37.42 | +2.8% | 0.067 | 0.064 | -3.5% | 0.0329 |
| 4 | 58.78 | 59.56 | +1.3% | 0.085 | 0.080 | -6.3% | 0.0329 |
| 8 | 82.71 | 84.61 | +2.3% | 0.113 | 0.110 | -2.9% | 0.0329 |
| 16 | 105.92 | 108.70 | +2.6% | 0.168 | 0.165 | -1.5% | 0.0329 |
| 32 | 112.91 | 120.46 | +6.7% | 0.390 | 0.354 | -9.3% | 0.0329 |

All 7,680 measured requests across both modes completed successfully. Batch size 1 uses the synchronous fast path, so its observed 1.3% difference is run-to-run noise rather than async work. A separate ten-repeat concurrency-32 stability run measured 119.49 req/s and 0.341 s P95 synchronously versus 130.68 req/s and 0.271 s P95 asynchronously, a 9.4% throughput increase and 20.6% P95 reduction. All 128 transcripts matched the synchronous baseline exactly. Request-stage profiling attributed the tail-latency reduction primarily to work after prefill: P95 from prefill completion to request completion fell from 304.6 ms to 195.4 ms, while scheduler queue P95 fell from 32.9 ms to 8.6 ms.

## Known Limitations

- Whisper ASR remains experimental. Validate checkpoint-specific accuracy and
  operational behavior before production deployment.
- Encoder CUDA Graph is enabled by default and requires SGLang generation CUDA
  Graph to remain enabled. Validate the selected buckets before production use.
- Chunked prefill is disabled because the Whisper encoder prefix must be
  admitted atomically. Requests that exceed the current prefill budget wait
  for the next batch instead of splitting the encoder prefix.
- Use `response_format=json`; other response formats are not validated for this
  experimental path.
- First startup can take several minutes.
- The endpoint accepts one uploaded file per request.
- Audio is resampled to 16 kHz before transcription.
- `prompt` conditions decoding via Whisper prev-context tokens. Only the last
  223 prompt tokens are kept (224 prev-context tokens including
  `<|startofprev|>`) — fewer when `max_new_tokens` is large, since prompt,
  task prefix, and output share Whisper's 448-token decoder context.
  `max_new_tokens` is likewise clamped to that context. The prompt must not
  contain Whisper special tokens.
