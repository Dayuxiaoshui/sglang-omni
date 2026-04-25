# Video-MME Talker-ON CI: Investigation Notes (Deferred)

This branch holds the **deferred** work for a Video-MME TTS-consistency CI
(Talker ON) that was intended to mirror
`tests/test_model/test_qwen3_omni_mmmu_tts_consistency_ci.py` and
`tests/test_model/test_qwen3_omni_mmsu_tts_consistency_ci.py`. The
thinker-only Video-MME CI (stage-7) is on `issue-253-ci` / PR #338. The
Talker-ON CI is sequestered here because every probe configuration we
ran on H200 collapses on the very first sample — there is no calibration
band to derive a threshold from. This branch carries:

* `examples/run_qwen3_omni_speech_server.py` — exposes
  `--thinker-max-seq-len` and `--talker-attention-backend` so any future
  diagnostic work has the same operator knobs we used.
* this doc — the full reproduction recipe and failure-mode taxonomy so
  the next contributor does not repeat the same six experiments.

The branch deliberately does NOT include a CI test file. We tried two
shapes (a normal failing test, and a `@pytest.mark.skip` shim); both are
inferior to "no CI stage at all" because:

* a failing CI confuses signal-to-noise on every PR until the upstream
  Talker bug lands, and
* a permanently-skipped CI is dead code in the workflow that nobody
  ever flips back on.

Once the upstream bug is fixed, dropping a freshly-calibrated test file
(modeled on `test_qwen3_omni_mmsu_tts_consistency_ci.py`) is a one-file
PR; this branch is the recipe, not a half-finished implementation.


## Goal

Mirror the existing Talker-ON TTS consistency CIs for Video-MME:

* **Sample shape** — small subset (5 samples) of `videomme-ci-50` at
  `concurrency=4`, `max_tokens=256`. Match the MMMU/MMSU TTS CI shape.
* **Asserts** — text accuracy floor, audio WER between text answer and
  ASR transcript of the Talker's audio, zero failed requests, per-
  concurrency speed thresholds via `apply_slack(0.75, 1.25)`.
* **Reproducibility** — fresh server per pytest invocation, cold-run
  P95 worst-of-5 calibration on H200.


## Six probes on H200, every one fails on the first sample

All probes ran `Qwen3-Omni-30B-A3B-Instruct` weights, dataset
`zhaochenyang20/Video_MME_ci`, against `examples/run_qwen3_omni_speech_server.py`.

| # | Config delta from probe #2 | Outcome |
| --- | --- | --- |
| 1 | `c=4`, 50 samples, default 8192 thinker context | Thinker length guard rejects a 9573-token Video-MME prompt. Pipeline relay propagates the failure; every concurrent request collapses. |
| 2 | `c=4`, 5 samples, `--thinker-max-seq-len 32768`, `--thinker-mem-fraction-static 0.55`, `--talker-mem-fraction-static 0.30` | First sample's Talker forward trips a CUDA "illegal memory access" inside FA3. CUDA context poisoned; the remaining 4 samples all fail with the same poisoned-context error. |
| 3 | `c=1`, otherwise same as #2 | First sample's Talker forward trips `IndexKernel.cu:111 "-sizes[i] <= index && index < sizes[i] index out of bounds"` device-side assert. Still inside the Talker's prompt-state reconstruction path. |
| 4 | #2 + `CUDA_LAUNCH_BLOCKING=1` + `TORCH_USE_CUDA_DSA=1` to pin the kernel source | The actual kernel surfaces as `_deps/repo-flash-attention-src/hopper/flash_fwd_launch_template.h:200: CUDA error: an illegal memory access was encountered`. So the c=4 failure is FA3 on Hopper mishandling the Talker's attention pattern at long-prompt inputs. |
| 5 | #4 + `--talker-attention-backend triton` (newly added flag, plus matching `mm_attention_backend`) | The FA3 failure goes away. The failure *moves* to `IndexKernel.cu:111` on the first sample, inside the Talker's prompt-state reconstruction path. Conclusion: the Talker has at least *two* independent CUDA-level bugs that fire on Video-MME prompts — FA3 mishandling, plus a downstream `index_select` OOB. Swapping attention backends papers over the first but not the second. |
| 6 | #5 + patched `_load_prompt_token_embeddings` to bypass `torch.unique(sorted=False, return_inverse=True) + unique_rows.index_select(0, inverse)` with a direct per-token `stack` | Still `IndexKernel.cu:111` on the first sample. The failing `index_select` is therefore **not** the embedding-cache helper — it lives further inside `_reconstruct_prompt_states` / `build_prefill_input`, likely inside `_talker_model.forward` itself. With `CUDA_LAUNCH_BLOCKING=1` we still don't get a Python frame for the assert, so the offending call is in a path that doesn't surface back to Python. |

All six probes failed on the very first sample. Zero successful Talker-ON
Video-MME runs to anchor any threshold against.


## Why concurrency=16 (the obvious next ask) does not save us either

Several reviewers asked whether bumping concurrency might help (more
parallelism amortizes per-request overhead, and SGLang's prefill
scheduler at higher batch sizes sometimes triggers different code
paths). The answer is no, and the failure modes already on hand
explain why:

* **`c=1` (probe #3) fails on the first sample.** That failure is not
  a contention / race / oversubscription issue. The Talker is
  processing exactly one request and still tripping the OOB. Concurrency
  is not what's pushing the index out of bounds; the Video-MME prompt
  *length* is.
* **`c=4` (probes #2/#4/#5) fails on the first sample.** Same story
  with four parallel requests. The CUDA context dies on the first one;
  the other three never get to run their forwards.
* **`c=16` (extrapolation).** With 16 concurrent Video-MME requests on
  a single H200, the talker stage's SGLang scheduler would batch them
  together, which simply means 16 requests fail at once rather than
  one. There is no scheduler/codepath toggle at higher concurrency
  that bypasses the offending `index_select`. The bug is per-request
  on prompt length, not per-batch.
* **Talker serialization on `code_predictor` / `code2wav`.** The
  existing Talker-ON CIs (MMMU, MMSU) intentionally use `c=1` because
  the post-Talker stages serialize on a single GPU regardless of
  upstream concurrency. Higher `c` therefore wastes GPU rather than
  scaling. It does not fix correctness.

Net: there is no `c` value that hides the underlying bug.


## What MMMU / MMSU TTS consistency do that Video-MME cannot

Both sibling Talker-ON CIs hit the same Talker code path —
`_reconstruct_prompt_states`, `_load_prompt_token_embeddings`,
`build_prefill_input`, `codec_embed_fn`. They pass:

* MMMU image-QA prompts: ~300-1500 thinker tokens.
* MMSU audio-QA prompts: ~500-1500 thinker tokens.
* Video-MME prompts: 2000-9000 thinker tokens, driven by dense
  per-frame vision-placeholder tokens from 32-64 sampled frames per
  clip.

The Talker bug only fires above some prompt-length threshold that lives
between ~1500 and ~2000 thinker tokens. Video-MME crosses that line
unconditionally; MMMU and MMSU sit below it.


## Why we did not ship a weakened Task-3 CI

Three weakenings were considered and explicitly rejected:

1. **Truncate Video-MME prompts to MMMU-length regime.** Defeats the
   purpose. The whole point of a Video-MME Talker CI is to exercise
   the Talker on *realistic* video prompts; if we cut the prompt to
   MMMU's size, the test is just a renamed copy of the MMMU CI.
2. **Lower accuracy / WER / speed thresholds until the failing run
   passes.** Produces a CI that reports green for a broken path. The
   next genuine Talker regression then slips through silently.
3. **Mark the test as `@pytest.mark.skip`.** Hides the deferral inside
   the workflow rather than documenting it. The file ends up dead code
   that no one revisits.

The chosen path — defer entirely, document the recipe in this branch
— preserves the work without faking signal.


## Partial work retained (the speech-server flags)

`examples/run_qwen3_omni_speech_server.py` on this branch exposes two
new flags. Both are net-positive operator knobs for any speech-server
workload, independent of the deferred CI:

* `--thinker-max-seq-len` — raise the Thinker stage's context. The
  thinker-only launcher has had this for a while; the speech launcher
  was the outlier. Long-prompt workloads (including any future Talker-
  ON Video-MME use, once the bug is fixed) need a way to raise the
  Thinker context above the factory default without editing the
  pipeline config.
* `--talker-attention-backend` — pin the Talker stage's SGLang
  attention backend (and the matching `mm_attention_backend`)
  independently of the Thinker. SGLang auto-selects `fa3` on Hopper;
  this flag gave us probe #5's evidence that FA3 was not the only bug
  in this path. It did not fix the regression — failure just moved
  from FA3 to a downstream `index_select` — but the flag itself is the
  right shape for any future diagnostic work.

The override-accumulator shape inside the speech launcher mirrors the
thinker-only launcher so the next speech-launcher CLI flag drops in
cleanly.


## Unblocking criteria

Any one of these clears the way for a Talker-ON Video-MME CI:

1. **Upstream Talker fix for the long-prompt `index_select` assert.**
   Reproducer: launch `examples/run_qwen3_omni_speech_server.py` with
   `--thinker-max-seq-len 32768`, send any Video-MME sample whose
   thinker-input length exceeds ~1500 tokens. With
   `CUDA_LAUNCH_BLOCKING=1` the assert appears at `IndexKernel.cu:111`
   on the first request. The failing `index_select` call site has not
   been pinned to a Python frame; the most likely region is inside
   `_talker_model.forward` (the standalone-Talker SGLang engine) or
   the post-`build_prefill_input` projection path.
2. **Talker-side input validator that clamps prompt token IDs to the
   Talker's `codec_vocab_size`** (currently 3072 per the
   `Qwen3OmniMoeForConditionalGeneration` config) before any
   `codec_embed_fn(...)` call, with an actionable error instead of
   the silent OOB.
3. **An explicitly-named subset (e.g. "videomme-tts-short") that is
   curated to stay below the Talker's failing prompt-length ceiling**,
   plus its own threshold set documented as a Talker-ON-subset rather
   than as a Video-MME proxy. This is a lower-quality answer than 1 or
   2 and should only be considered if the upstream Talker fix is far
   off.


## Reproducing the probes

```bash
# pick two free GPUs (avoid GPUs the thinker-only Task 1 / Task 2
# experiments touched recently — leftover shm allocations linger on
# Hopper for several seconds after kill -9 and can OOM the Talker
# stage at startup).
export CUDA_VISIBLE_DEVICES=2,3
export CUDA_LAUNCH_BLOCKING=1   # use this if you want a Python frame
export TORCH_USE_CUDA_DSA=1     # pair with CUDA_LAUNCH_BLOCKING

python examples/run_qwen3_omni_speech_server.py \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu-thinker 0 \
    --gpu-talker 1 --gpu-code-predictor 1 --gpu-code2wav 1 \
    --port 8000 --model-name qwen3-omni \
    --thinker-max-seq-len 32768 \
    --thinker-mem-fraction-static 0.55 \
    --talker-mem-fraction-static 0.30 \
    --talker-attention-backend triton    # optional; toggles probe 4↔5
```

Then issue the standard Video-MME `videomme-ci-50` benchmark with
`--max-samples 5 --enable-audio --max-concurrency 4`. The first sample
will trip the assert. Per-stage server logs land under
`/tmp/sglang_omni/<model-id>/` while the server is alive; the
`talker_ar` log holds the kernel-assertion text.


## Cross-reference

* PR #338 (`Jayon02/issue-253-ci`) — the **landed** thinker-only
  stage-7 CI (calibrated on H200 across 5 cold runs).
* PR #327 (merged to main) — the Video-MME benchmark framework this
  CI work sits on top of.
* PR #339 (merged to main) — the upstream `--encoder-mem-reserve` flag
  on the thinker-only launcher; this branch's `--talker-attention-
  backend` mirrors that pattern on the speech launcher.
* Issue #253 — the broader Qwen3-Omni CI coverage tracker.
* Sibling Talker-ON CIs (working): `tests/test_model/
  test_qwen3_omni_mmmu_tts_consistency_ci.py` and
  `tests/test_model/test_qwen3_omni_mmsu_tts_consistency_ci.py`.
