# Qwen3 Omni V1 CUDA Graph Calibration Report

This lightweight report records the second Qwen3 Omni V1 threshold calibration
run, after verifying that the optimized decode path was active at runtime.

- Model: `qwen3-omni-v1`
- Repeats: 5
- Stages: `mmmu`, `mmmu_talker`, `mmsu`, `mmsu_talker`, `tts`, `videoamme`, `videoamme_talker`, `videomme`, `videomme_talker`
- Excluded: docs smoke tests
- Local artifact directory: `.tune-runs/20260506T220900Z_qwen3-omni-v1_cuda-graph_no-docs_r5`
- Full raw logs and JSON results are intentionally kept local under `.tune-runs/` and are not included in git.
- Runtime evidence: final-repeat pytest logs for all calibrated stages contain `cuda graph: True` decode batches.

## Accuracy and WER

| Stage | Worst-of-5 |
|-------|------------|
| MMMU accuracy | 56.00% |
| MMMU talker accuracy | 70.00% |
| MMMU talker WER | 19.81% corpus WER, 3 samples >50% WER |
| MMSU accuracy | 69.60% |
| MMSU talker accuracy | 60.00% |
| MMSU talker WER | 7.84% corpus WER, 3 samples >50% WER |
| TTS WER | 2.66% corpus WER, 1 sample >50% WER |
| Video-AMME accuracy | 66.67% |
| Video-AMME talker accuracy | 50.00% |
| Video-AMME talker WER | 6.37% corpus WER, 2 samples >50% WER |
| Video-MME accuracy | 53.33% |
| Video-MME talker accuracy | 50.00% |
| Video-MME talker WER | 6.28% corpus WER, 0 samples >50% WER |

## Speed Worst-of-5

| Stage | Throughput | Tok/s | Latency | RTF |
|-------|------------|-------|---------|-----|
| MMMU | 0.677 req/s | 53.10 | 11.230s | - |
| MMMU talker | 0.159 req/s | 5.70 | 23.796s | 0.3550 |
| MMSU | 29.911 req/s | 7.70 | 0.267s | - |
| MMSU talker | 0.280 req/s | 3.80 | 16.154s | 0.3895 |
| TTS | 3.986 req/s | 7.50 | 1.950s | 0.5828 |
| Video-AMME | 0.236 req/s | 0.90 | 51.633s | - |
| Video-AMME talker | 0.126 req/s | 1.30 | 35.860s | 6.8759 |
| Video-MME | 0.219 req/s | 2.00 | 56.231s | - |
| Video-MME talker | 0.130 req/s | 1.30 | 33.213s | 3.8507 |

## Applied Threshold Policy

Smart apply was used. Automatically tightened speed thresholds were applied,
and user-selected custom or confirmed values were applied for the remaining
interactive metrics. Metrics explicitly kept at the current threshold are not
listed below.

| Stage | Metric | New threshold |
|-------|--------|---------------|
| MMMU speed | throughput / tok/s / latency | 0.70 / 53.1 / 10.6 |
| MMMU talker WER | corpus WER | 0.20 |
| MMMU talker speed | throughput / tok/s / latency / RTF | 0.159 / 5.7 / 23.796 / 0.355 |
| MMSU speed | throughput / tok/s / latency | 29.911 / 7.7 / 0.267 |
| MMSU talker accuracy | accuracy floor | 0.6 |
| MMSU talker speed | tok/s / latency / RTF | 5.0 / 10.08 / 0.3895 |
| TTS WER | corpus WER | 0.03 |
| TTS speed | throughput / tok/s / latency / RTF | 3.986 / 7.5 / 1.95 / 0.5828 |
| Video-AMME talker speed | RTF | 6.8759 |
| Video-MME speed | tok/s | 2.0 |
| Video-MME talker speed | RTF | 3.8507 |

## Notes

Accuracy did not show a broad regression in this run. MMSU text-only was
slightly below the existing 70% threshold at 69.60%, while MMSU talker improved
to 60.00%.

Performance improved strongly for TTS and several talker/text paths. Video
stages were mixed, likely because preprocessing, long prefill, audio synthesis,
ASR, or video decoding can dominate over decode replay.
