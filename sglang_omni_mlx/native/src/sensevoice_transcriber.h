// SPDX-License-Identifier: Apache-2.0
// SenseVoice Small transcription as Voxt's Swift path runs it: one pass with
// the language and inverse text normalization queries, greedy CTC decoding
// with the language, emotion and event the query frames predict, and, for
// long audio, Silero VAD speech cut into overlapping chunks whose texts merge
// as Voxt merges them.
#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "sensevoice_model.h"
#include "silero_vad.h"
#include "swift_port.h"
#include "transcriber.h"

namespace sensevoice {

struct SenseVoiceOptions {
  // Voxt's language hint; zh, en, yue, ja, ko and nospeech pick their query,
  // anything else (or none) detects the language.
  std::optional<std::string> language;
  // Inverse text normalization: written numbers, dates and the like.
  bool use_itn = false;
  // Long audio as Voxt decodes it past 30 s: the speech this Silero VAD finds
  // with these settings, cut into chunks of at most max_chunk_seconds that
  // overlap by chunk_overlap_seconds; no speech is no segments.
  const silero_vad::SileroVAD *voice_activity_detector = nullptr;
  silero_vad::TimestampOptions speech;
  float max_chunk_seconds = 0.0f;
  float chunk_overlap_seconds = 0.0f;
};

class SenseVoiceTranscriber {
public:
  // Reads the model and the directory's SentencePiece model.
  explicit SenseVoiceTranscriber(const std::filesystem::path &model_directory);

  // The text, the language detected (on long audio, the one most chunks with
  // text detect), and one segment per pass with its language, emotion and
  // event, which Voxt merges into its metadata.
  qwen3_asr::TranscriptionResult
  Transcribe(const std::vector<float> &samples,
             const SenseVoiceOptions &options,
             const std::atomic<bool> &cancel) const;

private:
  struct Pass {
    std::string text;
    std::string language;
    std::string emotion;
    std::string event;
    int token_count = 0;
  };

  Pass TranscribePass(const std::vector<float> &samples, int language_id,
                      bool use_itn) const;

  SenseVoiceModel model_;
  swift_port::SentencePieceVocabulary vocabulary_;
};

} // namespace sensevoice
