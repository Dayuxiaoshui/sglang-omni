// SPDX-License-Identifier: Apache-2.0
// Native Cohere Transcribe server: transcriptions over HTTP (JSON or SSE) with
// the API of qwen3_asr_server, without its realtime API, and Voxt's supervisor
// protocol.
//
//   cohere_transcribe_server --model-path DIR [--model-name NAME] [--host H]
//     [--port P]
//   cohere_transcribe_server --supervised --model-kind cohere_transcribe
//     --model-directory DIR
//
// Request fields beside the audio: language (an ISO code or an English name),
// use_punctuation, max_new_tokens, temperature, chunk_duration,
// min_chunk_duration, stream and include_generation_metadata. With
// vad_model_directory (a Silero VAD checkpoint), long audio is cut at speech
// by vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
// vad_merge_gap_seconds and vad_max_chunk_seconds, all required then.
#include <map>
#include <stdexcept>

#include "asr_service.h"
#include "cohere_transcriber.h"
#include "speech_segments.h"

namespace {

template <typename Value>
Value Required(const std::optional<Value> &value, const std::string &name) {
  if (!value.has_value()) {
    throw std::invalid_argument(name + " is required with vad_model_directory");
  } else {
  }
  return *value;
}

class CohereModelService : public asr_service::ServedModel {
public:
  explicit CohereModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    cohere_transcribe::CohereOptions options;
    options.language =
        asr_service::TextField(form, "language").value_or(options.language);
    const auto use_punctuation =
        asr_service::TextField(form, "use_punctuation");
    if (use_punctuation.has_value()) {
      options.use_punctuation = qwen3_asr::FormFlag(use_punctuation);
    } else {
    }
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens")
                                 .value_or(options.max_new_tokens);
    options.temperature = asr_service::NumberField(form, "temperature")
                              .value_or(options.temperature);
    options.chunk_duration_seconds =
        asr_service::NumberField(form, "chunk_duration")
            .value_or(options.chunk_duration_seconds);
    options.min_chunk_duration_seconds =
        asr_service::NumberField(form, "min_chunk_duration")
            .value_or(options.min_chunk_duration_seconds);
    const std::optional<std::string> vad_directory =
        asr_service::TextField(form, "vad_model_directory");
    if (vad_directory.has_value()) {
      options.speech_segments = {
          Required(asr_service::NumberField(form, "vad_threshold"),
                   "vad_threshold"),
          Required(asr_service::IntegerField(form, "vad_min_speech_ms"),
                   "vad_min_speech_ms"),
          Required(asr_service::IntegerField(form, "vad_min_silence_ms"),
                   "vad_min_silence_ms"),
          Required(asr_service::IntegerField(form, "vad_speech_pad_ms"),
                   "vad_speech_pad_ms"),
          Required(asr_service::NumberField(form, "vad_merge_gap_seconds"),
                   "vad_merge_gap_seconds"),
          Required(asr_service::NumberField(form, "vad_max_chunk_seconds"),
                   "vad_max_chunk_seconds")};
    } else {
    }
    return [this, samples = std::move(samples), options,
            vad_directory](const std::atomic<bool> &cancel) mutable {
      if (vad_directory.has_value()) {
        std::unique_ptr<silero_vad::SileroVAD> &detector =
            detectors_[*vad_directory];
        if (detector == nullptr) {
          detector = std::make_unique<silero_vad::SileroVAD>(*vad_directory);
        } else {
        }
        options.voice_activity_detector = detector.get();
      } else {
      }
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

private:
  cohere_transcribe::CohereTranscriber transcriber_;
  // Each Silero VAD loaded on first use and kept, on the worker thread only.
  mutable std::map<std::string, std::unique_ptr<silero_vad::SileroVAD>>
      detectors_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "cohere_transcribe";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<CohereModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
