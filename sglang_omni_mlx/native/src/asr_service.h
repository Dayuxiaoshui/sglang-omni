// SPDX-License-Identifier: Apache-2.0
// One model kind served the way qwen3_asr_server serves Qwen3-ASR: the same
// transcription API (JSON or SSE) and Voxt's supervisor protocol, without the
// realtime API. Each model binary supplies how it loads and reads a request.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "form.h"
#include "transcriber.h"

namespace asr_service {

using FormFields = std::map<std::string, qwen3_asr::FormField>;
// One transcription bound to its audio and options; runs on the worker
// thread, the thread that loaded the model.
using Transcription =
    std::function<qwen3_asr::TranscriptionResult(const std::atomic<bool> &)>;

// A loaded checkpoint of the served kind.
class ServedModel {
public:
  virtual ~ServedModel() = default;
  // Binds one request's form fields to its samples; throws
  // std::invalid_argument for a field it does not accept.
  virtual Transcription Prepare(std::vector<float> samples,
                                const FormFields &form) const = 0;
};

using ModelLoader =
    std::function<std::unique_ptr<ServedModel>(const std::filesystem::path &)>;

// A form field as given, or as an integer or a finite number; the latter two
// throw std::invalid_argument when the field is not one, and leave out empty
// fields.
std::optional<std::string> TextField(const FormFields &form,
                                     const std::string &name);
std::optional<int> IntegerField(const FormFields &form,
                                const std::string &name);
std::optional<float> NumberField(const FormFields &form,
                                 const std::string &name);

// Runs the server for model_kind until it is stopped; returns the exit code.
//
//   BINARY --model-path DIR [--model-name NAME] [--host H] [--port P]
//   BINARY --supervised --model-kind KIND --model-directory DIR
int Serve(int argc, char **argv, const std::string &model_kind,
          const ModelLoader &load);

} // namespace asr_service
