// SPDX-License-Identifier: Apache-2.0
// Native Qwen3-ASR server with the API of sglang_omni_mlx.qwen3_asr.server:
// asr_service's transcription API and supervisor protocol, plus the realtime
// API on /v1/realtime.
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "asr_service.h"
#include "civetweb.h"
#include "nlohmann/json.hpp"
#include "realtime.h"

namespace {

using qwen3_asr::AudioLayout;
using qwen3_asr::TranscriptionOptions;

struct SocketState {
  std::shared_ptr<qwen3_asr::RealtimeSession> session;
  std::string fragments;
};

class Qwen3ASRModel : public asr_service::ServedModel {
public:
  Qwen3ASRModel(const std::filesystem::path &model_directory,
                qwen3_asr::RealtimeSettings realtime)
      : transcriber_(model_directory), realtime_(realtime) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    TranscriptionOptions options;
    const auto language = asr_service::TextField(form, "language");
    options.language = language.has_value()
                           ? qwen3_asr::NormalizeLanguage(*language)
                           : std::nullopt;
    options.context = asr_service::TextField(form, "prompt");
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens");
    if (options.max_new_tokens.value_or(0) < 0) {
      throw std::invalid_argument("max_new_tokens must not be negative");
    } else {
    }
    options.stop_at_end_of_text = qwen3_asr::FormFlag(
        asr_service::TextField(form, "stop_at_end_of_text"));
    options.stop_on_token_loop =
        qwen3_asr::FormFlag(asr_service::TextField(form, "stop_on_token_loop"));
    const std::string layout =
        asr_service::TextField(form, "audio_layout").value_or("");
    if (layout.empty() || layout == "reference") {
      options.layout = AudioLayout::kReference;
    } else if (layout == "voxt_swift") {
      options.layout = AudioLayout::kVoxtSwift;
    } else {
      throw std::invalid_argument(
          "audio_layout must be reference or voxt_swift");
    }
    return [this, samples = std::move(samples),
            options = std::move(options)](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override {
    worker_ = &worker;
    mg_set_websocket_handler(context, "/v1/realtime", nullptr, SocketReady,
                             SocketData, SocketClosed, this);
  }

private:
  static void SocketReady(mg_connection *connection, void *data) {
    const auto *model = static_cast<Qwen3ASRModel *>(data);
    auto *socket = new SocketState();
    socket->session = std::make_shared<qwen3_asr::RealtimeSession>(
        *model->worker_, model->transcriber_, model->realtime_,
        [connection](const std::string &text) {
          mg_lock_connection(connection);
          const int written = mg_websocket_write(
              connection, MG_WEBSOCKET_OPCODE_TEXT, text.data(), text.size());
          mg_unlock_connection(connection);
          return written > 0;
        });
    mg_set_user_connection_data(connection, socket);
  }

  static int SocketData(mg_connection *connection, int bits, char *data,
                        size_t length, void *) {
    auto *socket =
        static_cast<SocketState *>(mg_get_user_connection_data(connection));
    const int opcode = bits & 0x0F;
    if (socket == nullptr || opcode == MG_WEBSOCKET_OPCODE_CONNECTION_CLOSE) {
      return 0;
    } else if (opcode == MG_WEBSOCKET_OPCODE_PING) {
      mg_lock_connection(connection);
      mg_websocket_write(connection, MG_WEBSOCKET_OPCODE_PONG, data, length);
      mg_unlock_connection(connection);
      return 1;
    } else if (opcode == MG_WEBSOCKET_OPCODE_PONG) {
      return 1;
    } else {
    }
    socket->fragments.append(data, length);
    if ((bits & 0x80) == 0) {
      return 1;
    } else {
    }
    const std::string message = std::move(socket->fragments);
    socket->fragments.clear();
    nlohmann::json event;
    try {
      event = nlohmann::json::parse(message);
    } catch (const nlohmann::json::exception &) {
      socket->session->SendError("invalid_request_error", "invalid_json",
                                 "Events must be JSON.");
      return 1;
    }
    if (!event.is_object()) {
      return 0;
    } else {
    }
    try {
      return socket->session->Handle(event) ? 1 : 0;
    } catch (const std::exception &error) {
      // Note (Jiaxin Deng): log the type alone, never audio or text.
      std::cerr << "realtime session failed: " << typeid(error).name() << "\n";
      return 0;
    }
  }

  static void SocketClosed(const mg_connection *connection, void *) {
    auto *socket =
        static_cast<SocketState *>(mg_get_user_connection_data(connection));
    if (socket != nullptr) {
      socket->session->Close();
      delete socket;
    } else {
    }
  }

  qwen3_asr::Qwen3ASRTranscriber transcriber_;
  const qwen3_asr::RealtimeSettings realtime_;
  qwen3_asr::TranscriptionWorker *worker_ = nullptr;
};

} // namespace

int main(int argc, char **argv) {
  int decode_interval_ms = 1000;
  int first_decode_ms = 100;
  double max_segment_seconds = 30.0;
  asr_service::ServedKind kind;
  kind.model_kind = "qwen3_asr";
  kind.flags = {
      {"--decode-interval-ms",
       [&](const std::string &value) {
         decode_interval_ms = std::stoi(value);
       }},
      {"--first-decode-ms",
       [&](const std::string &value) { first_decode_ms = std::stoi(value); }},
      {"--max-segment-seconds",
       [&](const std::string &value) {
         max_segment_seconds = std::stod(value);
       }},
  };
  kind.check_flags = [&]() {
    const double max_segment_samples =
        max_segment_seconds * qwen3_asr::kSampleRate;
    if (decode_interval_ms <= 0) {
      throw std::invalid_argument("--decode-interval-ms must be positive");
    } else if (first_decode_ms < 0) {
      throw std::invalid_argument("--first-decode-ms must not be negative");
    } else if (!(max_segment_samples >= 1 &&
                 max_segment_samples <= std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "--max-segment-seconds must span one sample to 134217 s");
    } else {
    }
  };
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<Qwen3ASRModel>(
        model_directory,
        qwen3_asr::MakeRealtimeSettings(decode_interval_ms, first_decode_ms,
                                        max_segment_seconds));
  };
  return asr_service::Serve(argc, argv, kind);
}
