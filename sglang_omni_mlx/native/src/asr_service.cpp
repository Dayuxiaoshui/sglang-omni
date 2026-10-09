// SPDX-License-Identifier: Apache-2.0
#include "asr_service.h"

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <typeinfo>

#include "audio.h"
#include "civetweb.h"
#include "nlohmann/json.hpp"
#include "worker.h"

namespace asr_service {

namespace {

namespace mx = mlx::core;
using Json = nlohmann::ordered_json;
using qwen3_asr::CancelFlag;
using qwen3_asr::TranscriptionResult;
using qwen3_asr::TranscriptionWorker;

constexpr auto kHeartbeatInterval = std::chrono::milliseconds(250);

struct ServerState {
  TranscriptionWorker *worker = nullptr;
  const ServedModel *model = nullptr;
  std::string model_name;
};

void WriteResponse(mg_connection *connection, int status,
                   const std::string &reason, const std::string &content_type,
                   const std::string &body) {
  mg_printf(connection,
            "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: "
            "%zu\r\nConnection: close\r\n\r\n",
            status, reason.c_str(), content_type.c_str(), body.size());
  mg_write(connection, body.data(), body.size());
}

int WriteJson(mg_connection *connection, int status, const Json &body) {
  WriteResponse(connection, status,
                status == 200   ? "OK"
                : status == 400 ? "Bad Request"
                : status == 405 ? "Method Not Allowed"
                                : "Internal Server Error",
                "application/json", body.dump());
  return status;
}

int BadRequest(mg_connection *connection, const std::string &detail) {
  return WriteJson(connection, 400, {{"detail", detail}});
}

int HandleHealth(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  Json states = Json::object();
  for (const auto &[name, count] : state->worker->RequestStates())
    states[name] = count;
  return WriteJson(
      connection, 200,
      {{"status", "healthy"}, {"running", true}, {"request_states", states}});
}

int HandleModels(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  return WriteJson(connection, 200,
                   {{"object", "list"},
                    {"data", Json::array({{{"id", state->model_name},
                                           {"object", "model"}}})}});
}

std::string ReadBody(mg_connection *connection) {
  std::string body;
  char buffer[65536];
  int read = 0;
  while ((read = mg_read(connection, buffer, sizeof(buffer))) > 0) {
    body.append(buffer, static_cast<size_t>(read));
  }
  return body;
}

Json DoneEvent(const TranscriptionResult &result,
               bool include_generation_metadata) {
  Json event = {{"type", "transcript.text.done"}, {"text", result.text}};
  if (include_generation_metadata) {
    event["generation_metadata"] = {
        {"generated_token_count", result.generated_token_count},
        {"language",
         result.language.has_value() ? Json(*result.language) : Json()},
        {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)}};
  } else {
  }
  return event;
}

const Json &FailureEvent() {
  static const Json event = {{"type", "error"},
                             {"error",
                              {{"type", "server_error"},
                               {"code", "transcription_failed"},
                               {"message", "Transcription failed."}}}};
  return event;
}

bool WriteSse(mg_connection *connection, const std::string &payload) {
  const std::string line = "data: " + payload + "\n\n";
  return mg_write(connection, line.data(), line.size()) > 0;
}

int HandleTranscriptions(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  const mg_request_info *request = mg_get_request_info(connection);
  if (std::strcmp(request->request_method, "POST") != 0) {
    return WriteJson(connection, 405, {{"detail", "Method Not Allowed"}});
  } else {
  }
  const char *content_type = mg_get_header(connection, "Content-Type");
  const auto form = qwen3_asr::ParseMultipartForm(
      content_type ? content_type : "", ReadBody(connection));
  if (!form.has_value() || form->count("file") == 0 ||
      !form->at("file").filename.has_value()) {
    return BadRequest(connection, "file is required");
  } else {
  }
  Transcription transcription;
  try {
    transcription = state->model->Prepare(
        qwen3_asr::DecodeWav(form->at("file").value), *form);
  } catch (const std::invalid_argument &error) {
    return BadRequest(connection, error.what());
  } catch (const std::out_of_range &) {
    return BadRequest(connection, "a numeric field is out of range");
  }
  const bool stream = qwen3_asr::FormFlag(TextField(*form, "stream"));
  const bool include_generation_metadata =
      qwen3_asr::FormFlag(TextField(*form, "include_generation_metadata"));
  if (include_generation_metadata && !stream) {
    return BadRequest(connection,
                      "include_generation_metadata requires stream=true");
  } else {
  }
  const CancelFlag cancel = qwen3_asr::NewCancelFlag();
  auto promise = std::make_shared<std::promise<TranscriptionResult>>();
  std::future<TranscriptionResult> future = promise->get_future();
  state->worker->Submit(std::move(transcription), cancel,
                        [promise](std::optional<TranscriptionResult> result,
                                  std::exception_ptr error) {
                          if (error) {
                            promise->set_exception(error);
                          } else {
                            promise->set_value(std::move(*result));
                          }
                        });
  if (!stream) {
    try {
      return WriteJson(connection, 200, {{"text", future.get().text}});
    } catch (...) {
      return WriteJson(connection, 500, {{"detail", "Transcription failed."}});
    }
  } else {
  }
  mg_printf(connection, "HTTP/1.1 200 OK\r\nContent-Type: "
                        "text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Connection: close\r\n\r\n");
  // Note (Jiaxin Deng): SSE comment heartbeats fail to write once the peer is
  // gone, which cancels the decode it was waiting for.
  while (future.wait_for(kHeartbeatInterval) != std::future_status::ready) {
    if (mg_write(connection, ":\n\n", 3) <= 0) {
      cancel->store(true);
    } else {
    }
  }
  try {
    const TranscriptionResult result = future.get();
    WriteSse(connection, DoneEvent(result, include_generation_metadata).dump());
  } catch (const qwen3_asr::TranscriptionCancelled &) {
    return 200;
  } catch (const std::exception &error) {
    std::cerr << "transcription failed: " << typeid(error).name() << "\n";
    WriteSse(connection, FailureEvent().dump());
  }
  WriteSse(connection, "[DONE]");
  return 200;
}

std::string RandomHex(int length) {
  std::mt19937_64 generator{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  for (int i = 0; i < length; ++i)
    text.push_back(kHex[generator() % 16]);
  return text;
}

void Emit(const Json &event) {
  static std::mutex emit_mutex;
  std::lock_guard<std::mutex> lock(emit_mutex);
  std::cout << event.dump() << std::endl;
}

struct Arguments {
  std::string model_path;
  std::string model_name;
  std::string host = "127.0.0.1";
  int port = 0;
  bool supervised = false;
};

Arguments ParseArguments(int argc, char **argv, const ServedKind &kind) {
  Arguments arguments;
  std::string model_kind = kind.model_kind;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(flag + " needs a value");
      } else {
      }
      return argv[++i];
    };
    if (flag == "--model-path" || flag == "--model-directory") {
      arguments.model_path = value();
    } else if (flag == "--model-name") {
      arguments.model_name = value();
    } else if (flag == "--host") {
      arguments.host = value();
    } else if (flag == "--port") {
      arguments.port = std::stoi(value());
    } else if (flag == "--supervised") {
      arguments.supervised = true;
    } else if (flag == "--model-kind") {
      model_kind = value();
    } else if (flag == "--startup-timeout-s") {
      value(); // Note (Jiaxin Deng): Voxt passes it; unused here.
    } else if (kind.flags.count(flag) != 0) {
      kind.flags.at(flag)(value());
    } else {
      throw std::invalid_argument("unknown argument " + flag);
    }
  }
  if (model_kind != kind.model_kind) {
    throw std::invalid_argument("only --model-kind " + kind.model_kind +
                                " is served");
  } else if (arguments.model_path.empty()) {
    throw std::invalid_argument("--model-path is required");
  } else if (kind.check_flags) {
    kind.check_flags();
  } else {
  }
  if (arguments.model_name.empty()) {
    arguments.model_name = "voxt-" + kind.model_kind + "-" + RandomHex(12);
  } else {
  }
  return arguments;
}

class StopSignal {
public:
  void Set(const std::string &reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reason_.empty()) {
      reason_ = reason;
    } else {
    }
    stopped_.notify_all();
  }
  std::string Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    stopped_.wait(lock, [&] { return !reason_.empty(); });
    return reason_;
  }
  std::string Reason() {
    std::lock_guard<std::mutex> lock(mutex_);
    return reason_;
  }

private:
  std::mutex mutex_;
  std::condition_variable stopped_;
  std::string reason_;
};

} // namespace

std::optional<std::string> TextField(const FormFields &form,
                                     const std::string &name) {
  const auto found = form.find(name);
  if (found == form.end()) {
    return std::nullopt;
  } else {
    return found->second.value;
  }
}

std::optional<int> IntegerField(const FormFields &form,
                                const std::string &name) {
  const std::optional<std::string> text = TextField(form, name);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  } else {
  }
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(*text, &parsed);
  } catch (const std::logic_error &) {
    // Note (khazic): parsed stays 0, so the check below names the field.
  }
  if (parsed != text->size()) {
    throw std::invalid_argument(name + " must be an integer");
  } else {
  }
  return value;
}

std::optional<float> NumberField(const FormFields &form,
                                 const std::string &name) {
  const std::optional<std::string> text = TextField(form, name);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  } else {
  }
  size_t parsed = 0;
  float value = 0;
  try {
    value = std::stof(*text, &parsed);
  } catch (const std::logic_error &) {
    // Note (khazic): parsed stays 0, so the check below names the field.
  }
  if (parsed != text->size() || !std::isfinite(value)) {
    throw std::invalid_argument(name + " must be a finite number");
  } else {
  }
  return value;
}

int Serve(int argc, char **argv, const ServedKind &kind) {
  const auto started = std::chrono::steady_clock::now();
  // Note (Jiaxin Deng): stop signals go to one waiting thread, not to
  // whichever thread happens to run.
  sigset_t stop_signals;
  sigemptyset(&stop_signals);
  for (const int signal_number : {SIGTERM, SIGINT, SIGHUP, SIGQUIT})
    sigaddset(&stop_signals, signal_number);
  pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
  signal(SIGPIPE, SIG_IGN);

  Arguments arguments;
  try {
    arguments = ParseArguments(argc, argv, kind);
  } catch (const std::exception &error) {
    std::cerr << argv[0] << ": " << error.what() << "\n";
    return 2;
  }
  StopSignal stop;
  std::thread([&stop, stop_signals]() {
    int signal_number = 0;
    sigwait(&stop_signals, &signal_number);
    stop.Set("signal");
  }).detach();
  if (arguments.supervised) {
    std::thread([&stop]() {
      std::string line;
      while (std::getline(std::cin, line)) {
        try {
          if (nlohmann::json::parse(line).value("command", "") == "shutdown") {
            stop.Set("shutdown");
            return;
          } else {
          }
        } catch (const nlohmann::json::exception &) {
          // Note (Jiaxin Deng): malformed control lines are ignored.
        }
      }
      stop.Set("closed");
    }).detach();
  } else {
  }

  std::unique_ptr<ServedModel> model;
  std::unique_ptr<TranscriptionWorker> worker;
  std::atomic<bool> loaded(false);
  std::thread loader([&]() {
    try {
      worker = std::make_unique<TranscriptionWorker>(
          [&]() { model = kind.load(arguments.model_path); });
      loaded.store(true);
    } catch (const std::exception &error) {
      if (arguments.supervised) {
        Emit({{"event", "failed"},
              {"reason", std::string("model load failed: ") + error.what()}});
      } else {
        std::cerr << "model load failed: " << error.what() << "\n";
      }
      std::_Exit(1);
    }
  });
  // Note (Jiaxin Deng): a stop before the model is ready ends the process at
  // once.
  while (!loaded.load()) {
    const std::string reason = stop.Reason();
    if (!reason.empty()) {
      if (reason == "shutdown") {
        Emit({{"event", "stopped"}});
      } else {
      }
      std::_Exit(0);
    } else {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  loader.join();

  ServerState state{worker.get(), model.get(), arguments.model_name};
  mg_init_library(0);
  const std::string listening =
      arguments.host + ":" + std::to_string(arguments.port);
  const char *options[] = {"listening_ports",
                           listening.c_str(),
                           "num_threads",
                           "16",
                           "request_timeout_ms",
                           "3600000",
                           "websocket_timeout_ms",
                           "3600000",
                           nullptr};
  mg_callbacks callbacks{};
  mg_context *context = mg_start(&callbacks, &state, options);
  if (context == nullptr) {
    if (arguments.supervised) {
      Emit({{"event", "failed"}, {"reason", "cannot listen on " + listening}});
    } else {
      std::cerr << "cannot listen on " << listening << "\n";
    }
    return 1;
  } else {
  }
  // Note (Jiaxin Deng): port 0 lets the system pick a free port at bind time.
  mg_server_port server_port{};
  mg_get_server_ports(context, 1, &server_port);
  const std::string endpoint =
      arguments.host + ":" + std::to_string(server_port.port);
  mg_set_request_handler(context, "/health$", HandleHealth, &state);
  mg_set_request_handler(context, "/v1/models$", HandleModels, &state);
  mg_set_request_handler(context, "/v1/audio/transcriptions$",
                         HandleTranscriptions, &state);
  model->AddHandlers(context, *worker);
  const double startup_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  if (arguments.supervised) {
    Emit({{"event", "ready"},
          {"host", arguments.host},
          {"port", server_port.port},
          {"model_name", arguments.model_name},
          {"server_pid", static_cast<int>(getpid())},
          {"startup_s", std::round(startup_seconds * 1000) / 1000}});
  } else {
    std::cerr << "serving " << arguments.model_name << " on " << endpoint
              << " after " << startup_seconds << " s\n";
  }

  const std::string reason = stop.Wait();
  // Note (Jiaxin Deng): cancel and exit at once; civetweb's own stop waits out
  // its 2 s poll quantum, and the owner has no use for the open responses.
  worker->CancelAll();
  if (reason == "shutdown") {
    Emit({{"event", "stopped"}});
  } else {
  }
  std::_Exit(0);
}

} // namespace asr_service
