// SPDX-License-Identifier: Apache-2.0
#include "asr_service.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
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

constexpr auto kHeartbeatInterval = std::chrono::milliseconds(250);

// Runs every transcription on one thread, the thread that loaded the model:
// one model, one MLX stream, one request at a time.
class ModelWorker {
public:
  // Called on the worker thread with the result, or with the exception the
  // transcription raised (TranscriptionCancelled included).
  using Completion = std::function<void(std::optional<TranscriptionResult>,
                                        std::exception_ptr)>;

  // Loads the model on the worker thread; throws what loading throws.
  ModelWorker(const ModelLoader &load,
              const std::filesystem::path &model_directory) {
    std::promise<void> loaded;
    std::future<void> loaded_future = loaded.get_future();
    thread_ = std::thread(&ModelWorker::Run, this, std::move(loaded), load,
                          model_directory);
    try {
      loaded_future.get();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
      }
      wake_.notify_all();
      thread_.join();
      throw;
    }
  }

  ~ModelWorker() {
    CancelAll();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    } else {
    }
  }

  ModelWorker(const ModelWorker &) = delete;
  ModelWorker &operator=(const ModelWorker &) = delete;

  const ServedModel &model() const { return *model_; }

  void Submit(Transcription transcription, CancelFlag cancel,
              Completion completion) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(
          {std::move(transcription), std::move(cancel), std::move(completion)});
    }
    wake_.notify_one();
  }

  // Requests waiting for the worker and running on it; empty when idle.
  std::map<std::string, int> RequestStates() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, int> states;
    if (running_) {
      states["running"] = 1;
    } else {
    }
    if (!queue_.empty()) {
      states["queued"] = static_cast<int>(queue_.size());
    } else {
    }
    return states;
  }

  // Cancels everything queued or running, as on shutdown.
  void CancelAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Job &job : queue_) {
      job.cancel->store(true);
    }
    if (running_cancel_) {
      running_cancel_->store(true);
    } else {
    }
  }

private:
  struct Job {
    Transcription transcription;
    CancelFlag cancel;
    Completion completion;
  };

  void Run(std::promise<void> loaded, ModelLoader load,
           std::filesystem::path model_directory) {
    try {
      model_ = load(model_directory);
      loaded.set_value();
    } catch (...) {
      loaded.set_exception(std::current_exception());
      return;
    }
    while (true) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        // Decodes keep MLX's buffer cache, since freeing every step's buffers
        // slows each token. Once nothing is queued it goes back to the
        // system, so an idle server holds only the model. The wait lets the
        // step a decode queued ahead before it stopped return its buffers.
        if (queue_.empty()) {
          lock.unlock();
          mx::synchronize();
          mx::clear_cache();
          lock.lock();
        } else {
        }
        wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
          return;
        } else {
        }
        job = std::move(queue_.front());
        queue_.pop_front();
        running_ = true;
        running_cancel_ = job.cancel;
      }
      std::optional<TranscriptionResult> result;
      std::exception_ptr error;
      try {
        result = job.transcription(*job.cancel);
      } catch (...) {
        error = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        running_cancel_.reset();
      }
      job.completion(std::move(result), error);
    }
  }

  std::unique_ptr<ServedModel> model_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Job> queue_;
  CancelFlag running_cancel_;
  bool running_ = false;
  bool stopping_ = false;
  std::thread thread_;
};

struct ServerState {
  ModelWorker *worker = nullptr;
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
                                : "Error",
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
    transcription = state->worker->model().Prepare(
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
  // A client that disconnects stops the decode it was waiting for: SSE
  // comment lines fail to write once the peer is gone.
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

int FreeLoopbackPort() {
  const int descriptor = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  bind(descriptor, reinterpret_cast<sockaddr *>(&address), sizeof(address));
  socklen_t length = sizeof(address);
  getsockname(descriptor, reinterpret_cast<sockaddr *>(&address), &length);
  close(descriptor);
  return ntohs(address.sin_port);
}

std::string RandomHex(int length) {
  std::mt19937_64 generator{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  for (int i = 0; i < length; ++i)
    text.push_back(kHex[generator() % 16]);
  return text;
}

// Emits one supervisor event line on stdout.
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

Arguments ParseArguments(int argc, char **argv,
                         const std::string &served_model_kind) {
  Arguments arguments;
  std::string model_kind = served_model_kind;
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
    } else if (flag == "--derived-root" || flag == "--startup-timeout-s") {
      value(); // Accepted for the supervisor's command line; not needed here.
    } else {
      throw std::invalid_argument("unknown argument " + flag);
    }
  }
  if (model_kind != served_model_kind) {
    throw std::invalid_argument("only --model-kind " + served_model_kind +
                                " is served");
  } else if (arguments.model_path.empty()) {
    throw std::invalid_argument("--model-path is required");
  } else {
  }
  if (arguments.model_name.empty()) {
    arguments.model_name = "voxt-" + served_model_kind + "-" + RandomHex(12);
  } else {
  }
  if (arguments.port == 0) {
    arguments.port = FreeLoopbackPort();
  } else {
  }
  return arguments;
}

// Why the server stops: a shutdown command, end of stdin, or a signal.
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
  const int value = std::stoi(*text, &parsed);
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
  const float value = std::stof(*text, &parsed);
  if (parsed != text->size()) {
    throw std::invalid_argument(name + " must be a number");
  } else {
  }
  return value;
}

int Serve(int argc, char **argv, const std::string &model_kind,
          const ModelLoader &load) {
  const auto started = std::chrono::steady_clock::now();
  // Stop signals go to one waiting thread, not to whichever thread runs.
  sigset_t stop_signals;
  sigemptyset(&stop_signals);
  for (const int signal_number : {SIGTERM, SIGINT, SIGHUP, SIGQUIT})
    sigaddset(&stop_signals, signal_number);
  pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
  signal(SIGPIPE, SIG_IGN);

  Arguments arguments;
  try {
    arguments = ParseArguments(argc, argv, model_kind);
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
          // Malformed control lines are ignored.
        }
      }
      stop.Set("closed");
    }).detach();
  } else {
  }

  std::unique_ptr<ModelWorker> worker;
  std::atomic<bool> loaded(false);
  std::thread loader([&]() {
    try {
      worker = std::make_unique<ModelWorker>(load, arguments.model_path);
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
  // A stop before the model is ready ends the process at once.
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

  ServerState state{worker.get(), arguments.model_name};
  mg_init_library(0);
  const std::string listening =
      arguments.host + ":" + std::to_string(arguments.port);
  const char *options[] = {
      "listening_ports",    listening.c_str(), "num_threads", "16",
      "request_timeout_ms", "3600000",         nullptr};
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
  mg_set_request_handler(context, "/health$", HandleHealth, &state);
  mg_set_request_handler(context, "/v1/models$", HandleModels, &state);
  mg_set_request_handler(context, "/v1/audio/transcriptions$",
                         HandleTranscriptions, &state);
  const double startup_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  if (arguments.supervised) {
    Emit({{"event", "ready"},
          {"host", arguments.host},
          {"port", arguments.port},
          {"model_name", arguments.model_name},
          {"server_pid", static_cast<int>(getpid())},
          {"startup_s", std::round(startup_seconds * 1000) / 1000}});
  } else {
    std::cerr << "serving " << arguments.model_name << " on " << listening
              << " after " << startup_seconds << " s\n";
  }

  const std::string reason = stop.Wait();
  // Every decode in flight is cancelled and the process exits at once:
  // civetweb's own stop waits out its 2 s poll quantum, and an owner stopping
  // the server has no use for the open responses.
  worker->CancelAll();
  if (reason == "shutdown") {
    Emit({{"event", "stopped"}});
  } else {
  }
  std::_Exit(0);
}

} // namespace asr_service
