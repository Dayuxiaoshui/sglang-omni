// SPDX-License-Identifier: Apache-2.0
// Sortformer speaker diarization over the native server, for Voxt's meeting
// speaker analysis:
//
//   WS /v1/diarization/stream[?threshold=&min_duration=&merge_gap=
//                              &spkcache_max=&fifo_max=]
//       One socket per audio stream; the server keeps its StreamingState.
//       Every binary message is one feed (16 kHz float32 little-endian
//       samples, as Swift SortformerModel.feed takes a chunk) and is answered
//       with that feed's results:
//         {"frames": n, "speakers": k,
//          "probabilities": [[p_0 .. p_k-1] x n],
//          "segments": [{"start", "end", "speaker"}],   seconds from the
//                                                         stream's start
//          "state": {"fifo_length", "spkcache_length", "frames_processed"}}
//       The options default to the values Voxt passes (threshold 0.5,
//       min_duration 0, merge_gap 0.18, spkcache_max 188, fifo_max 188).
#pragma once

#include <filesystem>
#include <map>
#include <mutex>
#include <string>

#include "civetweb.h"
#include "sortformer.h"

namespace sortformer {

// Longest feed accepted at all: Voxt feeds 4.96 s; the encoder attends over
// the whole feed, so an unbounded one could exhaust memory. A checkpoint can
// lower it (SortformerService::MaxFeedSamples).
inline constexpr size_t kMaxFeedSamples = 16000 * 30;

class SortformerService {
public:
  explicit SortformerService(const std::filesystem::path &model_directory);

  void Register(mg_context *context);
  // Request counts for /health.
  std::map<std::string, int> RequestStates() const;

  StreamingState NewStream() const { return model_.InitStreamingState(); }
  // The longest feed: one frame short of the speaker cache update period, so
  // a feed's frames leave the FIFO in one update and the FIFO stays within
  // fifo_max however many feeds come.
  size_t MaxFeedSamples() const { return max_feed_samples_; }
  // Throws std::invalid_argument when the speaker cache, the FIFO, the left
  // context and the longest feed would not fit the transformer's positions.
  void CheckStateLimits(const FeedOptions &options) const;
  // One feed; inference runs one call at a time across all streams.
  FeedResult Feed(const std::vector<float> &samples, StreamingState &state,
                  const FeedOptions &options);

  void StreamOpened();
  void StreamClosed();

private:
  SortformerModel model_;
  size_t max_feed_samples_ = kMaxFeedSamples;
  std::mutex inference_mutex_;
  mutable std::mutex states_mutex_;
  int open_streams_ = 0;
  int running_requests_ = 0;
};

// Options from a query string ("threshold=0.4&merge_gap=0"); throws
// std::invalid_argument for unknown names or invalid values.
FeedOptions ParseFeedOptions(const std::string &query);

} // namespace sortformer
