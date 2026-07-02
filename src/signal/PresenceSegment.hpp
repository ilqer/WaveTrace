#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

#include "core/Errors.hpp"
#include "util/RingBuffer.hpp"

namespace wavetrace {

// Streaming presence detector: tracks mean channel energy's CV (sigma/mu, gain-invariant vs AGC drift) over a sliding
// window, flagging ACTIVE above an ENTER threshold and leaving below a lower EXIT threshold (hysteresis). Per-frame O(W).
class PresenceSegmenter {
public:
  // enterCv/exitCv: hysteresis thresholds (enter >= exit); minActiveLen: shorter segments suppressed as noise.
  PresenceSegmenter(size_t window, float enterCv, float exitCv, size_t minActiveLen = 1)
      : window_(window),
        enterCv_(enterCv),
        exitCv_(exitCv),
        minActiveLen_(minActiveLen),
        ring_(window) {
    if (window == 0) throw WaveTraceError("PresenceSegmenter: window must be non-zero");
    if (enterCv < exitCv) throw WaveTraceError("PresenceSegmenter: enterCv must be >= exitCv");
    win_.assign(window_, 0.0f);
  }

  // Returns whether now active; when a segment closes this push, segmentClosed()==true and lastSegment{Start,End}() give its half-open [start, end) bounds.
  bool push(const float* mags, size_t k) {
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) sum += static_cast<double>(mags[i]);
    const float energy = (k > 0) ? static_cast<float>(sum / static_cast<double>(k)) : 0.0f;
    ring_.push(energy);

    activity_ = windowCv();
    segmentClosed_ = false;
    const size_t idx = frame_++;

    if (!active_) {
      if (ring_.size() >= window_ && activity_ >= enterCv_) {
        active_ = true;
        segStart_ = idx;
      }
    } else if (activity_ <= exitCv_) {
      active_ = false;
      if (idx - segStart_ >= minActiveLen_) {  // suppress sub-threshold-length noise bursts
        segmentClosed_ = true;
        lastStart_ = segStart_;
        lastEnd_ = idx;  // half-open: last active frame was idx-1
      }
    }
    return active_;
  }

  bool active() const { return active_; }
  float activity() const { return activity_; }       // last windowed CV (sigma/mu)
  bool segmentClosed() const { return segmentClosed_; }
  size_t lastSegmentStart() const { return lastStart_; }
  size_t lastSegmentEnd() const { return lastEnd_; }
  size_t currentStart() const { return segStart_; }   // valid only while active()
  size_t window() const { return window_; }

  void reset() {
    ring_.clear();
    active_ = false;
    segmentClosed_ = false;
    frame_ = 0;
    activity_ = 0.0f;
    segStart_ = lastStart_ = lastEnd_ = 0;
  }

private:
  float windowCv() {
    const size_t n = ring_.size();
    if (n < 2) return 0.0f;
    ring_.copyTo(win_.data());  // order-independent (variance) — the unordered copy is fine
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += static_cast<double>(win_[i]);
    mean /= static_cast<double>(n);
    if (mean <= 1e-12) return 0.0f;
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double d = static_cast<double>(win_[i]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(n - 1);  // sample variance (M-1)
    return static_cast<float>(std::sqrt(var) / mean);
  }

  size_t window_;
  float enterCv_, exitCv_;
  size_t minActiveLen_;
  RingBuffer<float> ring_;
  std::vector<float> win_;
  bool active_ = false, segmentClosed_ = false;
  size_t frame_ = 0, segStart_ = 0, lastStart_ = 0, lastEnd_ = 0;
  float activity_ = 0.0f;
};

}  // namespace wavetrace
