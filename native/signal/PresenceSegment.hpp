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
  // enterCoefficientOfVariation/exitCoefficientOfVariation: hysteresis thresholds (enter >= exit);
  // minActiveLen: shorter segments suppressed as noise.
  PresenceSegmenter(size_t window, float enterCoefficientOfVariation, float exitCoefficientOfVariation,
                     size_t minActiveLen = 1)
      : window_(window),
        enterCoefficientOfVariation_(enterCoefficientOfVariation),
        exitCoefficientOfVariation_(exitCoefficientOfVariation),
        minActiveLen_(minActiveLen),
        ring_(window) {
    if (window == 0) throw WaveTraceError("PresenceSegmenter: window must be non-zero");
    if (enterCoefficientOfVariation < exitCoefficientOfVariation)
      throw WaveTraceError("PresenceSegmenter: enterCoefficientOfVariation must be >= exitCoefficientOfVariation");
    win_.assign(window_, 0.0f);
  }

  // Returns whether now active; when a segment closes this push, HasSegmentClosed()==true and LastSegment{Start,End}() give its half-open [start, end) bounds.
  bool Push(const float* magnitudes, size_t subcarrierCount) {
    double sum = 0.0;
    for (size_t i = 0; i < subcarrierCount; ++i) sum += static_cast<double>(magnitudes[i]);
    const float energy =
        (subcarrierCount > 0) ? static_cast<float>(sum / static_cast<double>(subcarrierCount)) : 0.0f;
    ring_.Push(energy);

    activity_ = WindowCoefficientOfVariation();
    bSegmentClosed_ = false;
    const size_t idx = frame_++;

    if (!bActive_) {
      if (ring_.Size() >= window_ && activity_ >= enterCoefficientOfVariation_) {
        bActive_ = true;
        segStart_ = idx;
      }
    } else if (activity_ <= exitCoefficientOfVariation_) {
      bActive_ = false;
      if (idx - segStart_ >= minActiveLen_) {  // suppress sub-threshold-length noise bursts
        bSegmentClosed_ = true;
        lastStart_ = segStart_;
        lastEnd_ = idx;  // half-open: last active frame was idx-1
      }
    }
    return bActive_;
  }

  bool IsActive() const { return bActive_; }
  float Activity() const { return activity_; }       // last windowed CV (sigma/mu)
  bool HasSegmentClosed() const { return bSegmentClosed_; }
  size_t LastSegmentStart() const { return lastStart_; }
  size_t LastSegmentEnd() const { return lastEnd_; }
  size_t CurrentStart() const { return segStart_; }   // valid only while IsActive()
  size_t Window() const { return window_; }

  void Reset() {
    ring_.Clear();
    bActive_ = false;
    bSegmentClosed_ = false;
    frame_ = 0;
    activity_ = 0.0f;
    segStart_ = lastStart_ = lastEnd_ = 0;
  }

private:
  float WindowCoefficientOfVariation() {
    const size_t n = ring_.Size();
    if (n < 2) return 0.0f;
    ring_.CopyTo(win_.data());  // order-independent (variance) — the unordered copy is fine
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
  float enterCoefficientOfVariation_, exitCoefficientOfVariation_;
  size_t minActiveLen_;
  RingBuffer<float> ring_;
  std::vector<float> win_;
  bool bActive_ = false, bSegmentClosed_ = false;
  size_t frame_ = 0, segStart_ = 0, lastStart_ = 0, lastEnd_ = 0;
  float activity_ = 0.0f;
};

}  // namespace wavetrace
