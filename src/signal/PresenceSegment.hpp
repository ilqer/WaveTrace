#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

#include "core/Errors.hpp"
#include "util/RingBuffer.hpp"

namespace wavetrace {

// Streaming presence / active-segment detector (Option A — variance gate; the lightweight stand-in
// for Zhou's LOF active-segment extraction, "Detection of Suspicious Objects Concealed by Walking
// Pedestrians"). Per frame it reduces the K subcarrier magnitudes to the mean channel energy, tracks
// that scalar over a sliding window, and flags an ACTIVE segment when its coefficient of variation
// (sigma/mu — gain-invariant, so AGC drift is not mistaken for motion) crosses an ENTER threshold; it
// leaves the segment when the CV falls back below a (lower) EXIT threshold. The hysteresis gap
// prevents chattering at the boundary. This auto-triggers "someone is here" for continuous operation
// and bounds the window a (future) weapon head votes over. Per-frame O(W) (two-pass CV over the small
// window, the §2.8 stable choice over a running variance); no hot-path allocation beyond the ctor.
//
// Axis note: this windowed (TEMPORAL) CV is orthogonal to interCarrierStats' sigma2[p] (CROSS-
// subcarrier at one packet, the metal discriminator) — this measures MOTION over time, that measures
// material at an instant. Push antenna-collapsed per-frame magnitudes (e.g. np.abs(grid).mean(0)).
class PresenceSegmenter {
public:
  // window: frames in the moving-CV window; enterCv/exitCv: hysteresis thresholds (enter >= exit);
  // minActiveLen: segments shorter than this (frames) are suppressed as noise (default 1 = keep all).
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

  // Push one frame's subcarrier magnitudes; returns whether the detector is now inside an active
  // segment. When a segment closes on this push, segmentClosed()==true and lastSegment{Start,End}()
  // give its half-open [start, end) frame bounds.
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
