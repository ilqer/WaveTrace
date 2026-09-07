#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "core/CsiFrame.hpp"
#include "core/Errors.hpp"

namespace wavetrace {

// CV = sigma/mu; invariant to any positive gain scale k since CV(k*A) = CV(A), so it survives ESP32 AGC
// oscillation unlocked. Two-pass variance avoids float32 cancellation. O(n). Returns 0 when the mean is ~0.
inline float CoefficientOfVariation(const float* amplitudes, size_t sampleCount) {
  if (sampleCount == 0) return 0.0f;
  double mean = 0.0;
  for (size_t i = 0; i < sampleCount; ++i) mean += amplitudes[i];
  mean /= static_cast<double>(sampleCount);
  if (std::fabs(mean) < 1e-12) return 0.0f;
  double var = 0.0;
  for (size_t i = 0; i < sampleCount; ++i) {
    const double d = static_cast<double>(amplitudes[i]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(sampleCount);
  return static_cast<float>(std::sqrt(var) / mean);
}

// Host-side surrogate for the firmware AGC lock: rescales frames to a quiet baseline's median mean magnitude (real-multiply, phase untouched).
// LIMITATION: divides out the frame's own mean, not the true AGC gain, so dynamic-scene amplitude changes get partly removed too — prefer CoefficientOfVariation() then.
class GainLock {
public:
  explicit GainLock(size_t baselinePackets = 300) : baselinePacketCount_(baselinePackets) {
    scales_.reserve(baselinePackets);
  }

  // Accumulate one quiet-baseline frame's overall scale (mean magnitude). O(n).
  void Observe(const CsiFrame& frame) {
    if (bLocked_) throw FrameError("GainLock: cannot observe after Finalize()");
    scales_.push_back(FrameScale(frame));
  }

  size_t ObservedCount() const { return scales_.size(); }
  bool IsReady() const { return scales_.size() >= baselinePacketCount_; }
  bool IsLocked() const { return bLocked_; }
  float ReferenceScale() const { return referenceScale_; }

  // Lock the reference scale = median of the observed per-frame scales. Requires >= 1 observation.
  void Finalize() {
    if (scales_.empty()) throw FrameError("GainLock: no baseline frames observed");
    const size_t mid = scales_.size() / 2;
    std::nth_element(scales_.begin(), scales_.begin() + mid, scales_.end());
    referenceScale_ = scales_[mid];
    bLocked_ = true;
  }

  // Rebuild a locked lock from a persisted reference scale; Apply() only needs referenceScale_ + bLocked_.
  void LockTo(float scale) {
    referenceScale_ = scale;
    bLocked_ = true;
  }

  // Rescale a frame's amplitudes to the locked reference (in place, phase preserved). O(n).
  void Apply(CsiFrame& frame) const {
    if (!bLocked_) throw FrameError("GainLock: Finalize() before Apply()");
    const float currentScale = FrameScale(frame);
    if (currentScale < 1e-12f) return;  // silent/empty frame: nothing to rescale
    const float factor = referenceScale_ / currentScale;
    CsiFrame::Sample* samples = frame.Data();
    const size_t sampleCount = frame.Size();
    for (size_t i = 0; i < sampleCount; ++i) samples[i] *= factor;
  }

private:
  static float FrameScale(const CsiFrame& frame) {
    const CsiFrame::Sample* samples = frame.Data();
    const size_t sampleCount = frame.Size();
    if (sampleCount == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < sampleCount; ++i) sum += std::abs(samples[i]);
    return static_cast<float>(sum / static_cast<double>(sampleCount));
  }

  size_t baselinePacketCount_;  // drives IsReady()
  std::vector<float> scales_;   // per-frame scales observed during calibration
  float referenceScale_ = 0.0f;
  bool bLocked_ = false;
};

}  // namespace wavetrace
