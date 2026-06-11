#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "core/CsiFrame.hpp"
#include "core/Errors.hpp"

namespace wavetrace {

// Coefficient of Variation = sigma/mu over a set of amplitudes. Invariant to any positive gain
// scale k, since CV(k*A) = (k*sigma)/(k*mu) = CV(A) — so amplitude variability survives ESP32 AGC
// oscillation with NO lock at all (REFERENCE_DIGEST §2.11). Two-pass (not running) variance to
// avoid float32 cancellation (REFERENCE §2.8). O(n). Returns 0 when the mean is ~0.
inline float coefficientOfVariation(const float* amp, size_t n) {
  if (n == 0) return 0.0f;
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i) mean += amp[i];
  mean /= static_cast<double>(n);
  if (std::fabs(mean) < 1e-12) return 0.0f;
  double var = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(amp[i]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(n);
  return static_cast<float>(std::sqrt(var) / mean);
}

// Software gain stabilization for ESP32 AGC (host-side surrogate for the firmware PHY lock, which
// is out of our scope). The AGC rescales every packet's amplitudes by a per-frame gain k_t, which
// wrecks amplitude features. We learn a reference amplitude scale from a QUIET baseline (median of
// per-frame mean magnitude over ~300 packets — median is robust to a stray moving frame), then
// rescale each later frame's amplitudes to that reference. Multiplying by a positive real leaves
// phase untouched, so the phase pipeline (Preprocessor) is unaffected; this is purely for the
// amplitude features (Phase 4).
//
// LIMITATION (read before relying on it): the per-frame scale we divide out is the frame's own mean
// magnitude, not the chip's reported AGC gain. On a static/quiet scene that IS the AGC term, but in
// a DYNAMIC scene real amplitude changes also move the mean and would be partly removed. When that
// matters, prefer coefficientOfVariation() (gain-invariant, no normalization artifact). A true lock
// needs the AGC value decoded from firmware — a future hardware task.
class GainLock {
public:
  explicit GainLock(size_t baselinePackets = 300) : baseline_(baselinePackets) {
    scales_.reserve(baselinePackets);
  }

  // Accumulate one quiet-baseline frame's overall scale (mean magnitude). O(n).
  void observe(const CsiFrame& frame) {
    if (locked_) throw FrameError("GainLock: cannot observe after finalize()");
    scales_.push_back(frameScale(frame));
  }

  size_t observed() const { return scales_.size(); }
  bool ready() const { return scales_.size() >= baseline_; }  // enough baseline collected?
  bool locked() const { return locked_; }
  float referenceScale() const { return referenceScale_; }

  // Lock the reference scale = median of the observed per-frame scales. Requires >= 1 observation.
  void finalize() {
    if (scales_.empty()) throw FrameError("GainLock: no baseline frames observed");
    const size_t mid = scales_.size() / 2;
    std::nth_element(scales_.begin(), scales_.begin() + mid, scales_.end());
    referenceScale_ = scales_[mid];
    locked_ = true;
  }

  // Rebuild a locked lock from a persisted reference scale (skips re-observing a baseline). apply()
  // needs only referenceScale_ + locked_, so a calibration can serialize the scalar and restore.
  void lockTo(float scale) {
    referenceScale_ = scale;
    locked_ = true;
  }

  // Rescale a frame's amplitudes to the locked reference (in place, phase preserved). O(n).
  void apply(CsiFrame& frame) const {
    if (!locked_) throw FrameError("GainLock: finalize() before apply()");
    const float s = frameScale(frame);
    if (s < 1e-12f) return;  // silent/empty frame: nothing to rescale
    const float factor = referenceScale_ / s;
    CsiFrame::Sample* g = frame.data();
    const size_t n = frame.size();
    for (size_t i = 0; i < n; ++i) g[i] *= factor;
  }

private:
  static float frameScale(const CsiFrame& frame) {
    const CsiFrame::Sample* g = frame.data();
    const size_t n = frame.size();
    if (n == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += std::abs(g[i]);
    return static_cast<float>(sum / static_cast<double>(n));
  }

  size_t baseline_;             // suggested baseline packet count (drives ready())
  std::vector<float> scales_;   // per-frame scales observed during calibration
  float referenceScale_ = 0.0f;
  bool locked_ = false;
};

}  // namespace wavetrace
