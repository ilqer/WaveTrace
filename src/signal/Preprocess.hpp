#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/CsiFrame.hpp"
#include "core/Errors.hpp"
#include "util/RingBuffer.hpp"

namespace wavetrace {

inline constexpr float WT_PI = 3.14159265358979323846f;
inline constexpr float WT_TWO_PI = 2.0f * WT_PI;

// --- Stateless transforms (also bound individually for unit tests) --------------------------

// Geometry-adaptive conjugate multiply cancels common-mode clock drift (CFO/SFO): >=2 antennas uses
// cross-antenna out[a-1][k]=H[a][k]*conj(H[0][k]); 1 antenna falls back to cross-subcarrier out[k-1]=H[k]*conj(H[k-1]). O(n).
inline void conjugateMultiply(const CsiFrame& in, CsiFrame& out) {
  const uint16_t A = in.numAntennas();
  const uint16_t S = in.numSubcarriers();
  const CsiFrame::Sample* H = in.data();
  if (A >= 2) {
    out.reshape(static_cast<uint16_t>(A - 1), S);
    CsiFrame::Sample* D = out.data();
    const CsiFrame::Sample* ref = H;  // antenna 0 = shared-clock reference
    for (uint16_t a = 1; a < A; ++a) {
      const CsiFrame::Sample* row = H + static_cast<size_t>(a) * S;
      CsiFrame::Sample* outRow = D + static_cast<size_t>(a - 1) * S;
      for (uint16_t k = 0; k < S; ++k) outRow[k] = row[k] * std::conj(ref[k]);
    }
  } else {
    if (S < 2) throw FrameError("conjugateMultiply: single antenna needs >= 2 subcarriers");
    out.reshape(1, static_cast<uint16_t>(S - 1));
    CsiFrame::Sample* D = out.data();
    for (uint16_t k = 1; k < S; ++k) D[k - 1] = H[k] * std::conj(H[k - 1]);
  }
}

// Antenna-difference out[a-1][k]=H[a][k]-H[0][k]: on one shared-clock radio this nulls the common environment
// (LOS + static furniture) and amplifies per-antenna scattering, unlike conjugateMultiply which cancels clock drift. O(n).
inline void combinedChannelDifference(const CsiFrame& in, CsiFrame& out) {
  const uint16_t A = in.numAntennas();
  const uint16_t S = in.numSubcarriers();
  if (A < 2) throw FrameError("combinedChannelDifference: requires >= 2 antennas on one radio");
  out.reshape(static_cast<uint16_t>(A - 1), S);
  const CsiFrame::Sample* H = in.data();
  CsiFrame::Sample* D = out.data();
  const CsiFrame::Sample* ref = H;  // antenna 0 = common reference
  for (uint16_t a = 1; a < A; ++a) {
    const CsiFrame::Sample* row = H + static_cast<size_t>(a) * S;
    CsiFrame::Sample* outRow = D + static_cast<size_t>(a - 1) * S;
    for (uint16_t k = 0; k < S; ++k) outRow[k] = row[k] - ref[k];
  }
}

// Hampel outlier test: returns `current` unless it deviates beyond k*1.4826*MAD from the window median
// (1.4826 makes MAD a consistent sigma estimator for Gaussians), else returns the median. O(w) (nth_element).
inline float hampel(const float* window, size_t w, float current, float* scratch, float k) {
  if (w == 0) return current;
  const size_t mid = w / 2;
  for (size_t i = 0; i < w; ++i) scratch[i] = window[i];
  std::nth_element(scratch, scratch + mid, scratch + w);
  const float med = scratch[mid];
  // nth_element only permuted scratch, so its median-of-deviations here is still the MAD.
  for (size_t i = 0; i < w; ++i) scratch[i] = std::fabs(scratch[i] - med);
  std::nth_element(scratch, scratch + mid, scratch + w);
  const float mad = scratch[mid];
  if (mad > 0.0f && std::fabs(current - med) > k * 1.4826f * mad) return med;
  return current;
}

// One streaming phase-unwrap step: bring the step from the previous wrapped phase into (-pi, pi] and add it to the running unwrapped value. O(1).
inline float unwrapStep(float curWrapped, float prevWrapped, float prevUnwrapped) {
  float d = curWrapped - prevWrapped;
  while (d > WT_PI) d -= WT_TWO_PI;
  while (d < -WT_PI) d += WT_TWO_PI;
  return prevUnwrapped + d;
}

// --- Streaming preprocessor (the hot-path chain) --------------------------------------------

// Per-frame DSP front-end: conjugate-multiply -> Hampel -> unwrap -> normalize into a drift-free, spike-cleaned,
// detrended differential-phase grid; stateful, alloc-free after ctor. Hampel gates on magnitude; on a spike the phase holds at the last good value.
class Preprocessor {
public:
  Preprocessor(uint16_t numAntennas, uint16_t numSubcarriers, size_t hampelWindow = 7,
               float hampelK = 5.0f, float normalizeAlpha = 0.1f)
      : inA_(numAntennas),
        inS_(numSubcarriers),
        hampelK_(hampelK),
        normAlpha_(normalizeAlpha) {
    if (numAntennas == 0 || numSubcarriers == 0) {
      throw FrameError("Preprocessor: dimensions must be non-zero");
    }
    if (numAntennas >= 2) {
      outRows_ = static_cast<uint16_t>(numAntennas - 1);
      outCols_ = numSubcarriers;
    } else {
      if (numSubcarriers < 2) throw FrameError("Preprocessor: single antenna needs >= 2 subcarriers");
      outRows_ = 1;
      outCols_ = static_cast<uint16_t>(numSubcarriers - 1);
    }
    const size_t cells = static_cast<size_t>(outRows_) * outCols_;
    output_.assign(cells, 0.0f);
    lastPhase_.assign(cells, 0.0f);
    prevWrapped_.assign(cells, 0.0f);
    prevUnwrapped_.assign(cells, 0.0f);
    ema_.assign(cells, 0.0f);
    hasPrev_.assign(cells, 0);
    emaInit_.assign(cells, 0);
    mags_.reserve(cells);
    for (size_t c = 0; c < cells; ++c) mags_.emplace_back(hampelWindow);
    window_.assign(hampelWindow, 0.0f);
    scratch_.assign(hampelWindow, 0.0f);
  }

  uint16_t outRows() const { return outRows_; }
  uint16_t outCols() const { return outCols_; }
  const float* data() const { return output_.data(); }

  // Process one frame; result is in the reused output_ grid (returned via data()). O(n)/frame.
  void process(const CsiFrame& in) {
    if (in.numAntennas() != inA_ || in.numSubcarriers() != inS_) {
      throw FrameError("Preprocessor: frame geometry does not match configuration");
    }
    const CsiFrame::Sample* H = in.data();
    size_t c = 0;
    if (inA_ >= 2) {
      const CsiFrame::Sample* ref = H;
      for (uint16_t a = 1; a < inA_; ++a) {
        const CsiFrame::Sample* row = H + static_cast<size_t>(a) * inS_;
        for (uint16_t k = 0; k < inS_; ++k, ++c) processCell(c, row[k] * std::conj(ref[k]));
      }
    } else {
      for (uint16_t k = 1; k < inS_; ++k, ++c) processCell(c, H[k] * std::conj(H[k - 1]));
    }
  }

  void reset() {
    std::fill(output_.begin(), output_.end(), 0.0f);
    std::fill(ema_.begin(), ema_.end(), 0.0f);
    std::fill(hasPrev_.begin(), hasPrev_.end(), 0);
    std::fill(emaInit_.begin(), emaInit_.end(), 0);
    for (auto& rb : mags_) rb.clear();
  }

private:
  void processCell(size_t c, const std::complex<float>& D) {
    const float m = std::abs(D);
    mags_[c].push(m);
    mags_[c].copyTo(window_.data());
    // Hampel on magnitude: a returned value != m means m was an interference spike.
    const float fm = hampel(window_.data(), mags_[c].size(), m, scratch_.data(), hampelK_);
    const bool spike = (fm != m);
    const float p = spike ? lastPhase_[c] : std::arg(D);
    lastPhase_[c] = p;

    float u;
    if (!hasPrev_[c]) {
      u = p;
      hasPrev_[c] = 1;
    } else {
      u = unwrapStep(p, prevWrapped_[c], prevUnwrapped_[c]);
    }
    prevWrapped_[c] = p;
    prevUnwrapped_[c] = u;

    // Subtract an EMA to remove the static phase offset / slow drift, centering the motion signal. O(1).
    if (!emaInit_[c]) {
      ema_[c] = u;
      emaInit_[c] = 1;
    } else {
      ema_[c] = normAlpha_ * u + (1.0f - normAlpha_) * ema_[c];
    }
    output_[c] = u - ema_[c];
  }

  uint16_t inA_, inS_;
  uint16_t outRows_ = 0, outCols_ = 0;
  float hampelK_;
  float normAlpha_;

  std::vector<float> output_;
  std::vector<float> lastPhase_, prevWrapped_, prevUnwrapped_, ema_;
  std::vector<uint8_t> hasPrev_, emaInit_;
  std::vector<RingBuffer<float>> mags_;  // per-cell magnitude window for Hampel
  std::vector<float> window_;            // reused: current window contents copied out of mags_
  std::vector<float> scratch_;           // reused Hampel work buffer (size = window)
};

}  // namespace wavetrace
