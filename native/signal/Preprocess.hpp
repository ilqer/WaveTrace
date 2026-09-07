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
inline void ConjugateMultiply(const CsiFrame& in, CsiFrame& out) {
  const uint16_t antennaCount = in.NumAntennas();
  const uint16_t subcarrierCount = in.NumSubcarriers();
  const CsiFrame::Sample* H = in.Data();
  if (antennaCount >= 2) {
    out.Reshape(static_cast<uint16_t>(antennaCount - 1), subcarrierCount);
    CsiFrame::Sample* D = out.Data();
    const CsiFrame::Sample* ref = H;  // antenna 0 = shared-clock reference
    for (uint16_t a = 1; a < antennaCount; ++a) {
      const CsiFrame::Sample* row = H + static_cast<size_t>(a) * subcarrierCount;
      CsiFrame::Sample* outRow = D + static_cast<size_t>(a - 1) * subcarrierCount;
      for (uint16_t k = 0; k < subcarrierCount; ++k) outRow[k] = row[k] * std::conj(ref[k]);
    }
  } else {
    if (subcarrierCount < 2) throw FrameError("ConjugateMultiply: single antenna needs >= 2 subcarriers");
    out.Reshape(1, static_cast<uint16_t>(subcarrierCount - 1));
    CsiFrame::Sample* D = out.Data();
    for (uint16_t k = 1; k < subcarrierCount; ++k) D[k - 1] = H[k] * std::conj(H[k - 1]);
  }
}

// Antenna-difference out[a-1][k]=H[a][k]-H[0][k]: on one shared-clock radio this nulls the common environment
// (LOS + static furniture) and amplifies per-antenna scattering, unlike ConjugateMultiply which cancels clock drift. O(n).
inline void CombinedChannelDifference(const CsiFrame& in, CsiFrame& out) {
  const uint16_t antennaCount = in.NumAntennas();
  const uint16_t subcarrierCount = in.NumSubcarriers();
  if (antennaCount < 2) throw FrameError("CombinedChannelDifference: requires >= 2 antennas on one radio");
  out.Reshape(static_cast<uint16_t>(antennaCount - 1), subcarrierCount);
  const CsiFrame::Sample* H = in.Data();
  CsiFrame::Sample* D = out.Data();
  const CsiFrame::Sample* ref = H;  // antenna 0 = common reference
  for (uint16_t a = 1; a < antennaCount; ++a) {
    const CsiFrame::Sample* row = H + static_cast<size_t>(a) * subcarrierCount;
    CsiFrame::Sample* outRow = D + static_cast<size_t>(a - 1) * subcarrierCount;
    for (uint16_t k = 0; k < subcarrierCount; ++k) outRow[k] = row[k] - ref[k];
  }
}

// Hampel outlier test: returns `current` unless it deviates beyond thresholdK*1.4826*MAD from the window median
// (1.4826 makes MAD a consistent sigma estimator for Gaussians), else returns the median. O(w) (nth_element).
inline float Hampel(const float* window, size_t windowSize, float current, float* scratch, float thresholdK) {
  if (windowSize == 0) return current;
  const size_t mid = windowSize / 2;
  for (size_t i = 0; i < windowSize; ++i) scratch[i] = window[i];
  std::nth_element(scratch, scratch + mid, scratch + windowSize);
  const float med = scratch[mid];
  // nth_element only permuted scratch, so its median-of-deviations here is still the MAD.
  for (size_t i = 0; i < windowSize; ++i) scratch[i] = std::fabs(scratch[i] - med);
  std::nth_element(scratch, scratch + mid, scratch + windowSize);
  const float mad = scratch[mid];
  if (mad > 0.0f && std::fabs(current - med) > thresholdK * 1.4826f * mad) return med;
  return current;
}

// One streaming phase-unwrap step: bring the step from the previous wrapped phase into (-pi, pi] and add it to the running unwrapped value. O(1).
inline float UnwrapStep(float curWrapped, float prevWrapped, float prevUnwrapped) {
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
      : inAntennas_(numAntennas),
        inSubcarriers_(numSubcarriers),
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
    bHasPrevious_.assign(cells, 0);
    bEmaInitialized_.assign(cells, 0);
    mags_.reserve(cells);
    for (size_t c = 0; c < cells; ++c) mags_.emplace_back(hampelWindow);
    window_.assign(hampelWindow, 0.0f);
    scratch_.assign(hampelWindow, 0.0f);
  }

  uint16_t OutRows() const { return outRows_; }
  uint16_t OutCols() const { return outCols_; }
  const float* Data() const { return output_.data(); }

  // Process one frame; result is in the reused output_ grid (returned via Data()). O(n)/frame.
  void Process(const CsiFrame& in) {
    if (in.NumAntennas() != inAntennas_ || in.NumSubcarriers() != inSubcarriers_) {
      throw FrameError("Preprocessor: frame geometry does not match configuration");
    }
    const CsiFrame::Sample* H = in.Data();
    size_t c = 0;
    if (inAntennas_ >= 2) {
      const CsiFrame::Sample* ref = H;
      for (uint16_t a = 1; a < inAntennas_; ++a) {
        const CsiFrame::Sample* row = H + static_cast<size_t>(a) * inSubcarriers_;
        for (uint16_t k = 0; k < inSubcarriers_; ++k, ++c) ProcessCell(c, row[k] * std::conj(ref[k]));
      }
    } else {
      for (uint16_t k = 1; k < inSubcarriers_; ++k, ++c) ProcessCell(c, H[k] * std::conj(H[k - 1]));
    }
  }

  void Reset() {
    std::fill(output_.begin(), output_.end(), 0.0f);
    std::fill(ema_.begin(), ema_.end(), 0.0f);
    std::fill(bHasPrevious_.begin(), bHasPrevious_.end(), 0);
    std::fill(bEmaInitialized_.begin(), bEmaInitialized_.end(), 0);
    for (auto& rb : mags_) rb.Clear();
  }

private:
  void ProcessCell(size_t c, const std::complex<float>& D) {
    const float m = std::abs(D);
    mags_[c].Push(m);
    mags_[c].CopyTo(window_.data());
    // Hampel on magnitude: a returned value != m means m was an interference spike.
    const float fm = Hampel(window_.data(), mags_[c].Size(), m, scratch_.data(), hampelK_);
    const bool bSpike = (fm != m);
    const float p = bSpike ? lastPhase_[c] : std::arg(D);
    lastPhase_[c] = p;

    float u;
    if (!bHasPrevious_[c]) {
      u = p;
      bHasPrevious_[c] = 1;
    } else {
      u = UnwrapStep(p, prevWrapped_[c], prevUnwrapped_[c]);
    }
    prevWrapped_[c] = p;
    prevUnwrapped_[c] = u;

    // Subtract an EMA to remove the static phase offset / slow drift, centering the motion signal. O(1).
    if (!bEmaInitialized_[c]) {
      ema_[c] = u;
      bEmaInitialized_[c] = 1;
    } else {
      ema_[c] = normAlpha_ * u + (1.0f - normAlpha_) * ema_[c];
    }
    output_[c] = u - ema_[c];
  }

  uint16_t inAntennas_, inSubcarriers_;
  uint16_t outRows_ = 0, outCols_ = 0;
  float hampelK_;
  float normAlpha_;

  std::vector<float> output_;
  std::vector<float> lastPhase_, prevWrapped_, prevUnwrapped_, ema_;
  std::vector<uint8_t> bHasPrevious_, bEmaInitialized_;
  std::vector<RingBuffer<float>> mags_;  // per-cell magnitude window for Hampel
  std::vector<float> window_;            // reused: current window contents copied out of mags_
  std::vector<float> scratch_;           // reused Hampel work buffer (size = window)
};

}  // namespace wavetrace
