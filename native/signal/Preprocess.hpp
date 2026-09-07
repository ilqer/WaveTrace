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
  const uint16_t antennaCount = in.AntennaCount();
  const uint16_t subcarrierCount = in.SubcarrierCount();
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
  const uint16_t antennaCount = in.AntennaCount();
  const uint16_t subcarrierCount = in.SubcarrierCount();
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
  const size_t medianIndex = windowSize / 2;
  for (size_t i = 0; i < windowSize; ++i) scratch[i] = window[i];
  std::nth_element(scratch, scratch + medianIndex, scratch + windowSize);
  const float median = scratch[medianIndex];
  // nth_element only permuted scratch, so its median-of-deviations here is still the MAD.
  for (size_t i = 0; i < windowSize; ++i) scratch[i] = std::fabs(scratch[i] - median);
  std::nth_element(scratch, scratch + medianIndex, scratch + windowSize);
  const float medianAbsoluteDeviation = scratch[medianIndex];
  if (medianAbsoluteDeviation > 0.0f && std::fabs(current - median) > thresholdK * 1.4826f * medianAbsoluteDeviation)
    return median;
  return current;
}

// One streaming phase-unwrap step: bring the step from the previous wrapped phase into (-pi, pi] and add it to the running unwrapped value. O(1).
inline float UnwrapStep(float currentWrappedPhaseRadians, float previousWrappedPhaseRadians,
                        float previousUnwrappedPhaseRadians) {
  float phaseStepRadians = currentWrappedPhaseRadians - previousWrappedPhaseRadians;
  while (phaseStepRadians > WT_PI) phaseStepRadians -= WT_TWO_PI;
  while (phaseStepRadians < -WT_PI) phaseStepRadians += WT_TWO_PI;
  return previousUnwrappedPhaseRadians + phaseStepRadians;
}

// --- Streaming preprocessor (the hot-path chain) --------------------------------------------

// Per-frame DSP front-end: conjugate-multiply -> Hampel -> unwrap -> normalize into a drift-free, spike-cleaned,
// detrended differential-phase grid; stateful, alloc-free after ctor. Hampel gates on magnitude; on a spike the phase holds at the last good value.
class Preprocessor {
public:
  Preprocessor(uint16_t antennaCount, uint16_t subcarrierCount, size_t hampelWindow = 7,
               float hampelK = 5.0f, float normalizeAlpha = 0.1f)
      : inAntennas_(antennaCount),
        inSubcarriers_(subcarrierCount),
        hampelK_(hampelK),
        normAlpha_(normalizeAlpha) {
    if (antennaCount == 0 || subcarrierCount == 0) {
      throw FrameError("Preprocessor: dimensions must be non-zero");
    }
    if (antennaCount >= 2) {
      outRows_ = static_cast<uint16_t>(antennaCount - 1);
      outCols_ = subcarrierCount;
    } else {
      if (subcarrierCount < 2) throw FrameError("Preprocessor: single antenna needs >= 2 subcarriers");
      outRows_ = 1;
      outCols_ = static_cast<uint16_t>(subcarrierCount - 1);
    }
    const size_t cells = static_cast<size_t>(outRows_) * outCols_;
    output_.assign(cells, 0.0f);
    lastPhase_.assign(cells, 0.0f);
    previousWrappedPhaseRadians_.assign(cells, 0.0f);
    previousUnwrappedPhaseRadians_.assign(cells, 0.0f);
    exponentialMovingAverage_.assign(cells, 0.0f);
    bHasPrevious_.assign(cells, 0);
    bExponentialMovingAverageInitialized_.assign(cells, 0);
    magnitudeWindows_.reserve(cells);
    for (size_t cellIndex = 0; cellIndex < cells; ++cellIndex) magnitudeWindows_.emplace_back(hampelWindow);
    window_.assign(hampelWindow, 0.0f);
    scratch_.assign(hampelWindow, 0.0f);
  }

  uint16_t OutRows() const { return outRows_; }
  uint16_t OutCols() const { return outCols_; }
  const float* Data() const { return output_.data(); }

  // Process one frame; result is in the reused output_ grid (returned via Data()). O(n)/frame.
  void Process(const CsiFrame& in) {
    if (in.AntennaCount() != inAntennas_ || in.SubcarrierCount() != inSubcarriers_) {
      throw FrameError("Preprocessor: frame geometry does not match configuration");
    }
    const CsiFrame::Sample* H = in.Data();
    size_t cellIndex = 0;
    if (inAntennas_ >= 2) {
      const CsiFrame::Sample* ref = H;
      for (uint16_t a = 1; a < inAntennas_; ++a) {
        const CsiFrame::Sample* row = H + static_cast<size_t>(a) * inSubcarriers_;
        for (uint16_t k = 0; k < inSubcarriers_; ++k, ++cellIndex) ProcessCell(cellIndex, row[k] * std::conj(ref[k]));
      }
    } else {
      for (uint16_t k = 1; k < inSubcarriers_; ++k, ++cellIndex) ProcessCell(cellIndex, H[k] * std::conj(H[k - 1]));
    }
  }

  void Reset() {
    std::fill(output_.begin(), output_.end(), 0.0f);
    std::fill(exponentialMovingAverage_.begin(), exponentialMovingAverage_.end(), 0.0f);
    std::fill(bHasPrevious_.begin(), bHasPrevious_.end(), 0);
    std::fill(bExponentialMovingAverageInitialized_.begin(), bExponentialMovingAverageInitialized_.end(), 0);
    for (auto& rb : magnitudeWindows_) rb.Clear();
  }

private:
  void ProcessCell(size_t cellIndex, const std::complex<float>& differentialSample) {
    const float magnitude = std::abs(differentialSample);
    magnitudeWindows_[cellIndex].Push(magnitude);
    magnitudeWindows_[cellIndex].CopyTo(window_.data());
    // Hampel on magnitude: a returned value != magnitude means magnitude was an interference spike.
    const float filteredMagnitude =
        Hampel(window_.data(), magnitudeWindows_[cellIndex].Size(), magnitude, scratch_.data(), hampelK_);
    const bool bSpike = (filteredMagnitude != magnitude);
    const float phase = bSpike ? lastPhase_[cellIndex] : std::arg(differentialSample);
    lastPhase_[cellIndex] = phase;

    float unwrappedPhase;
    if (!bHasPrevious_[cellIndex]) {
      unwrappedPhase = phase;
      bHasPrevious_[cellIndex] = 1;
    } else {
      unwrappedPhase = UnwrapStep(phase, previousWrappedPhaseRadians_[cellIndex], previousUnwrappedPhaseRadians_[cellIndex]);
    }
    previousWrappedPhaseRadians_[cellIndex] = phase;
    previousUnwrappedPhaseRadians_[cellIndex] = unwrappedPhase;

    // Subtract an EMA to remove the static phase offset / slow drift, centering the motion signal. O(1).
    if (!bExponentialMovingAverageInitialized_[cellIndex]) {
      exponentialMovingAverage_[cellIndex] = unwrappedPhase;
      bExponentialMovingAverageInitialized_[cellIndex] = 1;
    } else {
      exponentialMovingAverage_[cellIndex] =
          normAlpha_ * unwrappedPhase + (1.0f - normAlpha_) * exponentialMovingAverage_[cellIndex];
    }
    output_[cellIndex] = unwrappedPhase - exponentialMovingAverage_[cellIndex];
  }

  uint16_t inAntennas_, inSubcarriers_;
  uint16_t outRows_ = 0, outCols_ = 0;
  float hampelK_;
  float normAlpha_;

  std::vector<float> output_;
  std::vector<float> lastPhase_, previousWrappedPhaseRadians_, previousUnwrappedPhaseRadians_, exponentialMovingAverage_;
  std::vector<uint8_t> bHasPrevious_, bExponentialMovingAverageInitialized_;
  std::vector<RingBuffer<float>> magnitudeWindows_;  // per-cell magnitude window for Hampel
  std::vector<float> window_;                        // reused: current window contents copied out of magnitudeWindows_
  std::vector<float> scratch_;                       // reused Hampel work buffer (size = window)
};

}  // namespace wavetrace
