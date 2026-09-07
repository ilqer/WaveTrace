#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "core/Errors.hpp"
#include "util/Fft.hpp"
#include "util/RingBuffer.hpp"

namespace wavetrace {

// Windowed feature front-end, two paths: a compact nine-feature TIME-domain vector per series (fewer
// features generalized better; kurtosis/entropy/slope deliberately dropped), and FREQUENCY-domain PSD + Doppler (f_d = 2v/lambda).

// --- §2.9 order statistics helper -----------------------------------------------------------

// Percentile of an ascending-sorted buffer via linear interpolation, matching numpy's default. percentile in [0,1]. O(1).
inline float PercentileSorted(const float* sortedValues, size_t sampleCount, float percentile) {
  if (sampleCount == 1) return sortedValues[0];
  const float index = percentile * static_cast<float>(sampleCount - 1);
  const size_t lowerIndex = static_cast<size_t>(index);
  const size_t upperIndex = (lowerIndex + 1 < sampleCount) ? lowerIndex + 1 : lowerIndex;
  const float interpolationFraction = index - static_cast<float>(lowerIndex);
  return sortedValues[lowerIndex] + interpolationFraction * (sortedValues[upperIndex] - sortedValues[lowerIndex]);
}

// Nine-feature vector over one chronological window, out[0..8] = mean, standard deviation, max, min,
// interquartile range (P75-P25), skewness, lag-1 autocorrelation, median absolute deviation (median|x-median|),
// waveform length (sum|x_i-x_i-1|). O(n log n) (sorts for interquartile range and median absolute deviation).
inline void NineFeatures(const float* values, size_t sampleCount, float* scratch, float* out) {
  if (sampleCount == 0) {
    for (size_t i = 0; i < 9; ++i) out[i] = 0.0f;
    return;
  }
  double mean = 0.0;
  for (size_t i = 0; i < sampleCount; ++i) mean += static_cast<double>(values[i]);
  mean /= static_cast<double>(sampleCount);

  double variance = 0.0, thirdCentralMoment = 0.0, autoCovariance = 0.0, waveformLength = 0.0;
  float maximum = values[0], minimum = values[0];
  for (size_t i = 0; i < sampleCount; ++i) {
    const double deviationFromMean = static_cast<double>(values[i]) - mean;
    variance += deviationFromMean * deviationFromMean;
    thirdCentralMoment += deviationFromMean * deviationFromMean * deviationFromMean;
    if (values[i] > maximum) maximum = values[i];
    if (values[i] < minimum) minimum = values[i];
    if (i >= 1) {
      waveformLength += std::fabs(static_cast<double>(values[i]) - static_cast<double>(values[i - 1]));
      autoCovariance += (static_cast<double>(values[i]) - mean) * (static_cast<double>(values[i - 1]) - mean);
    }
  }
  const double sumOfSquaredDeviations = variance;  // autocorrelation denominator
  variance /= static_cast<double>(sampleCount);
  thirdCentralMoment /= static_cast<double>(sampleCount);
  const double standardDeviation = std::sqrt(variance);
  const double skewness =
      (standardDeviation > 1e-12) ? (thirdCentralMoment / (standardDeviation * standardDeviation * standardDeviation)) : 0.0;
  const double lag1AutoCorrelation =
      (sumOfSquaredDeviations > 1e-12) ? (autoCovariance / sumOfSquaredDeviations) : 0.0;

  // Median from the sorted window, then median absolute deviation = median(|x - median|) reusing the scratch.
  for (size_t i = 0; i < sampleCount; ++i) scratch[i] = values[i];
  std::sort(scratch, scratch + sampleCount);
  const float median = PercentileSorted(scratch, sampleCount, 0.5f);
  const float interquartileRange =
      PercentileSorted(scratch, sampleCount, 0.75f) - PercentileSorted(scratch, sampleCount, 0.25f);
  for (size_t i = 0; i < sampleCount; ++i) scratch[i] = std::fabs(values[i] - median);
  std::sort(scratch, scratch + sampleCount);
  const float medianAbsoluteDeviation = PercentileSorted(scratch, sampleCount, 0.5f);

  out[0] = static_cast<float>(mean);
  out[1] = static_cast<float>(standardDeviation);
  out[2] = maximum;
  out[3] = minimum;
  out[4] = interquartileRange;
  out[5] = static_cast<float>(skewness);
  out[6] = static_cast<float>(lag1AutoCorrelation);
  out[7] = medianAbsoluteDeviation;
  out[8] = static_cast<float>(waveformLength);
}

// --- Per-packet inter-subcarrier dispersion (REFERENCE §0B — weapon discriminator) ----------

struct InterCarrierStat {
  float mean;      // mu[p]      = mean amplitude across subcarriers at one packet
  float variance;  // sigma2[p]  = sample variance (M-1) across subcarriers; metal -> LOWER
};

// Per-packet dispersion of K subcarrier magnitudes: a flat metal reflector reflects evenly so sigma2 is smaller
// than for a diffuse human body, the documented concealed-metal discriminator. Run over all valid subcarriers, not the NBVI subset. O(K).
inline InterCarrierStat ComputeInterCarrierStats(const float* magnitudes, size_t subcarrierCount) {
  if (subcarrierCount == 0) return {0.0f, 0.0f};
  double mean = 0.0;
  for (size_t i = 0; i < subcarrierCount; ++i) mean += static_cast<double>(magnitudes[i]);
  mean /= static_cast<double>(subcarrierCount);
  if (subcarrierCount == 1) return {static_cast<float>(mean), 0.0f};
  double variance = 0.0;
  for (size_t i = 0; i < subcarrierCount; ++i) {
    const double deviationFromMean = static_cast<double>(magnitudes[i]) - mean;
    variance += deviationFromMean * deviationFromMean;
  }
  variance /= static_cast<double>(subcarrierCount - 1);  // sample variance (M-1), matches Yousaf/LUMS
  return {static_cast<float>(mean), static_cast<float>(variance)};
}

// --- Per-frame inter-subcarrier PHASE dispersion (phase counterpart of sigma2[p]) ------------

struct InterCarrierPhaseStat {
  float slope;        // least-squares phase slope across subcarriers (rad/subcarrier) ~ group delay (ToF)
  float residualStd;  // RMS residual after removing ToF slope: lower for a coherent reflector, higher for a diffuse body
};

// Inter-subcarrier phase dispersion: unwrap phase across subcarriers, fit a line (slope = group-delay/ToF),
// return slope + RMS residual — the phase analogue of ComputeInterCarrierStats. O(k), no allocation beyond scratch.
inline InterCarrierPhaseStat ComputeInterCarrierPhaseStats(const float* phase, size_t subcarrierCount,
                                                            float* scratch) {
  if (subcarrierCount < 2) return {0.0f, 0.0f};
  constexpr float PI = 3.14159265358979323846f;
  constexpr float TWO_PI = 2.0f * PI;
  // Unwrap across subcarriers so the linear slope is not corrupted by 2*pi jumps.
  scratch[0] = phase[0];
  for (size_t i = 1; i < subcarrierCount; ++i) {
    float phaseStepRadians = phase[i] - phase[i - 1];
    while (phaseStepRadians > PI) phaseStepRadians -= TWO_PI;
    while (phaseStepRadians < -PI) phaseStepRadians += TWO_PI;
    scratch[i] = scratch[i - 1] + phaseStepRadians;
  }
  // Closed-form least-squares line fit of scratch[i] against i = 0..k-1.
  const double subcarrierCountAsDouble = static_cast<double>(subcarrierCount);
  double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
  for (size_t i = 0; i < subcarrierCount; ++i) {
    const double x = static_cast<double>(i), y = static_cast<double>(scratch[i]);
    sumX += x;
    sumY += y;
    sumXX += x * x;
    sumXY += x * y;
  }
  const double denominator = subcarrierCountAsDouble * sumXX - sumX * sumX;  // > 0 for k >= 2
  const double slope = (subcarrierCountAsDouble * sumXY - sumX * sumY) / denominator;
  const double intercept = (sumY - slope * sumX) / subcarrierCountAsDouble;
  double sumOfSquaredErrors = 0.0;
  for (size_t i = 0; i < subcarrierCount; ++i) {
    const double residual = static_cast<double>(scratch[i]) - (slope * static_cast<double>(i) + intercept);
    sumOfSquaredErrors += residual * residual;
  }
  return {static_cast<float>(slope), static_cast<float>(std::sqrt(sumOfSquaredErrors / subcarrierCountAsDouble))};
}

// --- Complex-CSI material reconstruction (in-baggage CNS'18 §IV / material-ID) ---------------

// Reconstruct sanitized complex CSI: unwrap phase, subtract the least-squares ToF/STO slope, recombine
// the residual with the original magnitude -> drift-free value whose clustering separates materials. O(k).
inline void ReconstructComplexCsi(const std::complex<float>* in, size_t subcarrierCount,
                                  std::complex<float>* out, float* scratch) {
  if (subcarrierCount == 0) return;
  if (subcarrierCount == 1) {
    out[0] = in[0];
    return;
  }
  constexpr float PI = 3.14159265358979323846f;
  constexpr float TWO_PI = 2.0f * PI;
  scratch[0] = std::arg(in[0]);
  for (size_t i = 1; i < subcarrierCount; ++i) {  // unwrap across subcarriers so the slope isn't broken by 2*pi
    float phaseStepRadians = std::arg(in[i]) - std::arg(in[i - 1]);
    while (phaseStepRadians > PI) phaseStepRadians -= TWO_PI;
    while (phaseStepRadians < -PI) phaseStepRadians += TWO_PI;
    scratch[i] = scratch[i - 1] + phaseStepRadians;
  }
  // Closed-form least-squares fit (same as ComputeInterCarrierPhaseStats).
  const double subcarrierCountAsDouble = static_cast<double>(subcarrierCount);
  double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
  for (size_t i = 0; i < subcarrierCount; ++i) {
    const double x = static_cast<double>(i), y = static_cast<double>(scratch[i]);
    sumX += x;
    sumY += y;
    sumXX += x * x;
    sumXY += x * y;
  }
  const double denominator = subcarrierCountAsDouble * sumXX - sumX * sumX;  // > 0 for k >= 2
  const double slope = (subcarrierCountAsDouble * sumXY - sumX * sumY) / denominator;
  const double intercept = (sumY - slope * sumX) / subcarrierCountAsDouble;
  for (size_t i = 0; i < subcarrierCount; ++i) {  // residual phase recombined with the original magnitude
    const float residualPhase = scratch[i] - static_cast<float>(slope * static_cast<double>(i) + intercept);
    out[i] = std::polar(std::abs(in[i]), residualPhase);
  }
}

// Beta-null reflection isolation: out=h1+beta*h2, beta=-hb1/hb2 from an empty-room baseline, chosen so the
// two paths cancel when empty; an object's reflection then survives as |out| (complex-level null, requires 2 paths). O(k).
inline void ComputeReflectionNull(const std::complex<float>* h1, const std::complex<float>* h2,
                           const std::complex<float>* hb1, const std::complex<float>* hb2, size_t subcarrierCount,
                           std::complex<float>* out) {
  for (size_t i = 0; i < subcarrierCount; ++i) {
    const std::complex<float> beta =
        (std::abs(hb2[i]) > 1e-12f) ? -hb1[i] / hb2[i] : std::complex<float>(0.0f, 0.0f);
    out[i] = h1[i] + beta * h2[i];
  }
}

// Non-overlapping block-average decimation: averages every `factor` samples into one (length n/factor),
// denoising and shrinking the CNN input; trailing remainder dropped. Returns the output count. O(n).
inline size_t BlockAverageDecimate(const float* values, size_t sampleCount, size_t factor, float* out) {
  if (factor == 0) return 0;
  const size_t blockCount = sampleCount / factor;
  for (size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
    double blockSum = 0.0;
    for (size_t i = 0; i < factor; ++i) blockSum += static_cast<double>(values[blockIndex * factor + i]);
    out[blockIndex] = static_cast<float>(blockSum / static_cast<double>(factor));
  }
  return blockCount;
}

// --- Frequency domain: PSD + Doppler (REFERENCE §2.6) ---------------------------------------

// PSD of a real series: detrend, Hann window (reduces leakage), zero-pad to fft.Size(), FFT, power = |z|^2
// over the first nfft/2+1 bins. Caller owns the Fft (power-of-two size >= n). O(nfft log nfft).
inline void ComputePowerSpectrum(const float* values, size_t sampleCount, const Fft& fft,
                          std::complex<float>* scratch, float* power) {
  const size_t nfft = fft.Size();
  double mean = 0.0;
  for (size_t i = 0; i < sampleCount; ++i) mean += static_cast<double>(values[i]);
  mean /= static_cast<double>(sampleCount);
  constexpr float TWO_PI = 6.28318530717958647692f;
  const float hannAngularStep = (sampleCount > 1) ? TWO_PI / static_cast<float>(sampleCount - 1) : 0.0f;
  for (size_t i = 0; i < sampleCount; ++i) {
    const float hannWeight = 0.5f * (1.0f - std::cos(hannAngularStep * static_cast<float>(i)));  // Hann
    scratch[i] = std::complex<float>((static_cast<float>(values[i]) - static_cast<float>(mean)) * hannWeight, 0.0f);
  }
  for (size_t i = sampleCount; i < nfft; ++i) scratch[i] = std::complex<float>(0.0f, 0.0f);  // zero-pad
  fft.Forward(scratch);
  const size_t bins = nfft / 2 + 1;
  for (size_t k = 0; k < bins; ++k) power[k] = std::norm(scratch[k]);  // |z|^2 = re^2 + im^2
}

struct DopplerFeature {
  float maxShiftHz;  // dominant Doppler frequency in the band (f_d = 2v/lambda)
  float spreadHz;    // power-weighted spectral spread (how broad the motion energy is)
};

// Doppler features from the power spectrum: peak frequency in (0, highCutoffHz] is the max shift, power-weighted RMS
// deviation about the centroid is the spread; DC (bin 0) excluded so a residual offset can't masquerade as the peak. O(nfft log nfft).
inline DopplerFeature ComputeDopplerFeatures(const float* values, size_t sampleCount, float sampleRateHz,
                                      float highCutoffHz, const Fft& fft,
                                      std::complex<float>* scratch, float* power) {
  ComputePowerSpectrum(values, sampleCount, fft, scratch, power);
  const size_t nfft = fft.Size();
  const size_t bins = nfft / 2 + 1;
  const float freqRes = sampleRateHz / static_cast<float>(nfft);
  size_t kHi = (highCutoffHz > 0.0f) ? static_cast<size_t>(highCutoffHz / freqRes) : (bins - 1);
  if (kHi >= bins) kHi = bins - 1;
  if (kHi < 1) kHi = (bins > 1) ? 1 : 0;

  float peakPow = -1.0f;
  size_t peakK = (bins > 1) ? 1 : 0;
  double sumP = 0.0, sumPf = 0.0;
  for (size_t k = 1; k <= kHi; ++k) {
    const float binPower = power[k];
    if (binPower > peakPow) {
      peakPow = binPower;
      peakK = k;
    }
    const double frequencyHz = static_cast<double>(k) * freqRes;
    sumP += binPower;
    sumPf += static_cast<double>(binPower) * frequencyHz;
  }
  const double centroid = (sumP > 0.0) ? (sumPf / sumP) : 0.0;
  double sumPdf2 = 0.0;
  for (size_t k = 1; k <= kHi; ++k) {
    const double frequencyHz = static_cast<double>(k) * freqRes;
    sumPdf2 += static_cast<double>(power[k]) * (frequencyHz - centroid) * (frequencyHz - centroid);
  }
  const double spread = (sumP > 0.0) ? std::sqrt(sumPdf2 / sumP) : 0.0;
  return {static_cast<float>(peakK) * freqRes, static_cast<float>(spread)};
}

// --- Streaming §2.9 feature extractor -------------------------------------------------------

// Streaming nine-feature extractor over C parallel scalar series: buffers the last `window` samples per series
// and emits a 9*C feature block every `hop` frames. Push O(C); emit O(C*window log window), buffers sized in ctor.
class FeatureExtractor {
public:
  static constexpr size_t FEATURES_PER_SERIES = 9;

  FeatureExtractor(size_t seriesCount, size_t window, size_t hop)
      : seriesCount_(seriesCount), window_(window), hop_(hop) {
    if (seriesCount == 0 || window == 0 || hop == 0) {
      throw WaveTraceError("FeatureExtractor: seriesCount, window, hop must be non-zero");
    }
    rings_.reserve(seriesCount_);
    for (size_t i = 0; i < seriesCount_; ++i) rings_.emplace_back(window_);
    windowBuffer_.assign(window_, 0.0f);
    scratch_.assign(window_, 0.0f);
    output_.assign(seriesCount_ * FEATURES_PER_SERIES, 0.0f);
  }

  size_t SeriesCount() const { return seriesCount_; }
  size_t Window() const { return window_; }
  size_t Hop() const { return hop_; }
  size_t OutputSize() const { return output_.size(); }
  const float* Data() const { return output_.data(); }

  // Returns true when a feature vector was emitted (window full and `hop` frames since last), then available via Data(). Non-emit frames are O(C).
  bool Push(const float* values) {
    for (size_t i = 0; i < seriesCount_; ++i) rings_[i].Push(values[i]);
    ++sinceEmit_;
    if (rings_[0].Size() < window_ || sinceEmit_ < hop_) return false;
    sinceEmit_ = 0;
    for (size_t i = 0; i < seriesCount_; ++i) {
      rings_[i].CopyOrdered(windowBuffer_.data());  // chronological window (lag-1/WL need order)
      NineFeatures(windowBuffer_.data(), window_, scratch_.data(), &output_[i * FEATURES_PER_SERIES]);
    }
    return true;
  }

  void Reset() {
    for (auto& r : rings_) r.Clear();
    sinceEmit_ = 0;
    std::fill(output_.begin(), output_.end(), 0.0f);
  }

private:
  size_t seriesCount_, window_, hop_;
  size_t sinceEmit_ = 0;
  std::vector<RingBuffer<float>> rings_;
  std::vector<float> windowBuffer_, scratch_, output_;
};

// --- Streaming inter-subcarrier amplitude-dispersion extractor (windows sigma2[p]) -----------

// Turns per-packet inter-subcarrier dispersion into a classifier-ready block: reduces K magnitudes to {mu[p], sigma2[p], cv[p]=std/mu}, emitting NineFeatures of each series every `hop` frames (output 27 = 3*9).
// INPUT CONTRACT: push RAW magnitudes, not gain-locked — a mean lock cancels the flatness that IS the metal signature. O(K) push.
class InterCarrierExtractor {
public:
  static constexpr size_t SERIES_COUNT = 3;  // 0 mu, 1 sigma2, 2 cv
  static constexpr size_t FEATURES_PER_SERIES = 9;

  InterCarrierExtractor(size_t window, size_t hop) : window_(window), hop_(hop) {
    if (window == 0 || hop == 0) {
      throw WaveTraceError("InterCarrierExtractor: window, hop must be non-zero");
    }
    rings_.reserve(SERIES_COUNT);
    for (size_t i = 0; i < SERIES_COUNT; ++i) rings_.emplace_back(window_);
    windowBuffer_.assign(window_, 0.0f);
    scratch_.assign(window_, 0.0f);
    output_.assign(SERIES_COUNT * FEATURES_PER_SERIES, 0.0f);
  }

  size_t Window() const { return window_; }
  size_t Hop() const { return hop_; }
  size_t OutputSize() const { return output_.size(); }
  const float* Data() const { return output_.data(); }

  // Push one frame's K subcarrier magnitudes; True when a feature block was emitted (see Data()).
  bool Push(const float* magnitudes, size_t subcarrierCount) {
    const InterCarrierStat stats = ComputeInterCarrierStats(magnitudes, subcarrierCount);
    const float coefficientOfVariation = (stats.mean > 1e-12f) ? std::sqrt(stats.variance) / stats.mean : 0.0f;
    rings_[0].Push(stats.mean);
    rings_[1].Push(stats.variance);
    rings_[2].Push(coefficientOfVariation);
    ++sinceEmit_;
    if (rings_[0].Size() < window_ || sinceEmit_ < hop_) return false;
    sinceEmit_ = 0;
    for (size_t i = 0; i < SERIES_COUNT; ++i) {
      rings_[i].CopyOrdered(windowBuffer_.data());  // chronological (lag-1/WL need order)
      NineFeatures(windowBuffer_.data(), window_, scratch_.data(), &output_[i * FEATURES_PER_SERIES]);
    }
    return true;
  }

  void Reset() {
    for (auto& r : rings_) r.Clear();
    sinceEmit_ = 0;
    std::fill(output_.begin(), output_.end(), 0.0f);
  }

private:
  size_t window_, hop_;
  size_t sinceEmit_ = 0;
  std::vector<RingBuffer<float>> rings_;
  std::vector<float> windowBuffer_, scratch_, output_;
};

}  // namespace wavetrace
