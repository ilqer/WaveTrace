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

// Percentile of an ascending-sorted buffer via linear interpolation, matching numpy's default. p in [0,1]. O(1).
inline float percentileSorted(const float* sorted, size_t n, float p) {
  if (n == 1) return sorted[0];
  const float idx = p * static_cast<float>(n - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = (lo + 1 < n) ? lo + 1 : lo;
  const float frac = idx - static_cast<float>(lo);
  return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

// Nine-feature vector over one chronological window, out[0..8] = mean, std, max, min, IQR(P75-P25), skewness,
// lag-1 autocorrelation, MAD(median|x-median|), waveform-length(sum|x_i-x_i-1|). O(n log n) (IQR/MAD sorts).
inline void nineFeatures(const float* x, size_t n, float* scratch, float* out) {
  if (n == 0) {
    for (size_t i = 0; i < 9; ++i) out[i] = 0.0f;
    return;
  }
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i) mean += static_cast<double>(x[i]);
  mean /= static_cast<double>(n);

  double var = 0.0, m3 = 0.0, autocov = 0.0, wl = 0.0;
  float mx = x[0], mn = x[0];
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(x[i]) - mean;
    var += d * d;
    m3 += d * d * d;
    if (x[i] > mx) mx = x[i];
    if (x[i] < mn) mn = x[i];
    if (i >= 1) {
      wl += std::fabs(static_cast<double>(x[i]) - static_cast<double>(x[i - 1]));
      autocov += (static_cast<double>(x[i]) - mean) * (static_cast<double>(x[i - 1]) - mean);
    }
  }
  const double sumSq = var;  // total sum of squared deviations (autocorr denominator)
  var /= static_cast<double>(n);
  m3 /= static_cast<double>(n);
  const double sd = std::sqrt(var);
  const double skew = (sd > 1e-12) ? (m3 / (sd * sd * sd)) : 0.0;
  const double lag1 = (sumSq > 1e-12) ? (autocov / sumSq) : 0.0;  // lag-1 autocorrelation coeff

  // IQR + median from the sorted window, then MAD = median(|x - median|) reusing the scratch.
  for (size_t i = 0; i < n; ++i) scratch[i] = x[i];
  std::sort(scratch, scratch + n);
  const float med = percentileSorted(scratch, n, 0.5f);
  const float iqr = percentileSorted(scratch, n, 0.75f) - percentileSorted(scratch, n, 0.25f);
  for (size_t i = 0; i < n; ++i) scratch[i] = std::fabs(x[i] - med);
  std::sort(scratch, scratch + n);
  const float mad = percentileSorted(scratch, n, 0.5f);

  out[0] = static_cast<float>(mean);
  out[1] = static_cast<float>(sd);
  out[2] = mx;
  out[3] = mn;
  out[4] = iqr;
  out[5] = static_cast<float>(skew);
  out[6] = static_cast<float>(lag1);
  out[7] = mad;
  out[8] = static_cast<float>(wl);
}

// --- Per-packet inter-subcarrier dispersion (REFERENCE §0B — weapon discriminator) ----------

struct InterCarrierStat {
  float mean;      // mu[p]      = mean amplitude across subcarriers at one packet
  float variance;  // sigma2[p]  = sample variance (M-1) across subcarriers; metal -> LOWER
};

// Per-packet dispersion of K subcarrier magnitudes: a flat metal reflector reflects evenly so sigma2 is smaller
// than for a diffuse human body, the documented concealed-metal discriminator. Run over all valid subcarriers, not the NBVI subset. O(K).
inline InterCarrierStat interCarrierStats(const float* mags, size_t k) {
  if (k == 0) return {0.0f, 0.0f};
  double mean = 0.0;
  for (size_t i = 0; i < k; ++i) mean += static_cast<double>(mags[i]);
  mean /= static_cast<double>(k);
  if (k == 1) return {static_cast<float>(mean), 0.0f};
  double var = 0.0;
  for (size_t i = 0; i < k; ++i) {
    const double d = static_cast<double>(mags[i]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(k - 1);  // sample variance (M-1), matches Yousaf/LUMS
  return {static_cast<float>(mean), static_cast<float>(var)};
}

// --- Per-frame inter-subcarrier PHASE dispersion (phase counterpart of sigma2[p]) ------------

struct InterCarrierPhaseStat {
  float slope;        // least-squares phase slope across subcarriers (rad/subcarrier) ~ group delay (ToF)
  float residualStd;  // RMS residual after removing ToF slope: lower for a coherent reflector, higher for a diffuse body
};

// Inter-subcarrier phase dispersion: unwrap phase across subcarriers, fit a line (slope = group-delay/ToF),
// return slope + RMS residual — the phase analogue of interCarrierStats. O(k), no allocation beyond scratch.
inline InterCarrierPhaseStat interCarrierPhaseStats(const float* phase, size_t k, float* scratch) {
  if (k < 2) return {0.0f, 0.0f};
  constexpr float PI = 3.14159265358979323846f;
  constexpr float TWO_PI = 2.0f * PI;
  // Unwrap across subcarriers so the linear slope is not corrupted by 2*pi jumps.
  scratch[0] = phase[0];
  for (size_t i = 1; i < k; ++i) {
    float d = phase[i] - phase[i - 1];
    while (d > PI) d -= TWO_PI;
    while (d < -PI) d += TWO_PI;
    scratch[i] = scratch[i - 1] + d;
  }
  // Least-squares line y = a*x + b over x = 0..k-1 (closed form).
  const double n = static_cast<double>(k);
  double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
  for (size_t i = 0; i < k; ++i) {
    const double x = static_cast<double>(i), y = static_cast<double>(scratch[i]);
    sumX += x;
    sumY += y;
    sumXX += x * x;
    sumXY += x * y;
  }
  const double denom = n * sumXX - sumX * sumX;  // > 0 for k >= 2
  const double a = (n * sumXY - sumX * sumY) / denom;
  const double b = (sumY - a * sumX) / n;
  double sse = 0.0;
  for (size_t i = 0; i < k; ++i) {
    const double r = static_cast<double>(scratch[i]) - (a * static_cast<double>(i) + b);
    sse += r * r;
  }
  return {static_cast<float>(a), static_cast<float>(std::sqrt(sse / n))};
}

// --- Complex-CSI material reconstruction (in-baggage CNS'18 §IV / material-ID) ---------------

// Reconstruct sanitized complex CSI: unwrap phase, subtract the least-squares ToF/STO slope, recombine
// the residual with the original magnitude -> drift-free value whose clustering separates materials. O(k).
inline void reconstructComplexCsi(const std::complex<float>* in, size_t k, std::complex<float>* out,
                                  float* scratch) {
  if (k == 0) return;
  if (k == 1) {
    out[0] = in[0];
    return;
  }
  constexpr float PI = 3.14159265358979323846f;
  constexpr float TWO_PI = 2.0f * PI;
  scratch[0] = std::arg(in[0]);
  for (size_t i = 1; i < k; ++i) {  // unwrap across subcarriers so the slope isn't broken by 2*pi
    float d = std::arg(in[i]) - std::arg(in[i - 1]);
    while (d > PI) d -= TWO_PI;
    while (d < -PI) d += TWO_PI;
    scratch[i] = scratch[i - 1] + d;
  }
  // Least-squares line y = a*x + b over x = 0..k-1 (closed form, same fit as interCarrierPhaseStats).
  const double n = static_cast<double>(k);
  double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
  for (size_t i = 0; i < k; ++i) {
    const double x = static_cast<double>(i), y = static_cast<double>(scratch[i]);
    sumX += x;
    sumY += y;
    sumXX += x * x;
    sumXY += x * y;
  }
  const double denom = n * sumXX - sumX * sumX;  // > 0 for k >= 2
  const double a = (n * sumXY - sumX * sumY) / denom;
  const double b = (sumY - a * sumX) / n;
  for (size_t i = 0; i < k; ++i) {  // residual phase recombined with the original magnitude
    const float resid = scratch[i] - static_cast<float>(a * static_cast<double>(i) + b);
    out[i] = std::polar(std::abs(in[i]), resid);
  }
}

// Beta-null reflection isolation: out=h1+beta*h2, beta=-hb1/hb2 from an empty-room baseline, chosen so the
// two paths cancel when empty; an object's reflection then survives as |out| (complex-level null, requires 2 paths). O(k).
inline void reflectionNull(const std::complex<float>* h1, const std::complex<float>* h2,
                           const std::complex<float>* hb1, const std::complex<float>* hb2, size_t k,
                           std::complex<float>* out) {
  for (size_t i = 0; i < k; ++i) {
    const std::complex<float> beta =
        (std::abs(hb2[i]) > 1e-12f) ? -hb1[i] / hb2[i] : std::complex<float>(0.0f, 0.0f);
    out[i] = h1[i] + beta * h2[i];
  }
}

// Non-overlapping block-average decimation: averages every `factor` samples into one (length n/factor),
// denoising and shrinking the CNN input; trailing remainder dropped. Returns the output count. O(n).
inline size_t blockAverageDecimate(const float* x, size_t n, size_t factor, float* out) {
  if (factor == 0) return 0;
  const size_t m = n / factor;
  for (size_t b = 0; b < m; ++b) {
    double s = 0.0;
    for (size_t i = 0; i < factor; ++i) s += static_cast<double>(x[b * factor + i]);
    out[b] = static_cast<float>(s / static_cast<double>(factor));
  }
  return m;
}

// --- Frequency domain: PSD + Doppler (REFERENCE §2.6) ---------------------------------------

// PSD of a real series: detrend, Hann window (reduces leakage), zero-pad to fft.size(), FFT, power = |z|^2
// over the first nfft/2+1 bins. Caller owns the Fft (power-of-two size >= n). O(nfft log nfft).
inline void powerSpectrum(const float* x, size_t n, const Fft& fft, std::complex<float>* scratch,
                          float* power) {
  const size_t nfft = fft.size();
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i) mean += static_cast<double>(x[i]);
  mean /= static_cast<double>(n);
  constexpr float TWO_PI = 6.28318530717958647692f;
  const float c = (n > 1) ? TWO_PI / static_cast<float>(n - 1) : 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float w = 0.5f * (1.0f - std::cos(c * static_cast<float>(i)));  // Hann
    scratch[i] = std::complex<float>((static_cast<float>(x[i]) - static_cast<float>(mean)) * w, 0.0f);
  }
  for (size_t i = n; i < nfft; ++i) scratch[i] = std::complex<float>(0.0f, 0.0f);  // zero-pad
  fft.forward(scratch);
  const size_t bins = nfft / 2 + 1;
  for (size_t k = 0; k < bins; ++k) power[k] = std::norm(scratch[k]);  // |z|^2 = re^2 + im^2
}

struct DopplerFeature {
  float maxShiftHz;  // dominant Doppler frequency in the band (f_d = 2v/lambda)
  float spreadHz;    // power-weighted spectral spread (how broad the motion energy is)
};

// Doppler features from the power spectrum: peak frequency in (0, fHi] is the max shift, power-weighted RMS
// deviation about the centroid is the spread; DC (bin 0) excluded so a residual offset can't masquerade as the peak. O(nfft log nfft).
inline DopplerFeature dopplerFeatures(const float* x, size_t n, float fs, float fHi, const Fft& fft,
                                      std::complex<float>* scratch, float* power) {
  powerSpectrum(x, n, fft, scratch, power);
  const size_t nfft = fft.size();
  const size_t bins = nfft / 2 + 1;
  const float freqRes = fs / static_cast<float>(nfft);
  size_t kHi = (fHi > 0.0f) ? static_cast<size_t>(fHi / freqRes) : (bins - 1);
  if (kHi >= bins) kHi = bins - 1;
  if (kHi < 1) kHi = (bins > 1) ? 1 : 0;

  float peakPow = -1.0f;
  size_t peakK = (bins > 1) ? 1 : 0;
  double sumP = 0.0, sumPf = 0.0;
  for (size_t k = 1; k <= kHi; ++k) {
    const float p = power[k];
    if (p > peakPow) {
      peakPow = p;
      peakK = k;
    }
    const double f = static_cast<double>(k) * freqRes;
    sumP += p;
    sumPf += static_cast<double>(p) * f;
  }
  const double centroid = (sumP > 0.0) ? (sumPf / sumP) : 0.0;
  double sumPdf2 = 0.0;
  for (size_t k = 1; k <= kHi; ++k) {
    const double f = static_cast<double>(k) * freqRes;
    sumPdf2 += static_cast<double>(power[k]) * (f - centroid) * (f - centroid);
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

  FeatureExtractor(size_t numSeries, size_t window, size_t hop)
      : c_(numSeries), window_(window), hop_(hop) {
    if (numSeries == 0 || window == 0 || hop == 0) {
      throw WaveTraceError("FeatureExtractor: numSeries, window, hop must be non-zero");
    }
    rings_.reserve(c_);
    for (size_t i = 0; i < c_; ++i) rings_.emplace_back(window_);
    win_.assign(window_, 0.0f);
    scratch_.assign(window_, 0.0f);
    output_.assign(c_ * FEATURES_PER_SERIES, 0.0f);
  }

  size_t numSeries() const { return c_; }
  size_t window() const { return window_; }
  size_t hop() const { return hop_; }
  size_t outputSize() const { return output_.size(); }
  const float* data() const { return output_.data(); }

  // Returns true when a feature vector was emitted (window full and `hop` frames since last), then available via data(). Non-emit frames are O(C).
  bool push(const float* values) {
    for (size_t i = 0; i < c_; ++i) rings_[i].push(values[i]);
    ++sinceEmit_;
    if (rings_[0].size() < window_ || sinceEmit_ < hop_) return false;
    sinceEmit_ = 0;
    for (size_t i = 0; i < c_; ++i) {
      rings_[i].copyOrdered(win_.data());  // chronological window (lag-1/WL need order)
      nineFeatures(win_.data(), window_, scratch_.data(), &output_[i * FEATURES_PER_SERIES]);
    }
    return true;
  }

  void reset() {
    for (auto& r : rings_) r.clear();
    sinceEmit_ = 0;
    std::fill(output_.begin(), output_.end(), 0.0f);
  }

private:
  size_t c_, window_, hop_;
  size_t sinceEmit_ = 0;
  std::vector<RingBuffer<float>> rings_;
  std::vector<float> win_, scratch_, output_;
};

// --- Streaming inter-subcarrier amplitude-dispersion extractor (windows sigma2[p]) -----------

// Turns per-packet inter-subcarrier dispersion into a classifier-ready block: reduces K magnitudes to {mu[p], sigma2[p], cv[p]=std/mu}, emitting nineFeatures of each series every `hop` frames (output 27 = 3*9).
// INPUT CONTRACT: push RAW magnitudes, not gain-locked — a mean lock cancels the flatness that IS the metal signature. O(K) push.
class InterCarrierExtractor {
public:
  static constexpr size_t NUM_SERIES = 3;  // 0 mu, 1 sigma2, 2 cv
  static constexpr size_t FEATURES_PER_SERIES = 9;

  InterCarrierExtractor(size_t window, size_t hop) : window_(window), hop_(hop) {
    if (window == 0 || hop == 0) {
      throw WaveTraceError("InterCarrierExtractor: window, hop must be non-zero");
    }
    rings_.reserve(NUM_SERIES);
    for (size_t i = 0; i < NUM_SERIES; ++i) rings_.emplace_back(window_);
    win_.assign(window_, 0.0f);
    scratch_.assign(window_, 0.0f);
    output_.assign(NUM_SERIES * FEATURES_PER_SERIES, 0.0f);
  }

  size_t window() const { return window_; }
  size_t hop() const { return hop_; }
  size_t outputSize() const { return output_.size(); }
  const float* data() const { return output_.data(); }

  // Push one frame's K subcarrier magnitudes; True when a feature block was emitted (see data()).
  bool push(const float* mags, size_t k) {
    const InterCarrierStat s = interCarrierStats(mags, k);
    const float cv = (s.mean > 1e-12f) ? std::sqrt(s.variance) / s.mean : 0.0f;
    rings_[0].push(s.mean);
    rings_[1].push(s.variance);
    rings_[2].push(cv);
    ++sinceEmit_;
    if (rings_[0].size() < window_ || sinceEmit_ < hop_) return false;
    sinceEmit_ = 0;
    for (size_t i = 0; i < NUM_SERIES; ++i) {
      rings_[i].copyOrdered(win_.data());  // chronological (lag-1/WL need order)
      nineFeatures(win_.data(), window_, scratch_.data(), &output_[i * FEATURES_PER_SERIES]);
    }
    return true;
  }

  void reset() {
    for (auto& r : rings_) r.clear();
    sinceEmit_ = 0;
    std::fill(output_.begin(), output_.end(), 0.0f);
  }

private:
  size_t window_, hop_;
  size_t sinceEmit_ = 0;
  std::vector<RingBuffer<float>> rings_;
  std::vector<float> win_, scratch_, output_;
};

}  // namespace wavetrace
