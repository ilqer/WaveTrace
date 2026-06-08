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

// Phase 4 — windowed feature front-end (target-agnostic recognition input). Two paths:
//   * §2.9 nine-feature TIME-domain vector per series (the proven, compact base — fewer features
//     generalized better in the reference; kurtosis/entropy/slope were deliberately dropped, §2.9/§4).
//   * FREQUENCY-domain cues the time-domain nine cannot capture: a power spectrum (PSD) and Doppler
//     (max shift + spectral spread). Doppler is the high-value addition from the plan's richer set —
//     a motion/velocity cue (f_d = 2v/lambda) for carried/drawn weapon and dynamic posture. A full
//     amplitude+phase+PSD concatenation is intentionally deferred until a head needs it (the FFT is
//     now here, so it is cheap to add).

// --- §2.9 order statistics helper -----------------------------------------------------------

// Percentile of an ASCENDING-sorted buffer with linear interpolation (matches numpy's default,
// so the unit test can compare against np.percentile directly). p in [0,1]. O(1).
inline float percentileSorted(const float* sorted, size_t n, float p) {
  if (n == 1) return sorted[0];
  const float idx = p * static_cast<float>(n - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = (lo + 1 < n) ? lo + 1 : lo;
  const float frac = idx - static_cast<float>(lo);
  return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

// REFERENCE_DIGEST §2.9 nine-feature vector over one CHRONOLOGICAL window (a subcarrier's amplitude
// series, or the turbulence scalar). Fills out[0..8] in this fixed order:
//   0 mean · 1 std · 2 max · 3 min · 4 IQR(P75-P25) · 5 skewness · 6 lag-1 autocorrelation ·
//   7 MAD(median|x-median|) · 8 waveform-length(Σ|x_i - x_{i-1}|)
// scratch must hold >= n floats (order statistics). Two-pass moments (stable, §2.8). The IQR/MAD
// sorts make this O(n log n); the moments and the two order-dependent features are O(n).
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

// Per-packet dispersion of the K subcarrier magnitudes at ONE frame (Yousaf 2025 Eq.3-4 / LUMS Eq.3-4,
// REFERENCE §0B). A flat metal reflector reflects all subcarriers evenly -> sigma2 is SMALLER than for
// the diffuse human body, so this is the documented concealed-metal discriminator (and the cheapest
// threshold head). It is the TRANSPOSE of nineFeatures (which is per-subcarrier across time) and equals
// the espectre "spatial turbulence" scalar (§2.8 MVS), so one primitive serves Stage-A presence and
// Stage-E weapon. Run it over ALL valid subcarriers, not the NBVI subset (NBVI selects on time-variance,
// orthogonal to this cross-subcarrier reduction). Sample variance (M-1) matches the papers. Two-pass
// (stable, §2.8). O(K), no allocation.
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
  float residualStd;  // RMS phase after removing the linear ToF slope; a coherent flat reflector ->
                      // LOWER residual, the diffuse human body -> HIGHER (scattered phase)
};

// Inter-subcarrier phase dispersion at ONE frame: unwrap the phase across subcarriers, fit a line
// (the linear term is the group-delay/ToF slope), and return the slope + the RMS of the non-linear
// residual. A metal reflector reflects coherently -> near-linear phase across the band -> small
// residual; the diffuse human body scatters -> large residual. This is the PHASE analogue of
// interCarrierStats (Wi-Metal is phase-based: phase resolves mm-level path-length change, so the
// slope is the delay term compressed sensing later super-resolves). `phase` = per-subcarrier phase at
// one frame (e.g. std::arg of each H[k], or a Preprocessor differential-phase row). `scratch` holds
// >= k floats (the unwrapped phase). O(k), no allocation beyond the caller's scratch.
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

// --- Frequency domain: PSD + Doppler (REFERENCE §2.6) ---------------------------------------

// Power spectral density of a real series via the §2.6 recipe: detrend (subtract mean), Hann
// window (reduces leakage), zero-pad to fft.size(), FFT, power = re^2 + im^2 over the first
// nfft/2+1 bins. `scratch` is complex work of length fft.size(); `power` holds >= nfft/2+1 floats.
// Caller owns the Fft (size = a power of two >= n). O(nfft log nfft).
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

// Doppler features from the power spectrum of a (differential-phase) window. The peak frequency in
// (0, fHi] is the max Doppler shift; the power-weighted RMS deviation about the spectral centroid is
// the spread. DC (bin 0) is excluded so a residual offset cannot masquerade as the peak. Caller
// supplies fft + complex scratch (fft.size()) + power buffer (>= nfft/2+1). O(nfft log nfft).
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

// Streaming nine-feature extractor over C parallel scalar series (e.g. the gain-locked amplitudes at
// the K selected subcarriers, one series each). Buffers the last `window` samples per series and,
// once the window is full, emits the 9-feature block per series every `hop` frames into a reused
// output vector of length 9*C (consume/copy before the next emit — same zero-copy contract as
// Preprocessor). Per-frame push O(C); emit O(C * window log window), only every hop. Buffers sized
// in the ctor -> zero hot-path allocation. Memory ~ C * (window + ~9) floats.
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

  // Push one frame's C values; returns true when a feature vector was emitted (window full and `hop`
  // frames since the last emit), then available via data(). Non-emit frames are O(C).
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

// Turns the per-packet inter-subcarrier amplitude statistic into a classifier-ready feature block —
// the change that makes the §0B weapon discriminator usable by a head (the signal is in how
// sigma2[p] BEHAVES over the ~1.3 s window, not in one packet). Each frame's K subcarrier magnitudes
// are reduced to {mu[p], sigma2[p], cv[p]=std/mu}; each scalar is buffered as a series and, once
// `window` frames are in, the §2.9 nineFeatures of each series are emitted every `hop` frames
// (output length 27 = 3*9, order: mu | sigma2 | cv).
// INPUT CONTRACT: push RAW per-frame magnitudes (NOT gain-locked / mean-normalized) — a per-frame
// mean lock cancels the cross-subcarrier flatness that IS the metal signature; cv[p] is the
// gain-invariant series to prefer if a gain lock is unavoidable. Run over ALL valid subcarriers, not
// the NBVI subset (NBVI ranks on time-variance, orthogonal to this cross-subcarrier reduction).
// Per-frame push O(K); emit O(window log window) only every hop. Zero hot-path allocation.
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
