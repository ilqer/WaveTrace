#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "core/Errors.hpp"

namespace wavetrace {

// Smallest power of two >= n (>= 1). Used to choose a zero-padded FFT length (REFERENCE §2.6).
inline size_t nextPow2(size_t n) {
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

// Self-contained radix-2 Cooley-Tukey FFT (decimation-in-time), REFERENCE_DIGEST §2.6. Sized once
// for a fixed power-of-two length; the bit-reversal permutation and twiddle factors are precomputed
// in the ctor so forward() does ZERO allocation — it can run per emit in the hot path. O(n log n).
//
// Kept in-house (vs vendoring kissfft/pocketfft) so the native extension stays zero-dependency and
// C++ keeps ownership of the hot-path DSP (plan §2.1).
class Fft {
public:
  explicit Fft(size_t n) : n_(n) {
    if (n_ == 0 || (n_ & (n_ - 1)) != 0) {
      throw WaveTraceError("Fft: size must be a power of two");
    }
    size_t logn = 0;
    while ((static_cast<size_t>(1) << logn) < n_) ++logn;

    rev_.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
      size_t r = 0;
      for (size_t b = 0; b < logn; ++b) {
        if (i & (static_cast<size_t>(1) << b)) r |= (static_cast<size_t>(1) << (logn - 1 - b));
      }
      rev_[i] = r;
    }

    // Twiddles W_n^k = exp(-2*pi*i*k/n) for k in [0, n/2); double accumulation for accuracy.
    constexpr double TWO_PI = 6.283185307179586476925286766559;
    tw_.resize(n_ / 2);
    for (size_t k = 0; k < n_ / 2; ++k) {
      const double a = -TWO_PI * static_cast<double>(k) / static_cast<double>(n_);
      tw_[k] = std::complex<float>(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
    }
  }

  size_t size() const { return n_; }

  // In-place forward FFT on exactly size() complex samples. O(n log n), no allocation.
  void forward(std::complex<float>* x) const {
    for (size_t i = 0; i < n_; ++i) {
      const size_t j = rev_[i];
      if (i < j) std::swap(x[i], x[j]);  // bit-reversal reorder (each pair swapped once)
    }
    for (size_t len = 2; len <= n_; len <<= 1) {
      const size_t half = len >> 1;
      const size_t step = n_ / len;  // stride into the W_n twiddle table for this stage
      for (size_t base = 0; base < n_; base += len) {
        for (size_t k = 0; k < half; ++k) {
          const std::complex<float> w = tw_[k * step];
          const std::complex<float> u = x[base + k];
          const std::complex<float> v = x[base + k + half] * w;
          x[base + k] = u + v;
          x[base + k + half] = u - v;
        }
      }
    }
  }

private:
  size_t n_;
  std::vector<size_t> rev_;              // bit-reversal permutation
  std::vector<std::complex<float>> tw_;  // precomputed twiddle factors, length n/2
};

}  // namespace wavetrace
