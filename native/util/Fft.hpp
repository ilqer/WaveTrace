#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "core/Errors.hpp"

namespace wavetrace {

// Smallest power of two >= minimumSize (>= 1). Used to choose a zero-padded FFT length (REFERENCE §2.6).
inline size_t NextPow2(size_t minimumSize) {
  size_t candidate = 1;
  while (candidate < minimumSize) candidate <<= 1;
  return candidate;
}

// Self-contained radix-2 Cooley-Tukey FFT (decimation-in-time), kept in-house for a zero-dependency extension.
// Sized once for a fixed power-of-two length; bit-reversal + twiddles precomputed in ctor so Forward() is alloc-free. O(n log n).
class Fft {
public:
  explicit Fft(size_t length) : length_(length) {
    if (length_ == 0 || (length_ & (length_ - 1)) != 0) {
      throw WaveTraceError("Fft: size must be a power of two");
    }
    size_t log2Length = 0;
    while ((static_cast<size_t>(1) << log2Length) < length_) ++log2Length;

    bitReversalIndices_.resize(length_);
    for (size_t i = 0; i < length_; ++i) {
      size_t reversedIndex = 0;
      for (size_t bit = 0; bit < log2Length; ++bit) {
        if (i & (static_cast<size_t>(1) << bit))
          reversedIndex |= (static_cast<size_t>(1) << (log2Length - 1 - bit));
      }
      bitReversalIndices_[i] = reversedIndex;
    }

    // Twiddles W_n^k = exp(-2*pi*i*k/n) for k in [0, n/2); double accumulation for accuracy.
    constexpr double TWO_PI = 6.283185307179586476925286766559;
    twiddleFactors_.resize(length_ / 2);
    for (size_t k = 0; k < length_ / 2; ++k) {
      const double angle = -TWO_PI * static_cast<double>(k) / static_cast<double>(length_);
      twiddleFactors_[k] =
          std::complex<float>(static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle)));
    }
  }

  size_t Size() const { return length_; }

  // In-place forward FFT on exactly Size() complex samples. O(n log n), no allocation.
  void Forward(std::complex<float>* samples) const {
    for (size_t i = 0; i < length_; ++i) {
      const size_t j = bitReversalIndices_[i];
      if (i < j) std::swap(samples[i], samples[j]);  // bit-reversal reorder (each pair swapped once)
    }
    for (size_t stageLength = 2; stageLength <= length_; stageLength <<= 1) {
      const size_t halfStageLength = stageLength >> 1;
      const size_t twiddleStride = length_ / stageLength;  // stride into the W_n twiddle table for this stage
      for (size_t blockStart = 0; blockStart < length_; blockStart += stageLength) {
        for (size_t k = 0; k < halfStageLength; ++k) {
          const std::complex<float> twiddle = twiddleFactors_[k * twiddleStride];
          const std::complex<float> topTerm = samples[blockStart + k];
          const std::complex<float> bottomTerm = samples[blockStart + k + halfStageLength] * twiddle;
          samples[blockStart + k] = topTerm + bottomTerm;
          samples[blockStart + k + halfStageLength] = topTerm - bottomTerm;
        }
      }
    }
  }

private:
  size_t length_;
  std::vector<size_t> bitReversalIndices_;
  std::vector<std::complex<float>> twiddleFactors_;  // precomputed twiddle factors, length n/2
};

}  // namespace wavetrace
