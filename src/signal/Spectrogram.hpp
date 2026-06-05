#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

#include "core/Errors.hpp"

namespace wavetrace {

// Sliding "CSI image" builder (HANDOFF Q2 / plan "subcarrier x time"): the standard CNN input in
// CSI-activity literature. Buffers the last `timeSteps` frames of K per-frame values (e.g. the
// gain-locked amplitudes at the NBVI-selected subcarriers, or differential phase) and, once full,
// emits a (K x timeSteps) row-major image (subcarrier x time) every `hop` frames.
//
// Deliberately NO per-subcarrier FFT (HANDOFF Q2 = the CSI-image form): each column is just the raw
// per-frame K-vector, so push is O(K) and the image is a literal time window of the signal. Frames
// are stored as a ring of columns; an emit transposes that ring into the reused row-major output
// (consume/copy before the next emit — same zero-copy contract as Preprocessor). Per-frame push
// O(K); emit O(K * timeSteps) only every hop. Buffers sized in the ctor -> zero hot-path allocation.
class SpectrogramBuilder {
public:
  SpectrogramBuilder(size_t numSubcarriers, size_t timeSteps, size_t hop)
      : k_(numSubcarriers), t_(timeSteps), hop_(hop) {
    if (k_ == 0 || t_ == 0 || hop_ == 0) {
      throw WaveTraceError("SpectrogramBuilder: numSubcarriers, timeSteps, hop must be non-zero");
    }
    cols_.assign(k_ * t_, 0.0f);    // ring of columns: cols_[col*k_ + s]
    output_.assign(k_ * t_, 0.0f);  // (k_ x t_) row-major after an emit
  }

  size_t numSubcarriers() const { return k_; }
  size_t timeSteps() const { return t_; }
  size_t hop() const { return hop_; }
  const float* data() const { return output_.data(); }

  // Push one frame's K values; returns true when an image was emitted (window full and `hop` frames
  // since the last emit), then available via data() as a (K x timeSteps) row-major grid. O(K) on a
  // non-emit frame.
  bool push(const float* values) {
    float* col = &cols_[head_ * k_];
    for (size_t s = 0; s < k_; ++s) col[s] = values[s];
    head_ = (head_ + 1) % t_;
    if (count_ < t_) ++count_;
    ++sinceEmit_;
    if (count_ < t_ || sinceEmit_ < hop_) return false;
    sinceEmit_ = 0;
    // Transpose the column ring -> row-major (K x T): once full, head_ points at the oldest column.
    const size_t tail = head_;
    for (size_t j = 0; j < t_; ++j) {
      const float* c = &cols_[((tail + j) % t_) * k_];
      for (size_t s = 0; s < k_; ++s) output_[s * t_ + j] = c[s];
    }
    return true;
  }

  void reset() {
    head_ = 0;
    count_ = 0;
    sinceEmit_ = 0;
    std::fill(cols_.begin(), cols_.end(), 0.0f);
    std::fill(output_.begin(), output_.end(), 0.0f);
  }

private:
  size_t k_, t_, hop_;
  size_t head_ = 0, count_ = 0, sinceEmit_ = 0;
  std::vector<float> cols_, output_;
};

}  // namespace wavetrace
