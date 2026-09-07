#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

#include "core/Errors.hpp"

namespace wavetrace {

// Sliding "CSI image" builder: buffers the last `timeSteps` frames of K values as a column ring, emitting a
// (K x timeSteps) row-major image every `hop` frames. No per-subcarrier FFT. Push O(K), emit O(K*timeSteps), alloc-free.
class SpectrogramBuilder {
public:
  SpectrogramBuilder(size_t subcarrierCount, size_t timeSteps, size_t hop)
      : subcarrierCount_(subcarrierCount), timeStepCount_(timeSteps), hop_(hop) {
    if (subcarrierCount_ == 0 || timeStepCount_ == 0 || hop_ == 0) {
      throw WaveTraceError("SpectrogramBuilder: subcarrierCount, timeSteps, hop must be non-zero");
    }
    cols_.assign(subcarrierCount_ * timeStepCount_, 0.0f);
    output_.assign(subcarrierCount_ * timeStepCount_, 0.0f);
  }

  size_t SubcarrierCount() const { return subcarrierCount_; }
  size_t TimeSteps() const { return timeStepCount_; }
  size_t Hop() const { return hop_; }
  const float* Data() const { return output_.data(); }

  // Returns true when an image was emitted (window full and `hop` frames since the last emit), then available via Data(). O(K) otherwise.
  bool Push(const float* values) {
    float* destColumn = &cols_[head_ * subcarrierCount_];
    for (size_t subcarrierIndex = 0; subcarrierIndex < subcarrierCount_; ++subcarrierIndex)
      destColumn[subcarrierIndex] = values[subcarrierIndex];
    head_ = (head_ + 1) % timeStepCount_;
    if (count_ < timeStepCount_) ++count_;
    ++sinceEmit_;
    if (count_ < timeStepCount_ || sinceEmit_ < hop_) return false;
    sinceEmit_ = 0;
    // Transpose the column ring -> row-major (K x T): once full, head_ points at the oldest column.
    const size_t oldestColumnIndex = head_;
    for (size_t j = 0; j < timeStepCount_; ++j) {
      const float* srcColumn = &cols_[((oldestColumnIndex + j) % timeStepCount_) * subcarrierCount_];
      for (size_t subcarrierIndex = 0; subcarrierIndex < subcarrierCount_; ++subcarrierIndex)
        output_[subcarrierIndex * timeStepCount_ + j] = srcColumn[subcarrierIndex];
    }
    return true;
  }

  void Reset() {
    head_ = 0;
    count_ = 0;
    sinceEmit_ = 0;
    std::fill(cols_.begin(), cols_.end(), 0.0f);
    std::fill(output_.begin(), output_.end(), 0.0f);
  }

private:
  size_t subcarrierCount_, timeStepCount_, hop_;
  size_t head_ = 0, count_ = 0, sinceEmit_ = 0;
  std::vector<float> cols_, output_;
};

}  // namespace wavetrace
