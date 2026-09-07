#pragma once
#include <cstddef>
#include <vector>

namespace wavetrace {

// Fixed-capacity circular buffer for a per-cell time window; Push() overwrites the oldest sample once full, so
// steady-state use is alloc-free. CopyTo() is unordered (fine for order-independent consumers like median/MAD).
template <typename T>
class RingBuffer {
public:
  explicit RingBuffer(size_t capacity) : buffer_(capacity) {}

  void Push(T value) {
    buffer_[head_] = value;
    head_ = (head_ + 1) % buffer_.size();
    if (count_ < buffer_.size()) ++count_;
  }

  size_t Size() const { return count_; }
  size_t Capacity() const { return buffer_.size(); }

  // Copy the `Size()` valid samples into destination (unordered). destination must hold >= Size() elements.
  void CopyTo(T* destination) const {
    for (size_t i = 0; i < count_; ++i) destination[i] = buffer_[i];
  }

  // Chronological order (oldest -> newest); needed by order-dependent consumers (lag-1 autocorr, waveform-length).
  void CopyOrdered(T* destination) const {
    const size_t tail = (head_ + buffer_.size() - count_) % buffer_.size();  // index of the oldest sample
    for (size_t i = 0; i < count_; ++i) destination[i] = buffer_[(tail + i) % buffer_.size()];
  }

  void Clear() {
    head_ = 0;
    count_ = 0;
  }

private:
  std::vector<T> buffer_;
  size_t head_ = 0;
  size_t count_ = 0;
};

}  // namespace wavetrace
