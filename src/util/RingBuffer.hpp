#pragma once
#include <cstddef>
#include <vector>

namespace wavetrace {

// Fixed-capacity circular buffer for a per-cell time window; push() overwrites the oldest sample once full, so
// steady-state use is alloc-free. copyTo() is unordered (fine for order-independent consumers like median/MAD).
template <typename T>
class RingBuffer {
public:
  explicit RingBuffer(size_t capacity) : buf_(capacity) {}

  void push(T v) {
    buf_[head_] = v;
    head_ = (head_ + 1) % buf_.size();
    if (count_ < buf_.size()) ++count_;
  }

  size_t size() const { return count_; }
  size_t capacity() const { return buf_.size(); }

  // Copy the `size()` valid samples into dst (unordered). dst must hold >= size() elements.
  void copyTo(T* dst) const {
    for (size_t i = 0; i < count_; ++i) dst[i] = buf_[i];
  }

  // Chronological order (oldest -> newest); needed by order-dependent consumers (lag-1 autocorr, waveform-length).
  void copyOrdered(T* dst) const {
    const size_t tail = (head_ + buf_.size() - count_) % buf_.size();  // index of the oldest sample
    for (size_t i = 0; i < count_; ++i) dst[i] = buf_[(tail + i) % buf_.size()];
  }

  void clear() {
    head_ = 0;
    count_ = 0;
  }

private:
  std::vector<T> buf_;
  size_t head_ = 0;
  size_t count_ = 0;
};

}  // namespace wavetrace
