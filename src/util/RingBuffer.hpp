#pragma once
#include <cstddef>
#include <vector>

namespace wavetrace {

// Fixed-capacity circular buffer holding a per-cell time window (e.g. the last W samples for a
// Hampel filter). Preallocated at construction; push() overwrites the oldest sample once full, so
// steady-state use does zero heap allocation (CLAUDE "no hot-path allocations").
//
// Order is NOT preserved on read: copyTo() emits the valid samples in arbitrary order. That is
// fine for the only consumer (median/MAD over the window, which is order-independent) and avoids
// the bookkeeping of a chronological copy in the hot path.
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

  // Copy the `size()` valid samples into dst in chronological order (oldest -> newest). dst must
  // hold >= size() elements. Needed by order-dependent consumers (lag-1 autocorr, waveform-length).
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
