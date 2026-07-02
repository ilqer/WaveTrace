#pragma once
#include <cstddef>
#include <cstdint>

#include "core/CsiFrame.hpp"
#include "core/Errors.hpp"

namespace wavetrace {

// Decodes a raw CSI frame into a reused CsiFrame: interleaved int8 I/Q (imag first, two's-complement
// fixup), geometry-agnostic, alloc-free.
class FrameParser {
public:
  FrameParser(uint16_t numAntennas, uint16_t numSubcarriers)
      : frame_(numAntennas, numSubcarriers) {}

  uint16_t numAntennas() const { return frame_.numAntennas(); }
  uint16_t numSubcarriers() const { return frame_.numSubcarriers(); }

  // O(n) over n = antennas*subcarriers, one in-place pass, no allocation; throws FrameError on length mismatch.
  const CsiFrame& parse(const uint8_t* raw, size_t len, double timestamp, int32_t nodeId) {
    const size_t n = frame_.size();
    if (len != 2 * n) {
      throw FrameError("FrameParser: raw length does not match frame geometry");
    }
    CsiFrame::Sample* grid = frame_.data();
    for (size_t k = 0; k < n; ++k) {
      // [imag, real] interleaved, imaginary first; recover signed int8 from the unsigned byte.
      const float imag = static_cast<float>(fixSign(raw[2 * k]));
      const float real = static_cast<float>(fixSign(raw[2 * k + 1]));
      grid[k] = CsiFrame::Sample(real, imag);
    }
    frame_.setTimestamp(timestamp);
    frame_.setNodeId(nodeId);
    return frame_;
  }

private:
  // Two's-complement int8 carried in an unsigned byte: 128..255 map to -128..-1.
  static int fixSign(uint8_t v) {
    return v > 127 ? static_cast<int>(v) - 256 : static_cast<int>(v);
  }

  CsiFrame frame_;  // reused across frames; capacity preserved -> no hot-path alloc
};

}  // namespace wavetrace
