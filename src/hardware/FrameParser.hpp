#pragma once
#include <cstddef>
#include <cstdint>

#include "core/CsiFrame.hpp"
#include "core/Errors.hpp"

namespace wavetrace {

// Decodes one raw ESP32 CSI frame into a reused CsiFrame. Wire layout (REFERENCE_DIGEST §2.1):
// interleaved int8 I/Q, **imaginary first** ([imag, real] per subcarrier), delivered as unsigned
// bytes (0..255) that are two's-complement int8 — so a sign fixup (v -= 256 if v > 127, §4) is
// applied before building each complex sample.
//
// Bound once to a sensor's (antenna x subcarrier) geometry but geometry-agnostic: 1x1, 1xN, Nx1,
// MxN all work (the link-count decision stays open, plan §2.9). The owned CsiFrame is reused
// across frames, so steady-state parsing does zero heap allocation (CLAUDE "no hot-path allocs").
class FrameParser {
public:
  FrameParser(uint16_t numAntennas, uint16_t numSubcarriers)
      : frame_(numAntennas, numSubcarriers) {}

  uint16_t numAntennas() const { return frame_.numAntennas(); }
  uint16_t numSubcarriers() const { return frame_.numSubcarriers(); }

  // Decode `raw` (length must equal 2 * numAntennas * numSubcarriers) into the owned frame and
  // return it, stamped with timestamp + nodeId. O(n) over n = antennas*subcarriers; one in-place
  // pass, no allocation. Throws FrameError on a length mismatch (a dropped/truncated packet) so a
  // malformed frame never becomes a panic or a misaligned read.
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
