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
  FrameParser(uint16_t antennaCount, uint16_t subcarrierCount)
      : frame_(antennaCount, subcarrierCount) {}

  uint16_t AntennaCount() const { return frame_.AntennaCount(); }
  uint16_t SubcarrierCount() const { return frame_.SubcarrierCount(); }

  // O(n) over n = antennas*subcarriers, one in-place pass, no allocation; throws FrameError on length mismatch.
  const CsiFrame& Parse(const uint8_t* rawIqBytes, size_t rawByteCount, double timestampSeconds, int32_t nodeId) {
    const size_t sampleCount = frame_.Size();
    if (rawByteCount != 2 * sampleCount) {
      throw FrameError("FrameParser: raw length does not match frame geometry");
    }
    CsiFrame::Sample* grid = frame_.Data();
    for (size_t k = 0; k < sampleCount; ++k) {
      // [imag, real] interleaved, imaginary first; recover signed int8 from the unsigned byte.
      const float imag = static_cast<float>(FixSign(rawIqBytes[2 * k]));
      const float real = static_cast<float>(FixSign(rawIqBytes[2 * k + 1]));
      grid[k] = CsiFrame::Sample(real, imag);
    }
    frame_.SetTimestamp(timestampSeconds);
    frame_.SetNodeId(nodeId);
    return frame_;
  }

private:
  // Two's-complement int8 carried in an unsigned byte: 128..255 map to -128..-1.
  static int FixSign(uint8_t rawByte) {
    return rawByte > 127 ? static_cast<int>(rawByte) - 256 : static_cast<int>(rawByte);
  }

  CsiFrame frame_;  // reused across frames; capacity preserved -> no hot-path alloc
};

}  // namespace wavetrace
