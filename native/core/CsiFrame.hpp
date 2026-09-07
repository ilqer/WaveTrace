#pragma once
#include <complex>
#include <cstdint>
#include <vector>

#include "Errors.hpp"

namespace wavetrace {

// One CSI snapshot: (antenna x subcarrier) grid of complex gains + metadata, row-major, alloc-free
// after reshape(); complex<float> maps 1:1 to NumPy complex64 for zero-copy in Bindings.cpp.
class CsiFrame {
public:
  using Sample = std::complex<float>;

  CsiFrame(uint16_t antennaCount, uint16_t subcarrierCount)
      : antennaCount_(antennaCount), subcarrierCount_(subcarrierCount) {
    if (antennaCount == 0 || subcarrierCount == 0) {
      throw FrameError("CsiFrame dimensions must be non-zero");
    }
    grid_.resize(static_cast<size_t>(antennaCount) * subcarrierCount);
  }

  uint16_t AntennaCount() const { return antennaCount_; }
  uint16_t SubcarrierCount() const { return subcarrierCount_; }
  size_t Size() const { return grid_.size(); }

  double TimestampSeconds() const { return timestampSeconds_; }
  void SetTimestamp(double timestampSeconds) { timestampSeconds_ = timestampSeconds; }

  // -1 = single-node / unset; real node ids are assigned by the Phase 2 multi-node aggregator.
  int32_t NodeId() const { return nodeId_; }
  void SetNodeId(int32_t nodeId) { nodeId_ = nodeId; }

  // Reuse this frame for new dimensions without reallocating when capacity already suffices.
  void Reshape(uint16_t antennaCount, uint16_t subcarrierCount) {
    if (antennaCount == 0 || subcarrierCount == 0) {
      throw FrameError("CsiFrame dimensions must be non-zero");
    }
    antennaCount_ = antennaCount;
    subcarrierCount_ = subcarrierCount;
    grid_.resize(static_cast<size_t>(antennaCount) * subcarrierCount);
  }

  Sample& At(uint16_t antenna, uint16_t subcarrier) {
    if (antenna >= antennaCount_ || subcarrier >= subcarrierCount_) {
      throw FrameError("CsiFrame index out of range");
    }
    return grid_[static_cast<size_t>(antenna) * subcarrierCount_ + subcarrier];
  }
  const Sample& At(uint16_t antenna, uint16_t subcarrier) const {
    if (antenna >= antennaCount_ || subcarrier >= subcarrierCount_) {
      throw FrameError("CsiFrame index out of range");
    }
    return grid_[static_cast<size_t>(antenna) * subcarrierCount_ + subcarrier];
  }

  Sample* Data() { return grid_.data(); }
  const Sample* Data() const { return grid_.data(); }

private:
  uint16_t antennaCount_;
  uint16_t subcarrierCount_;
  double timestampSeconds_ = 0.0;
  int32_t nodeId_ = -1;
  std::vector<Sample> grid_;  // contiguous, reused; sized antennaCount_ * subcarrierCount_
};

}  // namespace wavetrace
