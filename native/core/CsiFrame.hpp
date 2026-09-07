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

  CsiFrame(uint16_t numAntennas, uint16_t numSubcarriers)
      : numAntennas_(numAntennas), numSubcarriers_(numSubcarriers) {
    if (numAntennas == 0 || numSubcarriers == 0) {
      throw FrameError("CsiFrame dimensions must be non-zero");
    }
    grid_.resize(static_cast<size_t>(numAntennas) * numSubcarriers);
  }

  uint16_t NumAntennas() const { return numAntennas_; }
  uint16_t NumSubcarriers() const { return numSubcarriers_; }
  size_t Size() const { return grid_.size(); }

  double Timestamp() const { return timestamp_; }
  void SetTimestamp(double timestampSeconds) { timestamp_ = timestampSeconds; }

  // -1 = single-node / unset; real node ids are assigned by the Phase 2 multi-node aggregator.
  int32_t NodeId() const { return nodeId_; }
  void SetNodeId(int32_t nodeId) { nodeId_ = nodeId; }

  // Reuse this frame for new dimensions without reallocating when capacity already suffices.
  void Reshape(uint16_t numAntennas, uint16_t numSubcarriers) {
    if (numAntennas == 0 || numSubcarriers == 0) {
      throw FrameError("CsiFrame dimensions must be non-zero");
    }
    numAntennas_ = numAntennas;
    numSubcarriers_ = numSubcarriers;
    grid_.resize(static_cast<size_t>(numAntennas) * numSubcarriers);
  }

  Sample& At(uint16_t antenna, uint16_t subcarrier) {
    if (antenna >= numAntennas_ || subcarrier >= numSubcarriers_) {
      throw FrameError("CsiFrame index out of range");
    }
    return grid_[static_cast<size_t>(antenna) * numSubcarriers_ + subcarrier];
  }
  const Sample& At(uint16_t antenna, uint16_t subcarrier) const {
    if (antenna >= numAntennas_ || subcarrier >= numSubcarriers_) {
      throw FrameError("CsiFrame index out of range");
    }
    return grid_[static_cast<size_t>(antenna) * numSubcarriers_ + subcarrier];
  }

  Sample* Data() { return grid_.data(); }
  const Sample* Data() const { return grid_.data(); }

private:
  uint16_t numAntennas_;
  uint16_t numSubcarriers_;
  double timestamp_ = 0.0;
  int32_t nodeId_ = -1;
  std::vector<Sample> grid_;  // contiguous, reused; sized numAntennas_ * numSubcarriers_
};

}  // namespace wavetrace
