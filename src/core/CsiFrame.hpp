#pragma once
#include <complex>
#include <cstdint>
#include <vector>

#include "Errors.hpp"

namespace wavetrace {

// One CSI snapshot: an (antenna x subcarrier) grid of decoded complex channel gains, plus
// metadata. Raw int8 I/Q decode lives in the Phase 2 parser; this type only ever holds the
// decoded complex grid.
//
// Storage is a single contiguous row-major buffer (grid[a*numSubcarriers + s]) sized once and
// reused: reshape() keeps capacity, so steady-state per-frame work does zero heap allocation
// (CLAUDE.md "no hot-path allocations"). complex<float> maps 1:1 to NumPy complex64 for the
// zero-copy view exposed in Bindings.cpp.
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

  uint16_t numAntennas() const { return numAntennas_; }
  uint16_t numSubcarriers() const { return numSubcarriers_; }
  size_t size() const { return grid_.size(); }

  double timestamp() const { return timestamp_; }
  void setTimestamp(double t) { timestamp_ = t; }

  // -1 = single-node / unset; real node ids are assigned by the Phase 2 multi-node aggregator.
  int32_t nodeId() const { return nodeId_; }
  void setNodeId(int32_t id) { nodeId_ = id; }

  // Reuse this frame for new dimensions without reallocating when capacity already suffices.
  void reshape(uint16_t numAntennas, uint16_t numSubcarriers) {
    if (numAntennas == 0 || numSubcarriers == 0) {
      throw FrameError("CsiFrame dimensions must be non-zero");
    }
    numAntennas_ = numAntennas;
    numSubcarriers_ = numSubcarriers;
    grid_.resize(static_cast<size_t>(numAntennas) * numSubcarriers);
  }

  Sample& at(uint16_t antenna, uint16_t subcarrier) {
    if (antenna >= numAntennas_ || subcarrier >= numSubcarriers_) {
      throw FrameError("CsiFrame index out of range");
    }
    return grid_[static_cast<size_t>(antenna) * numSubcarriers_ + subcarrier];
  }
  const Sample& at(uint16_t antenna, uint16_t subcarrier) const {
    if (antenna >= numAntennas_ || subcarrier >= numSubcarriers_) {
      throw FrameError("CsiFrame index out of range");
    }
    return grid_[static_cast<size_t>(antenna) * numSubcarriers_ + subcarrier];
  }

  Sample* data() { return grid_.data(); }
  const Sample* data() const { return grid_.data(); }

private:
  uint16_t numAntennas_;
  uint16_t numSubcarriers_;
  double timestamp_ = 0.0;
  int32_t nodeId_ = -1;
  std::vector<Sample> grid_;  // contiguous, reused; sized numAntennas_ * numSubcarriers_
};

}  // namespace wavetrace
