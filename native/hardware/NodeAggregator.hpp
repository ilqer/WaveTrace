#pragma once
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/CsiFrame.hpp"

namespace wavetrace {

// Buffers the latest CsiFrame per node; independent CFO/SFO per node rules out cross-node conjugate-
// multiply, so this groups frames within a timestamp tolerance instead. Submit is O(n), Synced/NumNodes O(m).
class NodeAggregator {
public:
  // Slot is reused after the node is first seen, so only a new node id allocates.
  void Submit(const CsiFrame& frame) {
    auto it = nodes_.find(frame.NodeId());
    if (it == nodes_.end()) {
      nodes_.emplace(frame.NodeId(), frame);  // first sighting of this node allocates its slot
    } else {
      it->second = frame;  // reuse existing slot
    }
    newestTimestamp_ = std::max(newestTimestamp_, frame.Timestamp());
  }

  size_t NumNodes() const { return nodes_.size(); }

  // Frames within `tolerance` seconds of the newest submitted frame; returns copies (safe across the FFI; m small).
  std::vector<CsiFrame> Synced(double tolerance) const {
    std::vector<CsiFrame> out;
    out.reserve(nodes_.size());
    for (const auto& [id, f] : nodes_) {
      (void)id;
      if (newestTimestamp_ - f.Timestamp() <= tolerance) out.push_back(f);
    }
    return out;
  }

private:
  std::unordered_map<int32_t, CsiFrame> nodes_;
  double newestTimestamp_ = 0.0;
};

}  // namespace wavetrace
