#pragma once
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/CsiFrame.hpp"

namespace wavetrace {

// Buffers the latest CsiFrame per node (multistatic capture). Nodes have independent CFO/SFO so they
// can't be conjugate-multiplied across nodes; this stages fusion by grouping frames within a timestamp tolerance.
// m = number of nodes (small): submit is O(n) (one frame's grid), synced/numNodes are O(m).
class NodeAggregator {
public:
  // Slot is reused after the node is first seen, so only a new node id allocates.
  void submit(const CsiFrame& frame) {
    auto it = nodes_.find(frame.nodeId());
    if (it == nodes_.end()) {
      nodes_.emplace(frame.nodeId(), frame);  // first sighting of this node allocates its slot
    } else {
      it->second = frame;  // reuse existing slot
    }
    newestTs_ = std::max(newestTs_, frame.timestamp());
  }

  size_t numNodes() const { return nodes_.size(); }

  // Frames within `tolerance` seconds of the newest submitted frame; returns copies (safe across the FFI; m small).
  std::vector<CsiFrame> synced(double tolerance) const {
    std::vector<CsiFrame> out;
    out.reserve(nodes_.size());
    for (const auto& [id, f] : nodes_) {
      (void)id;
      if (newestTs_ - f.timestamp() <= tolerance) out.push_back(f);
    }
    return out;
  }

private:
  std::unordered_map<int32_t, CsiFrame> nodes_;
  double newestTs_ = 0.0;
};

}  // namespace wavetrace
