#pragma once
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/CsiFrame.hpp"

namespace wavetrace {

// Buffers the latest CsiFrame from each of any number of independent-clock nodes (multistatic
// capture). Separate ESP32 nodes have independent CFO/SFO, so they CANNOT be conjugate-multiplied
// across nodes (plan §2.8) — each self-cancels, then fuses at the feature/label level. This class
// is that fusion staging point: it groups the most recent frame per node id and returns the set
// whose timestamps agree within a tolerance, the time-synced bundle a fuser consumes.
//
// m = number of nodes (small). submit is O(n) (copies one frame's grid into its node slot);
// synced/numNodes are O(m).
class NodeAggregator {
public:
  // Store or overwrite the latest frame for its node id. The per-node slot is reused after the
  // node is first seen (vector assignment keeps capacity), so only a new node id allocates.
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

  // Latest frame from every node whose timestamp is within `tolerance` seconds of the newest
  // submitted frame — the synchronized set to fuse. Returns copies (safe across the FFI; m small).
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
