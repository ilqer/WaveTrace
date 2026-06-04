#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wavetrace {

// Bounding box in normalized image coords (x, y, w, h) — used when a head/label is spatial
// (weapon box, person box). Kept as a fixed-size array so it carries no heap cost.
using BBox = std::array<float, 4>;

// Ground-truth supervision sample from the camera pipeline (Phase 5). Target-agnostic on
// purpose: posture, weapon, or any future task reuses the same shape (plan.md Phase 1 pivot).
struct Label {
  int32_t classId = -1;            // task-defined class index; -1 = unset
  std::string name;                // optional human-readable class name ("" if unset)
  double timestamp = 0.0;          // seconds, for camera<->CSI alignment
  std::optional<BBox> bbox;        // present only for spatial tasks
  std::vector<float> keypoints;    // flattened (x,y,conf...); empty when none
};

// Model output for one frame/window. Same optional bbox/keypoints as Label so a head can emit
// whatever its task needs without a new type per target.
struct RecognitionResult {
  int32_t classId = -1;            // predicted class index
  float confidence = 0.0f;         // [0,1]
  double timestamp = 0.0;
  std::optional<BBox> bbox;        // present only for spatial tasks
  std::vector<float> keypoints;    // flattened (x,y,conf...); empty when none
};

}  // namespace wavetrace
