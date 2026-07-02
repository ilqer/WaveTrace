#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wavetrace {

// Bounding box in normalized image coords (x, y, w, h); fixed-size array so it carries no heap cost.
using BBox = std::array<float, 4>;

// Ground-truth supervision sample from the camera pipeline; target-agnostic so any future task reuses the same shape.
struct Label {
  int32_t classId = -1;            // task-defined class index; -1 = unset
  std::string name;                // optional human-readable class name ("" if unset)
  double timestamp = 0.0;          // seconds, for camera<->CSI alignment
  std::optional<BBox> bbox;        // present only for spatial tasks
  std::vector<float> keypoints;    // flattened (x,y,conf...); empty when none
  // Flattened maskGrid x maskGrid occupancy heatmap in [0,1] (row-major); BCE/Dice target for the weapon-heatmap head.
  std::vector<float> mask;
  int32_t maskGrid = 0;            // G: side of the square mask grid; 0 = no mask
};

// Model output for one frame/window; mirrors Label's optional bbox/keypoints so any head can reuse it.
struct RecognitionResult {
  int32_t classId = -1;            // predicted class index
  float confidence = 0.0f;         // [0,1]
  double timestamp = 0.0;
  std::optional<BBox> bbox;        // present only for spatial tasks
  std::vector<float> keypoints;    // flattened (x,y,conf...); empty when none
};

}  // namespace wavetrace
