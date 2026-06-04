// pybind11 module `_wavetrace`: exposes the Phase 1 core types to Python orchestration.
// The hot-path DSP (Phase 3+) will be added to this same single extension.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/CsiFrame.hpp"
#include "core/Errors.hpp"
#include "core/Types.hpp"
#include "hardware/FrameParser.hpp"
#include "hardware/NodeAggregator.hpp"

namespace py = pybind11;
using namespace wavetrace;

// Zero-copy (numAntennas x numSubcarriers) complex64 view sharing the frame's buffer. Writable:
// mutations from NumPy land directly in the C++ grid (no per-frame copy). `frame` is the base
// object so the buffer outlives the array.
static py::array gridView(py::object frame) {
  CsiFrame& f = frame.cast<CsiFrame&>();
  const auto rows = static_cast<py::ssize_t>(f.numAntennas());
  const auto cols = static_cast<py::ssize_t>(f.numSubcarriers());
  const auto elem = static_cast<py::ssize_t>(sizeof(CsiFrame::Sample));
  return py::array_t<std::complex<float>>(
      {rows, cols},                 // shape
      {cols * elem, elem},          // row-major strides
      f.data(),                     // shared buffer
      frame);                       // base keepalive
}

PYBIND11_MODULE(_wavetrace, m) {
  m.doc() = "WaveTrace native core (Phase 1: shared types).";

  py::register_exception<WaveTraceError>(m, "WaveTraceError");
  py::register_exception<FrameError>(m, "FrameError");

  py::class_<CsiFrame>(m, "CsiFrame")
      .def(py::init<uint16_t, uint16_t>(), py::arg("num_antennas"), py::arg("num_subcarriers"))
      .def_property_readonly("num_antennas", &CsiFrame::numAntennas)
      .def_property_readonly("num_subcarriers", &CsiFrame::numSubcarriers)
      .def_property_readonly("size", &CsiFrame::size)
      .def_property("timestamp", &CsiFrame::timestamp, &CsiFrame::setTimestamp)
      .def_property("node_id", &CsiFrame::nodeId, &CsiFrame::setNodeId)
      .def("reshape", &CsiFrame::reshape, py::arg("num_antennas"), py::arg("num_subcarriers"))
      .def_property_readonly("grid", &gridView,
                             "Zero-copy writable complex64 view, shape (num_antennas, num_subcarriers).");

  py::class_<RecognitionResult>(m, "RecognitionResult")
      .def(py::init<>())
      .def_readwrite("class_id", &RecognitionResult::classId)
      .def_readwrite("confidence", &RecognitionResult::confidence)
      .def_readwrite("timestamp", &RecognitionResult::timestamp)
      .def_readwrite("bbox", &RecognitionResult::bbox)
      .def_readwrite("keypoints", &RecognitionResult::keypoints);

  py::class_<Label>(m, "Label")
      .def(py::init<>())
      .def_readwrite("class_id", &Label::classId)
      .def_readwrite("name", &Label::name)
      .def_readwrite("timestamp", &Label::timestamp)
      .def_readwrite("bbox", &Label::bbox)
      .def_readwrite("keypoints", &Label::keypoints);

  // Phase 2 — hardware ingest.
  py::class_<FrameParser>(m, "FrameParser")
      .def(py::init<uint16_t, uint16_t>(), py::arg("num_antennas"), py::arg("num_subcarriers"))
      .def_property_readonly("num_antennas", &FrameParser::numAntennas)
      .def_property_readonly("num_subcarriers", &FrameParser::numSubcarriers)
      .def(
          "parse",
          [](FrameParser& self,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> raw, double timestamp,
             int32_t nodeId) -> const CsiFrame& {
            py::buffer_info info = raw.request();
            return self.parse(static_cast<const uint8_t*>(info.ptr),
                              static_cast<size_t>(info.size), timestamp, nodeId);
          },
          py::arg("raw"), py::arg("timestamp") = 0.0, py::arg("node_id") = -1,
          // Returns the parser's reused CsiFrame (same object each call); tie its lifetime to the
          // parser and keep `raw` alive for the duration of the decode.
          py::return_value_policy::reference_internal, py::keep_alive<0, 2>(),
          "Decode one raw int8 [imag,real] frame into the reused CsiFrame (returned). O(n).");

  py::class_<NodeAggregator>(m, "NodeAggregator")
      .def(py::init<>())
      .def("submit", &NodeAggregator::submit, py::arg("frame"))
      .def_property_readonly("num_nodes", &NodeAggregator::numNodes)
      .def("synced", &NodeAggregator::synced, py::arg("tolerance"),
           "Latest frame per node within `tolerance` s of the newest submit (copies). O(m).");
}
