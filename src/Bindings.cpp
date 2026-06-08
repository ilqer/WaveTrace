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
#include "signal/Features.hpp"
#include "signal/GainLock.hpp"
#include "signal/PresenceSegment.hpp"
#include "signal/Preprocess.hpp"
#include "signal/Spectrogram.hpp"
#include "signal/SubcarrierSelect.hpp"
#include "util/Fft.hpp"

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

  // Phase 3 — signal preprocessing. Stateless transforms first (bound for unit tests), then the
  // streaming Preprocessor.
  m.def("conjugate_multiply", &conjugateMultiply, py::arg("in_frame"), py::arg("out_frame"),
        "Geometry-adaptive conjugate multiply (cancels CFO/SFO) into out_frame (reshaped). O(n).");
  m.def(
      "hampel",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> window, float current,
         float k) {
        py::buffer_info info = window.request();
        std::vector<float> scratch(static_cast<size_t>(info.size));
        return hampel(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size), current,
                      scratch.data(), k);
      },
      py::arg("window"), py::arg("current"), py::arg("k") = 5.0f,
      "Hampel test: median if `current` is an outlier vs `window`, else `current`.");
  m.def("unwrap_step", &unwrapStep, py::arg("cur_wrapped"), py::arg("prev_wrapped"),
        py::arg("prev_unwrapped"), "One streaming phase-unwrap step. O(1).");

  m.def(
      "coefficient_of_variation",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> amp) {
        py::buffer_info info = amp.request();
        return coefficientOfVariation(static_cast<const float*>(info.ptr),
                                      static_cast<size_t>(info.size));
      },
      py::arg("amplitudes"), "Gain-invariant variability sigma/mu. O(n).");

  py::class_<GainLock>(m, "GainLock")
      .def(py::init<size_t>(), py::arg("baseline_packets") = 300)
      .def("observe", &GainLock::observe, py::arg("frame"))
      .def("finalize", &GainLock::finalize)
      .def("apply", &GainLock::apply, py::arg("frame"))
      .def_property_readonly("observed", &GainLock::observed)
      .def_property_readonly("ready", &GainLock::ready)
      .def_property_readonly("locked", &GainLock::locked)
      .def_property_readonly("reference_scale", &GainLock::referenceScale);

  // NBVI subcarrier selection (offline). amp = float32 (frames x subcarriers) baseline matrix.
  m.def(
      "nbvi_scores",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> amp, float alpha) {
        py::buffer_info info = amp.request();
        if (info.ndim != 2) throw WaveTraceError("nbvi_scores: amp must be 2D (frames x subcarriers)");
        return nbviScores(static_cast<const float*>(info.ptr), static_cast<size_t>(info.shape[0]),
                          static_cast<size_t>(info.shape[1]), alpha);
      },
      py::arg("amp"), py::arg("alpha") = 0.75f, "Per-subcarrier NBVI over a baseline matrix.");
  m.def(
      "select_subcarriers_nbvi",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> amp, float alpha,
         size_t max_subcarriers, float noise_gate_percentile) {
        py::buffer_info info = amp.request();
        if (info.ndim != 2)
          throw WaveTraceError("select_subcarriers_nbvi: amp must be 2D (frames x subcarriers)");
        NbviParams p;
        p.alpha = alpha;
        p.maxSubcarriers = max_subcarriers;
        p.noiseGatePercentile = noise_gate_percentile;
        return selectSubcarriersNbvi(static_cast<const float*>(info.ptr),
                                     static_cast<size_t>(info.shape[0]),
                                     static_cast<size_t>(info.shape[1]), p);
      },
      py::arg("amp"), py::arg("alpha") = 0.75f, py::arg("max_subcarriers") = 12,
      py::arg("noise_gate_percentile") = 0.15f,
      "Up to max_subcarriers non-consecutive informative subcarriers (indices). Offline.");

  py::class_<Preprocessor>(m, "Preprocessor")
      .def(py::init<uint16_t, uint16_t, size_t, float, float>(), py::arg("num_antennas"),
           py::arg("num_subcarriers"), py::arg("hampel_window") = 7, py::arg("hampel_k") = 5.0f,
           py::arg("normalize_alpha") = 0.1f)
      .def_property_readonly("out_rows", &Preprocessor::outRows)
      .def_property_readonly("out_cols", &Preprocessor::outCols)
      .def("reset", &Preprocessor::reset)
      .def(
          "process",
          [](py::object self, const CsiFrame& in) -> py::array {
            Preprocessor& p = self.cast<Preprocessor&>();
            p.process(in);
            const auto rows = static_cast<py::ssize_t>(p.outRows());
            const auto cols = static_cast<py::ssize_t>(p.outCols());
            const auto elem = static_cast<py::ssize_t>(sizeof(float));
            // Zero-copy float32 view of the reused output grid (same buffer each call).
            return py::array_t<float>({rows, cols}, {cols * elem, elem}, p.data(), self);
          },
          py::arg("frame"),
          "Process one frame -> drift-free differential-phase grid (zero-copy view). O(n).");

  // Phase 4 — features + spectrogram. Stateless FFT/feature fns (bound for unit tests), then the
  // streaming FeatureExtractor and SpectrogramBuilder.
  m.def(
      "fft",
      [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> x) {
        py::buffer_info info = x.request();
        const size_t n = static_cast<size_t>(info.size);
        Fft fft(n);  // throws if n is not a power of two
        py::array_t<std::complex<float>> out(static_cast<py::ssize_t>(n));
        py::buffer_info oi = out.request();
        std::copy_n(static_cast<const std::complex<float>*>(info.ptr), n,
                    static_cast<std::complex<float>*>(oi.ptr));
        fft.forward(static_cast<std::complex<float>*>(oi.ptr));
        return out;
      },
      py::arg("x"), "Radix-2 forward FFT of a power-of-two complex64 array. O(n log n).");

  m.def(
      "nine_features",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> window) {
        py::buffer_info info = window.request();
        const size_t n = static_cast<size_t>(info.size);
        std::vector<float> scratch(n);
        std::vector<float> out(9);
        nineFeatures(static_cast<const float*>(info.ptr), n, scratch.data(), out.data());
        return out;
      },
      py::arg("window"),
      "REFERENCE §2.9 nine features [mean,std,max,min,IQR,skew,lag1,MAD,WL] over one window.");

  m.def(
      "inter_carrier_stats",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> mags) {
        py::buffer_info info = mags.request();
        const InterCarrierStat s = interCarrierStats(static_cast<const float*>(info.ptr),
                                                     static_cast<size_t>(info.size));
        return py::make_tuple(s.mean, s.variance);
      },
      py::arg("mags"),
      "Per-packet inter-subcarrier (mu, sigma2) over subcarrier magnitudes (REFERENCE §0B weapon "
      "discriminator: metal -> lower sigma2). Sample variance (M-1).");

  m.def(
      "inter_carrier_phase_stats",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> phase) {
        py::buffer_info info = phase.request();
        const size_t k = static_cast<size_t>(info.size);
        std::vector<float> scratch(k);
        const InterCarrierPhaseStat s =
            interCarrierPhaseStats(static_cast<const float*>(info.ptr), k, scratch.data());
        return py::make_tuple(s.slope, s.residualStd);
      },
      py::arg("phase"),
      "Per-frame inter-subcarrier phase (slope, residual_std): unwrap across subcarriers, fit the "
      "linear ToF slope, return slope + RMS non-linear residual (coherent metal -> lower residual).");

  m.def(
      "power_spectrum",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> x, size_t nfft) {
        py::buffer_info info = x.request();
        const size_t n = static_cast<size_t>(info.size);
        if (nfft == 0) nfft = nextPow2(std::max<size_t>(n, 64));
        Fft fft(nfft);
        std::vector<std::complex<float>> scratch(nfft);
        py::array_t<float> power(static_cast<py::ssize_t>(nfft / 2 + 1));
        py::buffer_info pi = power.request();
        powerSpectrum(static_cast<const float*>(info.ptr), n, fft, scratch.data(),
                      static_cast<float*>(pi.ptr));
        return power;
      },
      py::arg("x"), py::arg("nfft") = 0,
      "PSD (detrend + Hann + zero-pad to nfft) -> power over nfft/2+1 bins. nfft=0 -> "
      "nextPow2(max(n,64)).");

  m.def(
      "doppler_features",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> x, float fs, float f_hi,
         size_t nfft) {
        py::buffer_info info = x.request();
        const size_t n = static_cast<size_t>(info.size);
        if (nfft == 0) nfft = nextPow2(std::max<size_t>(n, 64));
        Fft fft(nfft);
        std::vector<std::complex<float>> scratch(nfft);
        std::vector<float> power(nfft / 2 + 1);
        const DopplerFeature d = dopplerFeatures(static_cast<const float*>(info.ptr), n, fs, f_hi,
                                                 fft, scratch.data(), power.data());
        return py::make_tuple(d.maxShiftHz, d.spreadHz);
      },
      py::arg("x"), py::arg("fs"), py::arg("f_hi") = 2.0f, py::arg("nfft") = 0,
      "Doppler (max_shift_hz, spread_hz) from the PSD of a series (f_d = 2v/lambda).");

  py::class_<FeatureExtractor>(m, "FeatureExtractor")
      .def(py::init<size_t, size_t, size_t>(), py::arg("num_series"), py::arg("window"),
           py::arg("hop"))
      .def_property_readonly("num_series", &FeatureExtractor::numSeries)
      .def_property_readonly("window", &FeatureExtractor::window)
      .def_property_readonly("hop", &FeatureExtractor::hop)
      .def_property_readonly("output_size", &FeatureExtractor::outputSize)
      .def("reset", &FeatureExtractor::reset)
      .def(
          "push",
          [](FeatureExtractor& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> values) {
            py::buffer_info info = values.request();
            if (static_cast<size_t>(info.size) != self.numSeries())
              throw WaveTraceError("FeatureExtractor.push: values length must equal num_series");
            return self.push(static_cast<const float*>(info.ptr));
          },
          py::arg("values"),
          "Push one frame's num_series values; True when a vector was emitted (see `features`).")
      .def_property_readonly(
          "features",
          [](py::object self) -> py::array {
            FeatureExtractor& f = self.cast<FeatureExtractor&>();
            const auto len = static_cast<py::ssize_t>(f.outputSize());
            const auto elem = static_cast<py::ssize_t>(sizeof(float));
            // Zero-copy float32 view of the reused output (len 9*num_series, same buffer each emit).
            return py::array_t<float>({len}, {elem}, f.data(), self);
          },
          "Zero-copy view of the latest emitted feature vector (length 9*num_series).");

  py::class_<InterCarrierExtractor>(m, "InterCarrierExtractor")
      .def(py::init<size_t, size_t>(), py::arg("window"), py::arg("hop"))
      .def_property_readonly("window", &InterCarrierExtractor::window)
      .def_property_readonly("hop", &InterCarrierExtractor::hop)
      .def_property_readonly("output_size", &InterCarrierExtractor::outputSize)
      .def("reset", &InterCarrierExtractor::reset)
      .def(
          "push",
          [](InterCarrierExtractor& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> mags) {
            py::buffer_info info = mags.request();
            return self.push(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size));
          },
          py::arg("mags"),
          "Push one frame's RAW subcarrier magnitudes (NOT gain-locked); True when a 27-feature "
          "block (mu|sigma2|cv x 9) was emitted (see `features`).")
      .def_property_readonly(
          "features",
          [](py::object self) -> py::array {
            InterCarrierExtractor& f = self.cast<InterCarrierExtractor&>();
            const auto len = static_cast<py::ssize_t>(f.outputSize());
            const auto elem = static_cast<py::ssize_t>(sizeof(float));
            // Zero-copy float32 view of the reused output (length 27, same buffer each emit).
            return py::array_t<float>({len}, {elem}, f.data(), self);
          },
          "Zero-copy view of the latest emitted feature block (length 27 = 3*9: mu|sigma2|cv).");

  py::class_<PresenceSegmenter>(m, "PresenceSegmenter")
      .def(py::init<size_t, float, float, size_t>(), py::arg("window"), py::arg("enter_cv"),
           py::arg("exit_cv"), py::arg("min_active_len") = 1)
      .def_property_readonly("window", &PresenceSegmenter::window)
      .def_property_readonly("active", &PresenceSegmenter::active)
      .def_property_readonly("activity", &PresenceSegmenter::activity)
      .def_property_readonly("segment_closed", &PresenceSegmenter::segmentClosed)
      .def_property_readonly("last_segment_start", &PresenceSegmenter::lastSegmentStart)
      .def_property_readonly("last_segment_end", &PresenceSegmenter::lastSegmentEnd)
      .def_property_readonly("current_start", &PresenceSegmenter::currentStart)
      .def("reset", &PresenceSegmenter::reset)
      .def(
          "push",
          [](PresenceSegmenter& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> mags) {
            py::buffer_info info = mags.request();
            return self.push(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size));
          },
          py::arg("mags"),
          "Push one frame's antenna-collapsed subcarrier magnitudes; True if now inside an active "
          "segment (windowed-CV gate with hysteresis). Check segment_closed for a just-closed [start,end).");

  py::class_<SpectrogramBuilder>(m, "SpectrogramBuilder")
      .def(py::init<size_t, size_t, size_t>(), py::arg("num_subcarriers"), py::arg("time_steps"),
           py::arg("hop"))
      .def_property_readonly("num_subcarriers", &SpectrogramBuilder::numSubcarriers)
      .def_property_readonly("time_steps", &SpectrogramBuilder::timeSteps)
      .def_property_readonly("hop", &SpectrogramBuilder::hop)
      .def("reset", &SpectrogramBuilder::reset)
      .def(
          "push",
          [](SpectrogramBuilder& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> values) {
            py::buffer_info info = values.request();
            if (static_cast<size_t>(info.size) != self.numSubcarriers())
              throw WaveTraceError(
                  "SpectrogramBuilder.push: values length must equal num_subcarriers");
            return self.push(static_cast<const float*>(info.ptr));
          },
          py::arg("values"),
          "Push one frame's num_subcarriers values; True when an image was emitted (see `image`).")
      .def_property_readonly(
          "image",
          [](py::object self) -> py::array {
            SpectrogramBuilder& s = self.cast<SpectrogramBuilder&>();
            const auto rows = static_cast<py::ssize_t>(s.numSubcarriers());
            const auto cols = static_cast<py::ssize_t>(s.timeSteps());
            const auto elem = static_cast<py::ssize_t>(sizeof(float));
            // Zero-copy float32 view of the reused (num_subcarriers x time_steps) image.
            return py::array_t<float>({rows, cols}, {cols * elem, elem}, s.data(), self);
          },
          "Zero-copy view of the latest emitted (num_subcarriers x time_steps) CSI image.");
}
