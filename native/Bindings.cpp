// pybind11 module `_wavetrace`: exposes the core C++ types and hot-path DSP to Python orchestration.
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

// Zero-copy writable (antennaCount x subcarrierCount) complex64 view sharing the frame's buffer; `frame` is the base object so the buffer outlives the array.
static py::array GridView(py::object frame) {
  CsiFrame& csiFrame = frame.cast<CsiFrame&>();
  const auto rows = static_cast<py::ssize_t>(csiFrame.AntennaCount());
  const auto columns = static_cast<py::ssize_t>(csiFrame.SubcarrierCount());
  const auto elementSizeBytes = static_cast<py::ssize_t>(sizeof(CsiFrame::Sample));
  return py::array_t<std::complex<float>>(
      {rows, columns},                 // shape
      {columns * elementSizeBytes, elementSizeBytes},          // row-major strides
      csiFrame.Data(),              // shared buffer
      frame);                       // base keepalive
}

PYBIND11_MODULE(_wavetrace, m) {
  m.doc() = "WaveTrace native core (Phase 1: shared types).";

  py::register_exception<WaveTraceError>(m, "WaveTraceError");
  py::register_exception<FrameError>(m, "FrameError");

  py::class_<CsiFrame>(m, "CsiFrame")
      .def(py::init<uint16_t, uint16_t>(), py::arg("num_antennas"), py::arg("num_subcarriers"))
      .def_property_readonly("num_antennas", &CsiFrame::AntennaCount)
      .def_property_readonly("num_subcarriers", &CsiFrame::SubcarrierCount)
      .def_property_readonly("size", &CsiFrame::Size)
      .def_property("timestamp", &CsiFrame::TimestampSeconds, &CsiFrame::SetTimestamp)
      .def_property("node_id", &CsiFrame::NodeId, &CsiFrame::SetNodeId)
      .def("reshape", &CsiFrame::Reshape, py::arg("num_antennas"), py::arg("num_subcarriers"))
      .def_property_readonly("grid", &GridView,
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
      .def_readwrite("keypoints", &Label::keypoints)
      .def_readwrite("mask", &Label::mask)
      .def_readwrite("mask_grid", &Label::maskGrid);

  // Phase 2 — hardware ingest.
  py::class_<FrameParser>(m, "FrameParser")
      .def(py::init<uint16_t, uint16_t>(), py::arg("num_antennas"), py::arg("num_subcarriers"))
      .def_property_readonly("num_antennas", &FrameParser::AntennaCount)
      .def_property_readonly("num_subcarriers", &FrameParser::SubcarrierCount)
      .def(
          "parse",
          [](FrameParser& self,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rawIqBytes,
             double timestampSeconds, int32_t nodeId) -> const CsiFrame& {
            py::buffer_info info = rawIqBytes.request();
            return self.Parse(static_cast<const uint8_t*>(info.ptr),
                              static_cast<size_t>(info.size), timestampSeconds, nodeId);
          },
          py::arg("raw"), py::arg("timestamp") = 0.0, py::arg("node_id") = -1,
          // Ties the reused CsiFrame's lifetime to the parser and keeps `raw` alive for the decode.
          py::return_value_policy::reference_internal, py::keep_alive<0, 2>(),
          "Decode one raw int8 [imag,real] frame into the reused CsiFrame (returned). O(n).");

  py::class_<NodeAggregator>(m, "NodeAggregator")
      .def(py::init<>())
      .def("submit", &NodeAggregator::Submit, py::arg("frame"))
      .def_property_readonly("num_nodes", &NodeAggregator::NumNodes)
      .def("synced", &NodeAggregator::CollectFramesWithin, py::arg("tolerance"),
           "Latest frame per node within `tolerance` s of the newest submit (copies). O(m).");

  // Signal preprocessing: stateless transforms first (bound for unit tests), then the streaming Preprocessor.
  m.def("conjugate_multiply", &ConjugateMultiply, py::arg("in_frame"), py::arg("out_frame"),
        "Geometry-adaptive conjugate multiply (cancels CFO/SFO) into out_frame (reshaped). O(n).");
  m.def("combined_channel_difference", &CombinedChannelDifference, py::arg("in_frame"),
        py::arg("out_frame"),
        "Antenna-difference combined channel H[a]-H[0] into out_frame (nulls the common environment, "
        "amplifies material scattering; in-baggage Eq.3). REQUIRES >=2 antennas on one radio. O(n).");
  m.def(
      "hampel",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> window, float current,
         float thresholdK) {
        py::buffer_info info = window.request();
        std::vector<float> scratch(static_cast<size_t>(info.size));
        return Hampel(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size), current,
                      scratch.data(), thresholdK);
      },
      py::arg("window"), py::arg("current"), py::arg("k") = 5.0f,
      "Hampel test: median if `current` is an outlier vs `window`, else `current`.");
  m.def("unwrap_step", &UnwrapStep, py::arg("cur_wrapped"), py::arg("prev_wrapped"),
        py::arg("prev_unwrapped"), "One streaming phase-unwrap step. O(1).");

  m.def(
      "coefficient_of_variation",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> amplitudes) {
        py::buffer_info info = amplitudes.request();
        return CoefficientOfVariation(static_cast<const float*>(info.ptr),
                                      static_cast<size_t>(info.size));
      },
      py::arg("amplitudes"), "Gain-invariant variability sigma/mu. O(n).");

  py::class_<GainLock>(m, "GainLock")
      .def(py::init<size_t>(), py::arg("baseline_packets") = 300)
      .def("observe", &GainLock::Observe, py::arg("frame"))
      .def("finalize", &GainLock::Finalize)
      .def("lock_to", &GainLock::LockTo, py::arg("scale"))
      .def("apply", &GainLock::Apply, py::arg("frame"))
      .def_property_readonly("observed", &GainLock::ObservedCount)
      .def_property_readonly("ready", &GainLock::IsReady)
      .def_property_readonly("locked", &GainLock::IsLocked)
      .def_property_readonly("reference_scale", &GainLock::ReferenceScale);

  // NBVI subcarrier selection (offline). amp = float32 (frames x subcarriers) baseline matrix.
  m.def(
      "nbvi_scores",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> amplitudes, float alpha) {
        py::buffer_info info = amplitudes.request();
        if (info.ndim != 2) throw WaveTraceError("nbvi_scores: amp must be 2D (frames x subcarriers)");
        return NbviScores(static_cast<const float*>(info.ptr), static_cast<size_t>(info.shape[0]),
                          static_cast<size_t>(info.shape[1]), alpha);
      },
      py::arg("amp"), py::arg("alpha") = 0.75f, "Per-subcarrier NBVI over a baseline matrix.");
  m.def(
      "valid_subcarriers",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> amplitudes,
         float noiseGatePercentile) {
        py::buffer_info info = amplitudes.request();
        if (info.ndim != 2)
          throw WaveTraceError("valid_subcarriers: amp must be 2D (frames x subcarriers)");
        return ValidSubcarriers(static_cast<const float*>(info.ptr),
                                static_cast<size_t>(info.shape[0]),
                                static_cast<size_t>(info.shape[1]), noiseGatePercentile);
      },
      py::arg("amp"), py::arg("noise_gate_percentile") = 0.15f,
      "ALL subcarriers passing the noise gate, sorted ascending (freq order) — CNN image rows.");
  m.def(
      "select_subcarriers_nbvi",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> amplitudes, float alpha,
         size_t maxSubcarriers, float noiseGatePercentile) {
        py::buffer_info info = amplitudes.request();
        if (info.ndim != 2)
          throw WaveTraceError("select_subcarriers_nbvi: amp must be 2D (frames x subcarriers)");
        NbviParams params;
        params.alpha = alpha;
        params.maxSubcarriers = maxSubcarriers;
        params.noiseGatePercentile = noiseGatePercentile;
        return SelectSubcarriersNbvi(static_cast<const float*>(info.ptr),
                                     static_cast<size_t>(info.shape[0]),
                                     static_cast<size_t>(info.shape[1]), params);
      },
      py::arg("amp"), py::arg("alpha") = 0.75f, py::arg("max_subcarriers") = 12,
      py::arg("noise_gate_percentile") = 0.15f,
      "Up to max_subcarriers non-consecutive informative subcarriers (indices). Offline.");

  py::class_<Preprocessor>(m, "Preprocessor")
      .def(py::init<uint16_t, uint16_t, size_t, float, float>(), py::arg("num_antennas"),
           py::arg("num_subcarriers"), py::arg("hampel_window") = 7, py::arg("hampel_k") = 5.0f,
           py::arg("normalize_alpha") = 0.1f)
      .def_property_readonly("out_rows", &Preprocessor::OutRows)
      .def_property_readonly("out_cols", &Preprocessor::OutCols)
      .def("reset", &Preprocessor::Reset)
      .def(
          "process",
          [](py::object self, const CsiFrame& in) -> py::array {
            Preprocessor& p = self.cast<Preprocessor&>();
            p.Process(in);
            const auto rows = static_cast<py::ssize_t>(p.OutRows());
            const auto columns = static_cast<py::ssize_t>(p.OutCols());
            const auto elementSizeBytes = static_cast<py::ssize_t>(sizeof(float));
            // Zero-copy float32 view of the reused output grid (same buffer each call).
            return py::array_t<float>({rows, columns}, {columns * elementSizeBytes, elementSizeBytes}, p.Data(), self);
          },
          py::arg("frame"),
          "Process one frame -> drift-free differential-phase grid (zero-copy view). O(n).");

  // Features + spectrogram: stateless FFT/feature fns (bound for unit tests), then the streaming extractors.
  m.def(
      "fft",
      [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> values) {
        py::buffer_info info = values.request();
        const size_t sampleCount = static_cast<size_t>(info.size);
        Fft fft(sampleCount);  // throws if sampleCount is not a power of two
        py::array_t<std::complex<float>> out(static_cast<py::ssize_t>(sampleCount));
        py::buffer_info oi = out.request();
        std::copy_n(static_cast<const std::complex<float>*>(info.ptr), sampleCount,
                    static_cast<std::complex<float>*>(oi.ptr));
        fft.Forward(static_cast<std::complex<float>*>(oi.ptr));
        return out;
      },
      py::arg("x"), "Radix-2 forward FFT of a power-of-two complex64 array. O(n log n).");

  m.def(
      "nine_features",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> window) {
        py::buffer_info info = window.request();
        const size_t sampleCount = static_cast<size_t>(info.size);
        std::vector<float> scratch(sampleCount);
        std::vector<float> out(9);
        NineFeatures(static_cast<const float*>(info.ptr), sampleCount, scratch.data(), out.data());
        return out;
      },
      py::arg("window"),
      "REFERENCE §2.9 nine features [mean,std,max,min,IQR,skew,lag1,MAD,WL] over one window.");

  m.def(
      "inter_carrier_stats",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> magnitudes) {
        py::buffer_info info = magnitudes.request();
        const InterCarrierStat s = ComputeInterCarrierStats(static_cast<const float*>(info.ptr),
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
        const size_t subcarrierCount = static_cast<size_t>(info.size);
        std::vector<float> scratch(subcarrierCount);
        const InterCarrierPhaseStat s =
            ComputeInterCarrierPhaseStats(static_cast<const float*>(info.ptr), subcarrierCount, scratch.data());
        return py::make_tuple(s.slope, s.residualStd);
      },
      py::arg("phase"),
      "Per-frame inter-subcarrier phase (slope, residual_std): unwrap across subcarriers, fit the "
      "linear ToF slope, return slope + RMS non-linear residual (coherent metal -> lower residual).");

  m.def(
      "reconstruct_complex_csi",
      [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> csi) {
        py::buffer_info info = csi.request();
        const size_t subcarrierCount = static_cast<size_t>(info.size);
        py::array_t<std::complex<float>> out(static_cast<py::ssize_t>(subcarrierCount));
        py::buffer_info oi = out.request();
        std::vector<float> scratch(subcarrierCount);
        ReconstructComplexCsi(static_cast<const std::complex<float>*>(info.ptr), subcarrierCount,
                              static_cast<std::complex<float>*>(oi.ptr), scratch.data());
        return out;
      },
      py::arg("csi"),
      "Sanitized complex CSI for one frame: remove the linear STO/CFO phase ramp across subcarriers "
      "and keep the ABSOLUTE residual phase + magnitude (in-baggage material discriminator). O(k).");

  m.def(
      "reflection_null",
      [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> h1,
         py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> h2,
         py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> baselineH1,
         py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> baselineH2) {
        py::buffer_info i1 = h1.request(), i2 = h2.request();
        py::buffer_info b1 = baselineH1.request(), b2 = baselineH2.request();
        const size_t subcarrierCount = static_cast<size_t>(i1.size);
        if (static_cast<size_t>(i2.size) != subcarrierCount || static_cast<size_t>(b1.size) != subcarrierCount ||
            static_cast<size_t>(b2.size) != subcarrierCount)
          throw WaveTraceError("reflection_null: all four arrays must have equal length");
        py::array_t<std::complex<float>> out(static_cast<py::ssize_t>(subcarrierCount));
        py::buffer_info oi = out.request();
        ComputeReflectionNull(static_cast<const std::complex<float>*>(i1.ptr),
                       static_cast<const std::complex<float>*>(i2.ptr),
                       static_cast<const std::complex<float>*>(b1.ptr),
                       static_cast<const std::complex<float>*>(b2.ptr), subcarrierCount,
                       static_cast<std::complex<float>*>(oi.ptr));
        return out;
      },
      py::arg("h1"), py::arg("h2"), py::arg("baseline_h1"), py::arg("baseline_h2"),
      "Beta-null reflection isolation: out = h1 + beta*h2, beta=-baseline_h1/baseline_h2 (nulls the "
      "empty-room LOS+static, leaving the object's pure reflection; in-baggage Eq.4). REQUIRES 2 paths.");

  m.def(
      "block_average_decimate",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> values, size_t factor) {
        py::buffer_info info = values.request();
        const size_t sampleCount = static_cast<size_t>(info.size);
        const size_t outputCount = (factor > 0) ? sampleCount / factor : 0;
        py::array_t<float> out(static_cast<py::ssize_t>(outputCount));
        py::buffer_info oi = out.request();
        BlockAverageDecimate(static_cast<const float*>(info.ptr), sampleCount, factor,
                             static_cast<float*>(oi.ptr));
        return out;
      },
      py::arg("x"), py::arg("factor"),
      "Non-overlapping block-average decimation (LUMS): mean every `factor` samples -> n/factor. O(n).");

  m.def(
      "power_spectrum",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> values, size_t nfft) {
        py::buffer_info info = values.request();
        const size_t sampleCount = static_cast<size_t>(info.size);
        if (nfft == 0) nfft = NextPow2(std::max<size_t>(sampleCount, 64));
        Fft fft(nfft);
        std::vector<std::complex<float>> scratch(nfft);
        py::array_t<float> power(static_cast<py::ssize_t>(nfft / 2 + 1));
        py::buffer_info pi = power.request();
        ComputePowerSpectrum(static_cast<const float*>(info.ptr), sampleCount, fft, scratch.data(),
                      static_cast<float*>(pi.ptr));
        return power;
      },
      py::arg("x"), py::arg("nfft") = 0,
      "PSD (detrend + Hann + zero-pad to nfft) -> power over nfft/2+1 bins. nfft=0 -> "
      "nextPow2(max(n,64)).");

  m.def(
      "doppler_features",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> values, float sampleRateHz,
         float highCutoffHz, size_t nfft) {
        py::buffer_info info = values.request();
        const size_t sampleCount = static_cast<size_t>(info.size);
        if (nfft == 0) nfft = NextPow2(std::max<size_t>(sampleCount, 64));
        Fft fft(nfft);
        std::vector<std::complex<float>> scratch(nfft);
        std::vector<float> power(nfft / 2 + 1);
        const DopplerFeature d = ComputeDopplerFeatures(static_cast<const float*>(info.ptr), sampleCount,
                                                 sampleRateHz, highCutoffHz,
                                                 fft, scratch.data(), power.data());
        return py::make_tuple(d.maxShiftHz, d.spreadHz);
      },
      py::arg("x"), py::arg("fs"), py::arg("f_hi") = 2.0f, py::arg("nfft") = 0,
      "Doppler (max_shift_hz, spread_hz) from the PSD of a series (f_d = 2v/lambda).");

  py::class_<FeatureExtractor>(m, "FeatureExtractor")
      .def(py::init<size_t, size_t, size_t>(), py::arg("num_series"), py::arg("window"),
           py::arg("hop"))
      .def_property_readonly("num_series", &FeatureExtractor::SeriesCount)
      .def_property_readonly("window", &FeatureExtractor::Window)
      .def_property_readonly("hop", &FeatureExtractor::Hop)
      .def_property_readonly("output_size", &FeatureExtractor::OutputSize)
      .def("reset", &FeatureExtractor::Reset)
      .def(
          "push",
          [](FeatureExtractor& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> values) {
            py::buffer_info info = values.request();
            if (static_cast<size_t>(info.size) != self.SeriesCount())
              throw WaveTraceError("FeatureExtractor.push: values length must equal num_series");
            return self.Push(static_cast<const float*>(info.ptr));
          },
          py::arg("values"),
          "Push one frame's num_series values; True when a vector was emitted (see `features`).")
      .def_property_readonly(
          "features",
          [](py::object self) -> py::array {
            FeatureExtractor& f = self.cast<FeatureExtractor&>();
            const auto outputLength = static_cast<py::ssize_t>(f.OutputSize());
            const auto elementSizeBytes = static_cast<py::ssize_t>(sizeof(float));
            // Zero-copy float32 view of the reused output (length 9*num_series, same buffer each emit).
            return py::array_t<float>({outputLength}, {elementSizeBytes}, f.Data(), self);
          },
          "Zero-copy view of the latest emitted feature vector (length 9*num_series).");

  py::class_<InterCarrierExtractor>(m, "InterCarrierExtractor")
      .def(py::init<size_t, size_t>(), py::arg("window"), py::arg("hop"))
      .def_property_readonly("window", &InterCarrierExtractor::Window)
      .def_property_readonly("hop", &InterCarrierExtractor::Hop)
      .def_property_readonly("output_size", &InterCarrierExtractor::OutputSize)
      .def("reset", &InterCarrierExtractor::Reset)
      .def(
          "push",
          [](InterCarrierExtractor& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> magnitudes) {
            py::buffer_info info = magnitudes.request();
            return self.Push(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size));
          },
          py::arg("mags"),
          "Push one frame's RAW subcarrier magnitudes (NOT gain-locked); True when a 27-feature "
          "block (mu|sigma2|cv x 9) was emitted (see `features`).")
      .def_property_readonly(
          "features",
          [](py::object self) -> py::array {
            InterCarrierExtractor& f = self.cast<InterCarrierExtractor&>();
            const auto outputLength = static_cast<py::ssize_t>(f.OutputSize());
            const auto elementSizeBytes = static_cast<py::ssize_t>(sizeof(float));
            // Zero-copy float32 view of the reused output (length 27, same buffer each emit).
            return py::array_t<float>({outputLength}, {elementSizeBytes}, f.Data(), self);
          },
          "Zero-copy view of the latest emitted feature block (length 27 = 3*9: mu|sigma2|cv).");

  py::class_<PresenceSegmenter>(m, "PresenceSegmenter")
      .def(py::init<size_t, float, float, size_t>(), py::arg("window"), py::arg("enter_cv"),
           py::arg("exit_cv"), py::arg("min_active_len") = 1)
      .def_property_readonly("window", &PresenceSegmenter::Window)
      .def_property_readonly("active", &PresenceSegmenter::IsActive)
      .def_property_readonly("activity", &PresenceSegmenter::Activity)
      .def_property_readonly("segment_closed", &PresenceSegmenter::HasSegmentClosed)
      .def_property_readonly("last_segment_start", &PresenceSegmenter::LastSegmentStart)
      .def_property_readonly("last_segment_end", &PresenceSegmenter::LastSegmentEnd)
      .def_property_readonly("current_start", &PresenceSegmenter::CurrentStart)
      .def("reset", &PresenceSegmenter::Reset)
      .def(
          "push",
          [](PresenceSegmenter& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> magnitudes) {
            py::buffer_info info = magnitudes.request();
            return self.Push(static_cast<const float*>(info.ptr), static_cast<size_t>(info.size));
          },
          py::arg("mags"),
          "Push one frame's antenna-collapsed subcarrier magnitudes; True if now inside an active "
          "segment (windowed-CV gate with hysteresis). Check segment_closed for a just-closed [start,end).");

  py::class_<SpectrogramBuilder>(m, "SpectrogramBuilder")
      .def(py::init<size_t, size_t, size_t>(), py::arg("num_subcarriers"), py::arg("time_steps"),
           py::arg("hop"))
      .def_property_readonly("num_subcarriers", &SpectrogramBuilder::SubcarrierCount)
      .def_property_readonly("time_steps", &SpectrogramBuilder::TimeSteps)
      .def_property_readonly("hop", &SpectrogramBuilder::Hop)
      .def("reset", &SpectrogramBuilder::Reset)
      .def(
          "push",
          [](SpectrogramBuilder& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> values) {
            py::buffer_info info = values.request();
            if (static_cast<size_t>(info.size) != self.SubcarrierCount())
              throw WaveTraceError(
                  "SpectrogramBuilder.push: values length must equal num_subcarriers");
            return self.Push(static_cast<const float*>(info.ptr));
          },
          py::arg("values"),
          "Push one frame's num_subcarriers values; True when an image was emitted (see `image`).")
      .def_property_readonly(
          "image",
          [](py::object self) -> py::array {
            SpectrogramBuilder& s = self.cast<SpectrogramBuilder&>();
            const auto rows = static_cast<py::ssize_t>(s.SubcarrierCount());
            const auto columns = static_cast<py::ssize_t>(s.TimeSteps());
            const auto elementSizeBytes = static_cast<py::ssize_t>(sizeof(float));
            // Zero-copy float32 view of the reused (num_subcarriers x time_steps) image.
            return py::array_t<float>({rows, columns}, {columns * elementSizeBytes, elementSizeBytes}, s.Data(), self);
          },
          "Zero-copy view of the latest emitted (num_subcarriers x time_steps) CSI image.");
}
