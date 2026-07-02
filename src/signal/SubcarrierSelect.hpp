#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wavetrace {

// NBVI subcarrier selection. Offline/periodic over a quiet baseline (never per frame, so allocation is fine):
// scores each subcarrier by amplitude variability relative to its level, keeping a spectrally-diverse
// (non-consecutive) subset above a low-amplitude noise gate.

// Per-subcarrier baseline means + noise-gate threshold (percentile of means); fills meansOut. O(F*S + S log S).
inline float noiseGate(const float* amp, size_t numFrames, size_t numSubcarriers,
                       float percentile, std::vector<float>& meansOut) {
  meansOut.assign(numSubcarriers, 0.0f);
  for (size_t s = 0; s < numSubcarriers; ++s) {
    double m = 0.0;
    for (size_t f = 0; f < numFrames; ++f) m += amp[f * numSubcarriers + s];
    meansOut[s] = static_cast<float>(m / static_cast<double>(numFrames));
  }
  std::vector<float> sortedMeans = meansOut;
  std::sort(sortedMeans.begin(), sortedMeans.end());
  size_t gi = static_cast<size_t>(percentile * static_cast<double>(numSubcarriers));
  if (gi >= numSubcarriers) gi = numSubcarriers - 1;
  return sortedMeans[gi];
}

// All subcarriers passing the noise gate, ascending — CNN image rows (contiguous-frequency, unlike the NBVI MLP subset). O(S log S), offline.
inline std::vector<uint16_t> validSubcarriers(const float* amp, size_t numFrames,
                                              size_t numSubcarriers, float noiseGatePercentile) {
  if (numFrames == 0 || numSubcarriers == 0) return {};
  std::vector<float> means;
  const float gate = noiseGate(amp, numFrames, numSubcarriers, noiseGatePercentile, means);
  std::vector<uint16_t> result;
  for (size_t s = 0; s < numSubcarriers; ++s) {
    if (means[s] >= gate) result.push_back(static_cast<uint16_t>(s));
  }
  return result;  // already ascending (iterated s=0..S-1)
}

// Per-subcarrier NBVI over row-major (numFrames x numSubcarriers) amp: alpha*(sigma/mu^2) + (1-alpha)*(sigma/mu).
// Higher = more informative. O(F*S). mu~0 -> score 0.
inline std::vector<float> nbviScores(const float* amp, size_t numFrames, size_t numSubcarriers,
                                     float alpha) {
  std::vector<float> scores(numSubcarriers, 0.0f);
  if (numFrames == 0) return scores;
  for (size_t s = 0; s < numSubcarriers; ++s) {
    double mean = 0.0;
    for (size_t f = 0; f < numFrames; ++f) mean += amp[f * numSubcarriers + s];
    mean /= static_cast<double>(numFrames);
    if (mean < 1e-12) continue;
    double var = 0.0;
    for (size_t f = 0; f < numFrames; ++f) {
      const double d = static_cast<double>(amp[f * numSubcarriers + s]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(numFrames);
    const double sigma = std::sqrt(var);
    scores[s] = static_cast<float>(alpha * (sigma / (mean * mean)) + (1.0 - alpha) * (sigma / mean));
  }
  return scores;
}

struct NbviParams {
  float alpha = 0.75f;
  size_t maxSubcarriers = 12;
  float noiseGatePercentile = 0.15f;  // drop subcarriers below this percentile of mean amplitude
};

// Up to maxSubcarriers non-consecutive subcarriers by NBVI after a noise gate (also removes DC/guard bands,
// geometry-agnostic); greedy by score with index tie-break for a stable result. O(S log S), ascending.
inline std::vector<uint16_t> selectSubcarriersNbvi(const float* amp, size_t numFrames,
                                                   size_t numSubcarriers, const NbviParams& p) {
  if (numFrames == 0 || numSubcarriers == 0) return {};

  std::vector<float> means;
  const float gate = noiseGate(amp, numFrames, numSubcarriers, p.noiseGatePercentile, means);
  const std::vector<float> scores = nbviScores(amp, numFrames, numSubcarriers, p.alpha);

  // Candidates passing the gate, ranked by score desc; stable_sort keeps the lower index on ties -> deterministic.
  std::vector<uint16_t> cand;
  for (size_t s = 0; s < numSubcarriers; ++s) {
    if (means[s] >= gate) cand.push_back(static_cast<uint16_t>(s));
  }
  std::stable_sort(cand.begin(), cand.end(),
                   [&](uint16_t a, uint16_t b) { return scores[a] > scores[b]; });

  // Greedy non-consecutive pick: taking a subcarrier blocks its two neighbours (spectral diversity).
  std::vector<uint8_t> blocked(numSubcarriers, 0);
  std::vector<uint16_t> selected;
  for (uint16_t idx : cand) {
    if (blocked[idx]) continue;
    selected.push_back(idx);
    if (idx > 0) blocked[idx - 1] = 1;
    blocked[idx] = 1;
    if (static_cast<size_t>(idx) + 1 < numSubcarriers) blocked[idx + 1] = 1;
    if (selected.size() >= p.maxSubcarriers) break;
  }
  std::sort(selected.begin(), selected.end());
  return selected;
}

}  // namespace wavetrace
