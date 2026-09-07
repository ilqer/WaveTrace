#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wavetrace {

// NBVI subcarrier selection (offline/periodic over a quiet baseline, not per-frame): scores each
// subcarrier by amplitude variability, keeps a spectrally-diverse subset above a noise gate.

// Per-subcarrier baseline means + noise-gate threshold (percentile of means); fills meansOut. O(F*S + S log S).
inline float NoiseGate(const float* amplitudes, size_t frameCount, size_t subcarrierCount,
                       float percentile, std::vector<float>& meansOut) {
  meansOut.assign(subcarrierCount, 0.0f);
  for (size_t s = 0; s < subcarrierCount; ++s) {
    double sum = 0.0;
    for (size_t f = 0; f < frameCount; ++f) sum += amplitudes[f * subcarrierCount + s];
    meansOut[s] = static_cast<float>(sum / static_cast<double>(frameCount));
  }
  std::vector<float> sortedMeans = meansOut;
  std::sort(sortedMeans.begin(), sortedMeans.end());
  size_t gateIndex = static_cast<size_t>(percentile * static_cast<double>(subcarrierCount));
  if (gateIndex >= subcarrierCount) gateIndex = subcarrierCount - 1;
  return sortedMeans[gateIndex];
}

// All subcarriers passing the noise gate, ascending — CNN image rows (contiguous-frequency, unlike the NBVI MLP subset). O(S log S), offline.
inline std::vector<uint16_t> ValidSubcarriers(const float* amplitudes, size_t frameCount,
                                              size_t subcarrierCount, float noiseGatePercentile) {
  if (frameCount == 0 || subcarrierCount == 0) return {};
  std::vector<float> means;
  const float gate = NoiseGate(amplitudes, frameCount, subcarrierCount, noiseGatePercentile, means);
  std::vector<uint16_t> result;
  for (size_t s = 0; s < subcarrierCount; ++s) {
    if (means[s] >= gate) result.push_back(static_cast<uint16_t>(s));
  }
  return result;  // already ascending (iterated s=0..S-1)
}

// Per-subcarrier NBVI over row-major (frameCount x subcarrierCount) amplitudes: alpha*(sigma/mu^2) + (1-alpha)*(sigma/mu).
// Higher = more informative. O(F*S). mu~0 -> score 0.
inline std::vector<float> NbviScores(const float* amplitudes, size_t frameCount, size_t subcarrierCount,
                                     float alpha) {
  std::vector<float> scores(subcarrierCount, 0.0f);
  if (frameCount == 0) return scores;
  for (size_t s = 0; s < subcarrierCount; ++s) {
    double mean = 0.0;
    for (size_t f = 0; f < frameCount; ++f) mean += amplitudes[f * subcarrierCount + s];
    mean /= static_cast<double>(frameCount);
    if (mean < 1e-12) continue;
    double variance = 0.0;
    for (size_t f = 0; f < frameCount; ++f) {
      const double deviationFromMean = static_cast<double>(amplitudes[f * subcarrierCount + s]) - mean;
      variance += deviationFromMean * deviationFromMean;
    }
    variance /= static_cast<double>(frameCount);
    const double sigma = std::sqrt(variance);
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
inline std::vector<uint16_t> SelectSubcarriersNbvi(const float* amplitudes, size_t frameCount,
                                                   size_t subcarrierCount, const NbviParams& params) {
  if (frameCount == 0 || subcarrierCount == 0) return {};

  std::vector<float> means;
  const float gate = NoiseGate(amplitudes, frameCount, subcarrierCount, params.noiseGatePercentile, means);
  const std::vector<float> scores = NbviScores(amplitudes, frameCount, subcarrierCount, params.alpha);

  // Candidates passing the gate, ranked by score desc; stable_sort keeps the lower index on ties -> deterministic.
  std::vector<uint16_t> candidates;
  for (size_t s = 0; s < subcarrierCount; ++s) {
    if (means[s] >= gate) candidates.push_back(static_cast<uint16_t>(s));
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [&](uint16_t a, uint16_t b) { return scores[a] > scores[b]; });

  // Greedy non-consecutive pick: taking a subcarrier blocks its two neighbours (spectral diversity).
  std::vector<uint8_t> blocked(subcarrierCount, 0);
  std::vector<uint16_t> selected;
  for (uint16_t candidateIndex : candidates) {
    if (blocked[candidateIndex]) continue;
    selected.push_back(candidateIndex);
    if (candidateIndex > 0) blocked[candidateIndex - 1] = 1;
    blocked[candidateIndex] = 1;
    if (static_cast<size_t>(candidateIndex) + 1 < subcarrierCount) blocked[candidateIndex + 1] = 1;
    if (selected.size() >= params.maxSubcarriers) break;
  }
  std::sort(selected.begin(), selected.end());
  return selected;
}

}  // namespace wavetrace
