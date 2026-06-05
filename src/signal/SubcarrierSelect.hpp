#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wavetrace {

// NBVI subcarrier selection (REFERENCE_DIGEST §2.10). OFFLINE / periodic — runs over a quiet
// baseline, never per frame, so allocation here is fine. Not all subcarriers help (guard bands, DC,
// low-SNR); NBVI scores each by how much its amplitude varies relative to its level, then we keep a
// spectrally-diverse (non-consecutive) subset above a low-amplitude noise gate.

// Per-subcarrier NBVI over a baseline amplitude matrix (row-major, numFrames x numSubcarriers):
//   NBVI = alpha*(sigma/mu^2) + (1-alpha)*(sigma/mu)
// Higher = more informative. Two-pass sigma (stable, REFERENCE §2.8). O(F*S). mu~0 -> score 0.
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

// Select up to maxSubcarriers NON-CONSECUTIVE subcarriers by NBVI, after a low-amplitude noise gate
// (the gate also removes DC/guard bands without hardcoding HT20 indices — geometry-agnostic).
// Greedy by score with index tie-break, so the same baseline always yields the same set (stable).
// O(S log S). Returns indices sorted ascending.
inline std::vector<uint16_t> selectSubcarriersNbvi(const float* amp, size_t numFrames,
                                                   size_t numSubcarriers, const NbviParams& p) {
  if (numFrames == 0 || numSubcarriers == 0) return {};

  std::vector<float> means(numSubcarriers, 0.0f);
  for (size_t s = 0; s < numSubcarriers; ++s) {
    double m = 0.0;
    for (size_t f = 0; f < numFrames; ++f) m += amp[f * numSubcarriers + s];
    means[s] = static_cast<float>(m / static_cast<double>(numFrames));
  }
  const std::vector<float> scores = nbviScores(amp, numFrames, numSubcarriers, p.alpha);

  // Noise gate = percentile of the per-subcarrier mean amplitudes.
  std::vector<float> sortedMeans = means;
  std::sort(sortedMeans.begin(), sortedMeans.end());
  size_t gi = static_cast<size_t>(p.noiseGatePercentile * static_cast<double>(numSubcarriers));
  if (gi >= numSubcarriers) gi = numSubcarriers - 1;
  const float gate = sortedMeans[gi];

  // Candidates passing the gate, ranked by score desc (stable_sort over ascending indices keeps the
  // lower index on ties -> deterministic).
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
