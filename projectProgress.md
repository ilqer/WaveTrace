# WaveTrace — Project Progress Report

Last updated: 2026-06-30 | Tests passing: 295

---

## 1. Executive Summary

WaveTrace is a WiFi CSI (channel state information) based sensing system built on ESP32 hardware. The goal is to detect concealed weapons on a person walking through a checkpoint — using WiFi signals the way a metal detector uses a magnetic field. The project has two independent operating modes: presence detection and weapon detection. As of June 2026, presence detection works on real hardware at LOGO 0.985. Weapon detection has shown no above-chance body-worn signal at 2.4 GHz in any dataset tested so far. The analysis points to geometry (all captures were line-of-sight; the only published gun detection result used strict non-LOS with a directional antenna) and data scale (1 subject, back-to-back sessions) as the main gaps. Antennas are physically aimed at the center zone now. The immediate next step is clearing the per-link litmus gate (AUC >= 0.65 on at least one link) before any further ML training.

---

## 2. Current Status

### 2.1 Phase Completion

| Phase | Description | Status |
|---|---|---|
| 0 | Firmware: 6-node ESP-NOW token ring, UDP streaming | Done |
| 1–4 | DSP pipeline: calibration, preprocessing, feature extraction, NBVI | Done (synthetic validated) |
| 5 | Real hardware validation: first live presence LOGO 0.985 | Done |
| 6 | Presence head: MLP/SVM backends, LOGO evaluation, baselines | Done |
| 7 | Weapon head architecture: ic27 features, WeaponHead, SegmentVoter | Done |
| 8 | CLI (capture / calibrate / collect-data / train / run), pipeline parity, JSONL publisher | Done |
| 8b | CIR super-resolution module (ISTA/L1 delay-domain) | Done |
| — | NLOS geometry: antennas aimed center, litmus gate | In progress |
| — | Walking weapon collection (metal_walk dataset) | In progress |
| — | People count: 4-class LOGO ~0.58 | In progress |
| — | Pi 5 / 5 GHz HT80 node (Nexmon) | Designed only — not set up |
| — | Camera-supervised heatmap (YOLO occupancy) | Designed only — not tested |

### 2.2 Key Results at a Glance

**Presence detection:**

| Evaluation | Score | Baselines |
|---|---|---|
| Synthetic LOGO (session-out) | 0.980 | Majority 0.51 / DSP gate 0.861 / DSP best threshold 0.963 |
| Synthetic LOGO (subject-out) | 0.975 | — |
| Real hardware LOGO | 0.985 | Majority 0.530 |

**Weapon detection — body-worn:**

| Dataset | Sessions | ic27 AUC | CNN AUC | Verdict |
|---|---|---|---|---|
| ilker_hand HT20 (Jun-18 cal) | 3 | 0.455 inverted | 0.372 | No signal |
| p0_na body-worn HT40 (Jun-23 cal) | 3 | 0.290 inverted | — | No signal |
| ilker_metal_walk HT40/ui (Jun-25 cal) | 4 | 0.300 inverted | 0.215 | No signal |

No above-chance body-worn weapon signal found in any dataset.

**Weapon detection — desk-based (no person):**
Node 2 LOGO 0.782, Node 3 LOGO 0.800. Valid result but a different physical problem: static metal object in an empty room.

**People count (4-class: 0, 1, 2, 3+):**

| Run | Sessions | LOGO range | Notes |
|---|---|---|---|
| Run 1 | 3 | 0.464–0.573 | Majority 0.241–0.253 |
| Run 2 | 6 | 0.528–0.611 | Generalization gap narrowing |

---

## 3. Locked Decisions

These are finalized. Do not revisit without strong new evidence.

1. **Two independent modes** — Presence and weapon are separate operating modes with no gate between them. The A→E gating was removed on 2026-06-11 and is irreversible.
2. **No cross-node phase multiply** — Independent ESP32 clocks make phase differences between nodes clock noise, not signal. Each node is processed independently; feature vectors are concatenated at the feature level.
3. **Per-link over per-node for weapon** — Pooling all TX directions into one RX node averages a good NLOS direction with noise links. This causes sign-flip and washout. Per-link models are the correct unit.
4. **LOGO evaluation only** — Within-session train/test splits inflate accuracy by 10–30% because of window overlap. Leave-One-Group-Out over sessions (and subjects when available) is the only honest evaluation.
5. **ic27 features must use raw (pre-gain-lock) magnitudes** — Gain normalization destroys amplitude flatness, which is the metal signature. ic27 features are computed before GainLock is applied; 9K presence features run after.
6. **Dedicated ESP-NOW controlled link** — Production WiFi has bursty traffic and variable MCS/AGC. This destroys uniform sampling (FFT frequency axis) and the amplitude reference (gain-lock). The sensing link must be separate.
7. **Token-ring TDMA firmware** — One node transmits per time slot, token is passed to the next. Not full-duplex, not shared-channel simultaneous.
8. **Feature-level concat for multi-node fusion** — Each node is processed independently; feature vectors are concatenated. Cross-node phase math is forbidden.
9. **ISTA over OMP for CIR** — OMP commits permanently on each iteration. When body and object fall in the same Nyquist bin, OMP picks wrong early and cannot recover. ISTA's soft thresholding degrades gracefully.
10. **Static TX buffers in firmware** — Dynamic TX buffers (sdkconfig default) compete with the CSI queue for DRAM. With a 128-entry HT40 queue (~50 KB), dynamic buffers cause ENOMEM under load. Static buffers (TX_BUFFER_TYPE=0) are pre-allocated and cannot be starved.
11. **AoA / `Localize.py` parked** — Analytic angle-of-arrival needs ≥ 2 phase-coherent receive chains. Both candidate radios are 1×1 (ESP32-S3 single RX chain; Pi CYW43455 single stream), so AoA is not achievable on current hardware. `Localize.py` stays in-tree but unused.
12. **No usable body-worn weapon signal at 2.4 GHz, omni antennas, LOS** — Three real datasets across three environments give ic27 AUC 0.29–0.46 (inverted or near-chance). The ceiling is physics — omnidirectional antennas in line-of-sight geometry — not the model. Above-chance weapon detection requires NLOS geometry and ideally a directional TX (Yousaf 2025 used a >25 dBi dish). Do not interpret a weak model as a tuning problem.

---

## 4. Theoretical Foundations

### 4.1 Signal Processing

**Conjugate multiplication (phase noise removal):**
- Hardware adds a random constant phase offset θ to every sample: Ĥ = H·e^(jθ)
- Multiplying Ĥ_k1 · Ĥ*_k2 cancels e^(jθ)·e^(-jθ)=1, leaving H_k1·H*_k2 with phase = φ_k1 − φ_k2
- This is a prerequisite for any phase-based feature. It applies to both multi-antenna (spatial) and cross-subcarrier (spectral) cases.

**FFT:**
- X[k] = Σ x[n]·e^(-j2πkn/N); frequency resolution = f_s/N
- Radix-2 Cooley-Tukey: O(N log N) vs O(N²) naive; roughly 100x speedup at N=1024
- Non-integer frequencies cause spectral leakage, mitigated by windowing or larger N

**Bandpass filter (IIR):**
- y[n] = b₀x[n] + b₁x[n-1] + b₂x[n-2] − a₁y[n-1] − a₂y[n-2]
- Weights computed via bilinear transform; target band for breathing: 0.1–0.5 Hz

**Fresnel zones:**
- r_n(x) = sqrt(n·λ·x·(d-x)/d); λ=0.125m at 2.4 GHz
- 5m node separation → r_max1 ≈ 0.395m; a human torso (~35 cm) fills ~87% of Zone 1
- A small object (1 cm) fills about 1/40 of the zone — near-undetectable without directional geometry
- A static wall produces constant attenuation, not a detectable change

**Doppler effect:**
- f_D = (2v·cosθ/c)·f_c; the factor of 2 comes from the round-trip path change
- Breathing at v=5mm/s, 2.4 GHz → f_D=0.08 Hz; Δφ≈2 rad over 4 seconds

**Kalman filter:**
- State: [x,y,z,v_x,v_y,v_z]^T (6D); Observation: [x,y,z] (3D); velocities inferred from the transition matrix
- K_t = P_{t|t-1}H^T(HP_{t|t-1}H^T + R)^-1 (Kalman gain = ratio of prediction uncertainty to total uncertainty)
- Large R (noisy sensor) → small K (trust the prediction more); small R → large K (trust the measurement more)

**Compressed sensing / ISTA:**
- h = Fs (channel = Fourier of sparse delay domain); L1 minimization gives the same solution as L0 (convex relaxation)
- ISTA alternates: gradient step (move toward matching measurements) + soft thresholding (enforces sparsity)
- Required measurements: p ≈ K·log(n/K); K=5 delay spikes, n=114 subcarriers → need ~16–50 measurements; 56 available at HT20 → comfortable margin
- Never quantize raw CSI amplitude — the signal lives at sub-millimeter precision

**Attention:**
- Q, K, V from input; scores = QK^T/sqrt(d_k); the division prevents winner-take-all in softmax
- WaveTrace application: 3 antennas = 3 tokens; attention learns per-subcarrier reliability weighting

**LoRA:**
- Full weight matrix W₀ frozen; patch = B·A where rank r << min(m,n)
- r=4: reduces 16M parameters to 32K; B initialized to zero → no change at start
- Removed from WaveTrace roadmap: infeasible for embedded deployment

**EWC++:**
- Catastrophic forgetting mitigation: penalty λ/2·Σ F_i·(θ_i − θ_i^A)²
- F_i = Fisher information (sensitivity of task A to weight i)
- Online update: F^(t)_i = γ·F^(t-1)_i + (1-γ)·(∂log P/∂θ_i)²

**HNSW:**
- Multi-layer graph; greedy descent top to bottom; ~100 comparisons vs 1M brute force (O(log N))
- HNSW itself does not cause data loss; quantization methods applied to vectors do
- Mitigation: two-stage retrieval + rescoring; tune ef_search, oversampling factor, PQ subquantizer count, M

### 4.2 Research Papers Reviewed

**Wi-Metal (Wu et al., 2016 IEEE ICC):**
- First paper using WiFi CSI for metal detection
- Hardware: Tenda AP (2.4 GHz, 7 dBi) + Intel IWL5300 (3 antennas, 30 subcarrier groups)
- Physical mechanism: metal reflects uniformly across subcarriers; biological tissue scatters irregularly
- Feature: amplitude |h_k| per subcarrier; Classification: K-Means (K=2, unsupervised)
- Result: metal clusters at amplitude ~6–8; non-metal ~1–2 (large gap)
- Multipath cancellation: phase differences between antennas (h11−h12) removes the dominant static paths
- Limitation: requires co-located TX/RX, person stationary at 1–3m

**Zhou et al. (2020 IEEE WCNC) — Walking Pedestrians:**
- Hardware: IWL5300, 3TX×3RX=9 antenna pairs × 30 subcarrier groups = 270 subcarriers; 5 GHz; 100 Hz
- Setup: TX and RX on opposite sides of corridor; person walks through (LOS doorway)
- Objects: 3 metal knife sizes (16 cm, 22.5 cm, 30 cm); 3 water volumes (180 ml, 360 ml, 540 ml)
- Pipeline: amplitude only → low-pass → LOF anomaly detection → extract middle 120 frames → CNN + majority voting
- Result: 93.3% accuracy for metal and liquid; majority voting across time snapshots is critical
- Physics: walking motion acts as a carrier — the knife modifies the shape of the crossing event
- Only published paper detecting knives; LOS walking-pedestrian regime

**Yousaf et al. (2025) — Gun Detection:**
- Hardware: 2.4 GHz ESP32 with 1.2m satellite dish (25 dBi, 7° beam); bistatic reflection setup
- Target: 6–9m; person stationary
- N-LOS geometry outperformed all 3 LOS configurations for gun detection
- Physics: the dominant direct path masks the faint gun reflection in LOS; N-LOS removes it
- Feature: phase variance σ²[p] across subcarriers (metal flattens the variance)
- 52 LLTF subcarriers at 2.4 GHz HT20

### 4.3 Architecture Gaps Identified from Papers

- **Single-antenna ESP32 cannot produce antenna-difference features.** All working papers key on h₁₁ − h₁₂ antenna-difference channel to cancel static multipath and expose the object's faint reflection. No DSP compensates for missing hardware.
- **Band mismatch for primary target.** The only knife paper (Zhou) used 5 GHz, 270 subcarriers, walking pedestrians. At 2.4 GHz, λ≈12.5 cm; a knife profile (~3–4 cm) is below the λ/2 resolution limit.
- **Within-session evaluation splits are misleading.** Sliding window hop creates heavy overlap between train/test windows. LOGO is required for honest evaluation.
- **Phase information discarded.** Wi-Metal's best result is phase-based; amplitude-only loses a key discriminator.
- **Variance feature (σ²[p]) fails for moving targets.** Body motion adds its own variance that overwhelms the weapon signal.
- **Geometry tension.** Yousaf (gun detection) used N-LOS static. Zhou (knife detection) used LOS walking-pedestrian. These measure different physical phenomena.

---

## 5. Hardware

### 5.1 Platform Decisions

- **IWL5300 (Halperin 2011):** 3 antennas, 30 subcarrier groups, 5 GHz, MIMO 3×3 — not viable; laptop-only, Linux driver, discontinued
- **ESP32:** correct platform despite RF frontend limitations (single antenna, hardware phase noise)
- **Upgrade path for 5 GHz:** Nexmon CSI on Raspberry Pi 5 (CYW43455 chipset); community patch, not in official Nexmon repo

**ESP32 multi-antenna reality:**
- ESP32 antenna diversity uses GPIO/IO MUX to route up to 16 switchable external antennas, but there is only ONE RX chain
- A single ESP32 cannot produce simultaneous multi-antenna CSI — antenna diversity is time-multiplexed, not simultaneous
- ESP32-WROOM-DA has dual PCB trace antennas but still a single chain
- Decision locked: additional antennas per board will NOT provide multi-antenna CSI

### 5.2 Node Configuration

- **6x ESP32-S3-DevKitC-1 (v1.1):** 2.4 GHz; STATUS_LED_GPIO=38 for WS2812 data line
- **1x Raspberry Pi 5:** 5 GHz via Nexmon on onboard CYW43455 chip; supports 5 GHz up to 80 MHz, 1x1; no external NIC needed
- **Antenna:** 8 dBi RP-SMA omnidirectional whip, 160mm, dual-band 2.4/5.8 GHz — not directional. Critical difference from Yousaf 2025 hardware (1.2m dish, 25 dBi, 7° beam).
- **Router:** old/cheap, locked to channel 6, 2.4 GHz HT40

**2.4 GHz HT40 environmental concern:**
- 40 MHz channel = 48% of the entire 2.4 GHz band (83.5 MHz total)
- Airport/school environment: congested 2.4 GHz → interference destroys CSI quality
- Espressif docs explicitly warn against HT40 in congested environments

### 5.3 Subcarrier Geometry

- HT20 → 20 MHz channel → 56 total subcarriers (52 data + 4 pilots); HT40 → 40 MHz → 114 subcarriers
- Mapping: 20 MHz → FFT 64 → 56 usable; 40 MHz → FFT 128 → 114 usable; 80 MHz → 256; 160 MHz → 484
- LLTF vs HT-LTF: HT20 LLTF covers −26 to +26 (52 active); HT-LTF covers −28 to +28 (56 active, adds ±27, ±28). 52 LLTF subcarriers are a subset of 56 HT-LTF.
- HT40: HT-LTF captures primary + secondary 20 MHz channels → 112 unique HT-LTF positions

### 5.4 Current Hardware Status (as of 2026-06-26)

- All 6 boards flashed with esp32_node firmware
- 3–4 boards used per experiment; remaining boards unused
- sdkconfig static TX buffer fix (TX_BUFFER_TYPE=0, rm sdkconfig + rebuild): a build was done after this change but whether it was confirmed flashed on all boards is unverified
- Antenna geometry: aimed toward center zone (NLOS bistatic); litmus gate not yet cleared

---

## 6. DSP Pipeline

### 6.1 Stage 0 — Calibration

- `Calibration.observe(frame)` feeds quiet-baseline frames into the GainLock accumulator
- `Calibration.ready` (property) returns True after ≥300 baseline packets
- `Calibration.finalize()` locks the reference amplitude scale (median of per-frame mean magnitudes)
- `GainLock.apply(frame)` rescales every sample by referenceScale / frameScale(frame); phase is untouched
- `GainLock.coefficientOfVariation()` fallback when locking is impossible (CV = σ/μ, gain-invariant)

**Calibration save/load:**
- `GainLock.lockTo(scale)` (C++): bypasses observe→finalize, directly sets referenceScale. Lets Python reconstruct a locked GainLock from a persisted scalar without re-running baseline capture.
- `save_calibration()`: writes meta.json (reference_scale, subcarriers, num_baseline), baseline_mag.npy, baseline_diff.npy
- `load_calibration()`: reads files, calls gain_lock.lock_to(ref); returns `(CalibrationResult, GainLock | None)` — the GainLock is None when reference_scale is NaN (gain lock was disabled at calibration time)
- Backward compat: meta.get("image_subcarriers", meta["subcarriers"]) — old calibrations without the key fall back to the NBVI set

### 6.2 Stage 2 — C++ Preprocessing

- `conjugateMultiply()`: ≥2 antennas → H[a][k] × conj(H[0][k]); single-antenna fallback → cross-subcarrier ratio H[0][k] × conj(H[0][k-1])
- `hampel()`: sliding window MAD test, k=5.0 default; 1.4826×MAD makes it a consistent σ estimator; spike detected → replace with median, phase held at last good value
- `unwrapStep()`: streaming O(1) phase unwrap per cell
- EMA normalize: α=0.1 (slow EMA removes DC drift, passes motion signal)

### 6.3 Stage 3 — Material Features (Weapon Path)

- `interCarrierStats()`: per-packet cross-subcarrier amplitude dispersion; flat metal → lower σ² than diffuse human body
- Critical constraint: must NOT run on NBVI-selected subset; must NOT gain-lock before; run over all K valid subcarriers
- `interCarrierPhaseStats()`: fits a least-squares line to unwrapped phase across subcarriers; metal → coherent/near-linear → small residualStd
- `reconstructComplexCsi()`: strips linear STO/CFO ramp, preserves absolute residual phase+magnitude (used in material-ID literature)

### 6.4 Stage 4 — Feature Extraction

- `nineFeatures()`: mean, std, max, min, IQR, skewness, lag-1 autocorrelation, MAD, waveform length — computed over a 128-frame window per subcarrier
- Window = 128 frames (~1.28s at 100 Hz), hop = 32 frames
- NBVI selects top-K subcarriers by time-variance; K=12 for the presence path
- Three parallel paths from iter_windows: 9·K feature vector (MLP/SVM), K×128 CSI image (CNN), 27 inter-carrier stats (weapon MLP)
- `SegmentVoter`: soft majority voting; per-snapshot CNN accuracy 51.1% → voting over a full walk → 93.3% (Zhou 2020)

### 6.5 GainLock Analysis

The ESP32 AGC varies transmit power, which changes amplitudes in ways unrelated to the environment.

- CV = σ/μ is gain-invariant: doubling all amplitudes scales σ and μ equally, so CV stays the same
- For the weapon path: CV is the preferred form for inter-carrier variance. Absolute σ² is destroyed by gain normalization. `coefficientOfVariation` is already implemented.
- This is a silent failure: a model trained on gain-locked amplitudes for weapon detection learns corrupted features with no error message.

### 6.6 NBVI (SubcarrierSelect)

- Offline selection of the most informative non-consecutive subcarriers for model input; K=12
- Runs after compressed sensing (which reconstructs missing subcarriers) — these are separate layers, not competing
- NBVI for CNN: real problem — CNN needs contiguous frequency rows for local spatial patterns; NBVI picks non-adjacent subcarriers. Fix: SpectrogramBuilder receives all S subcarriers in frequency order, not NBVI-selected K.

### 6.7 Architecture Fix: Dedicated ESP-NOW Link

**Problem:** CSI sensing breaks on production WiFi:
- Bursty traffic → non-uniform sampling → destroys the FFT frequency axis
- Variable MCS/AGC → moving amplitude reference → destroys gain-lock

**Fix:** Dedicated ESP-NOW controlled link (separate from data WiFi) with forced constant packet rate and locked TX power.

### 6.8 Variance Computation: Two-Pass vs Welford

- All WaveTrace sites (interCarrierStats, nineFeatures, PresenceSegmenter::windowCv) use two-pass computation (compute mean first, then sum squared deviations)
- Welford recurrence is loop-carried → prevents SIMD auto-vectorization; two-pass is faster on contiguous float arrays at small sizes (K≈52–128)
- NBVI in Calibration.py could use Welford to reduce memory from O(F·S) to O(S) during arbitrarily long calibration; not implemented (offline baseline has fixed F=3000)
- Sliding-window variance (windowCv): Welford has no deletion step; correct approach = ring-buffer sum + sum-of-squares with periodic exact recompute

### 6.9 InterCarrierExtractor Gap

- σ²[p] was computed per-packet via interCarrierStats but never aggregated into a window-level feature vector
- The weapon signal = how σ²[p] behaves over the ~1.3s window (shape, stability, drift), not a single-packet value
- Fix: InterCarrierExtractor mirrors FeatureExtractor; pushes per-frame {μ[p], σ²[p]}, runs nineFeatures over those series every hop

---

## 7. Software Pipeline

### 7.1 Phase 6 — Presence Head

**`recognition/Model.py` — PresenceHead:**
- Backend-agnostic: fit(), predict(), predict_proba(), save(), load()
- Default backend: StandardScaler + MLPClassifier pipeline (scaler is inside the pipeline → applied automatically at inference)
- SVM backend: CalibratedClassifierCV(SVC(), ensemble=False); ensemble=False fits one calibrator on the full training set; needed because sklearn 1.9 deprecated SVC(probability=True)

**`Evaluate.py` — Evaluation gate:**
- LeaveOneGroupOut over both session and subject
- Per-fold accuracy + pooled accuracy + confusion matrix
- Two baselines per fold: majority-class + PresenceSegmenter (C++ DSP gate replayed on the same windows)
- PresenceSegmenter baseline prevents reporting improvements that don't beat a well-tuned threshold rule

**`Infer.py`:**
- InferenceSession.predict_window(feat) → (class_id, proba)
- Measured latency: ~0.05 ms (160x under the 8 ms DoD budget)

**`Fusion.py` / `Resample.py`:**
- Per-node feature concat: O(m), no cross-node conjugate multiplication
- resample_uniform: linear/cubic interpolation onto a uniform time grid
- fs_ok: drops windows where measured fs deviates beyond tolerance
- out= buffer reuse for zero-allocation hot path

**Synthetic results:**
- LOGO (leave-one-session-out): 0.980
- LOGO (leave-one-subject-out): 0.975
- Majority-class baseline: 0.51
- PresenceSegmenter calibrated baseline: 0.861
- PresenceSegmenter best-possible sweep threshold: 0.963 — ML head beats it
- Turbulence floor ≤0.05 genuinely inseparable (held-out fold drops to 0.50); documented

**Fixture design choices (making evaluation honest):**
- Common-amplitude wobble in both classes (proxy for narrowband interference): prevents a scalar energy gate from identifying class by energy level alone
- Turbulence levels interleaved across subjects: prevents subject-identity leakage

### 7.2 Phase 7 — Weapon Head Architecture

**Input features — 27 raw inter-carrier features (ic27):**
- Must NOT use gain-locked amplitudes; normalization destroys amplitude flatness = the metal signature
- 27 features: mean, σ², CV, plus cross-subcarrier statistics from the inter-carrier variance series

**Alternative backend — CNN on spectrogram images:**
- Full temporal structure vs. summary statistics; potentially better for dynamic events
- Requires walking-pedestrian data to be effective (static body + knife = insufficient discriminative motion)

**SegmentVoter:**
- Accumulates probabilities across a detection segment; outputs a fused verdict after the segment closes
- Verdict gate thresholds: FP ≤ 10%, TPR ≥ 90%

**Feature fusion (9K + ic27):**
- Pipeline conflict: 9K features need gain-locked amplitudes; ic27 features need raw amplitudes
- Fix: compute raw magnitudes first → ic27 features → THEN apply gain lock for 9K path
- Decision: wire dual-block dataset builder for fusion-readiness; train fused model only after real data (overfitting risk at ~135 features on a few hundred samples)

**Three-category dataset requirement confirmed:**
- Empty room + Person without weapon + Person with weapon — NOT binary
- Without empty-room negatives, FP ≤ 10% in the lab does not reflect deployment performance

### 7.3 Phase 8 — Pipeline Completion

- 8a: GainLock::lockTo() (C++) + save_calibration() / load_calibration() (Python)
- 8b: Frontend.iter_windows parity fix (single emit loop shared by train and serve)
- 8c: Publisher ABC + JsonlPublisher (zero-dep, JSONL, flush every write)
- 8d: CsiSource / SyntheticSource / RecordingSource + serialization; SerialReader = documented Phase-0 seam
- 8e: WeaponHead feature_mode stamped into saved head; _serving_plan reads it to determine apply_lock, intercarrier, input pick
- 8f: Cli.py (argparse, 6 modes: capture / calibrate / collect-data / train / localize / run)
- 8g: TestPipeline.py (10 tests)

Rebuild required after touching C++: `pip install -e . --no-build-isolation`

**Parity invariant (critical engineering rule):**
- `Frontend.iter_windows` is shared by training and serving — divergent code paths produce feature mismatch and silent model corruption
- Per-frame order: IC extractor sees raw magnitudes BEFORE gain_lock.apply() → then gain lock applied → then NBVI-selected mags pushed to FeatureExtractor and SpectrogramBuilder

**Bugs found and fixed in Phase 8:**
- `train_presence`/`train_weapon` created ModelConfig with window=128 ignoring the dataset's actual window; fix: pull window/hop from dataset meta
- `result_to_dict` used result.classId (wrong); correct pybind attribute is result.class_id

**Multi-node architecture:**
- iter_windows_stacked(): lockstep-zip one iter_windows per node; yields (N·9·K features, N×K_img×window image, N·27 IC)
- Node/channel order = sorted node IDs (deterministic)
- Timestamp tolerance check: if max(ts) − min(ts) > node_tolerance → ValueError (Phase 0 owns time sync)

**Antenna handling:**
- Phase path (Preprocessor): conjugate multiply consumes the antenna dimension → (A-1)×S; preserves spatial phase difference (AoA); not averaging
- Amplitude path (Frontend.py:41): mags = np.abs(np.asarray(fr.grid)).mean(axis=0) — collapses antenna dimension; with single-antenna ESP32 nodes, fr.grid is (1,S) so mean(axis=0) is a no-op; the real channel axis is NODES, not antennas; fix = stack nodes as channels
- CNN subcarrier handling: SpectrogramBuilder receives all S subcarriers in frequency order, not NBVI-selected K (NBVI picks non-adjacent subcarriers → meaningless local correlations in CNN)

**AoA heatmap capability (corrected understanding):**
- Early diagnosis "single TX-RX ESP32 pair cannot produce 2D spatial heatmap" was incorrect
- Localize.py already builds AoA heatmap from ≥2 RX antennas on one radio (shared clock enables inter-antenna phase)
- Independent per-node STO/CFO makes absolute ToF infeasible without per-node delay calibration (not yet implemented)

**Bug fix (2026-06-26) — collect_source train/serve mismatch:**
- collect_source() in Cli.py was passing gain_lock to build_dataset for the weapon stage, applying gain-lock normalization to ic27 features at training time, destroying the σ²[p] amplitude-flatness signal. Serving correctly used apply_lock=False.
- Fix: `effective_lock = None if intercarrier else gain_lock`

### 7.4 CIR Super-Resolution Module (built 2026-06-22)

**Weapon signal discriminators — implementation status:**

| Discriminator | Physical mechanism | Status |
|---|---|---|
| Inter-subcarrier amplitude variance σ²[p] | Metal → flat reflection → lower σ² than diffuse human body | Done: interCarrierStats / ic27 features |
| Inter-subcarrier phase residual | Metal → coherent reflection → near-linear phase → small residualStd | Done: interCarrierPhaseStats |
| Pure object reflection (reflectionNull / β-null background subtraction) | Empty-room cancellation exposes object-only return; needs two-path geometry | Done: reflectionNull (needs 2 paths) |
| Reconstructed complex CSI | Strip linear STO/CFO ramp; dielectric signature in absolute residual phase+magnitude | Done: reconstruct_complex_csi |
| CIR delay taps | Metal → sharp single tap; body → diffuse multi-tap cluster; useful at HT80 (~1.2m path resolution) | Done: Cir.py |
| CNN + majority voting over crossing event | Learns amplitude pattern across full walk; requires walking-pedestrian geometry (Zhou 2020) | Done: WeaponHead(backend="cnn") + SegmentVoter |
| Permittivity-vs-frequency ε(f) | Wideband frequency-dispersive dielectric signature; needs wide bandwidth | Planned: unlocked by HT40/HT80 |

Usage constraints: σ²[p] and reflectionNull must run on raw (pre-gain-lock) magnitudes. reconstructComplexCsi must follow phase sanitization or STO/CFO ramps become ghost taps. CIR is useful only at 5 GHz HT80 (~1.2m path separation).

**Research basis:**
- Subcarrier infill (recover missing tones): not useful for ESP32 2.4 GHz HT20/HT40; active set is contiguous; ISTA handles gaps natively; infill does not extend aperture
- Delay-domain super-resolution via ISTA on sub-DFT dictionary: the valuable path
- Reference: RuView ADR-134 (ISTA/L1, G=3K oversample, pilot/null masking, phase-sanitize-first)

ISTA vs OMP: OMP commits permanently on each iteration; body+object may fall in one Nyquist bin → OMP's early wrong pick is irreversible. ISTA's soft thresholding degrades gracefully.

**CIR parameters (ADR-134 §2.3, Δf=312.5 kHz):**

| Band | Active | G | Delay Res | Path Res | λ | iters |
|---|---|---|---|---|---|---|
| HT20 | 52 | 156–168 | ~17 ns | ~5 m | 0.05 | 30 |
| HT40 | 108 | 342 | ~9 ns | ~2.7 m | 0.03 | 35 |
| 80 MHz Pi/Nexmon | 242 | 768 | ~4 ns | ~1.2 m | 0.02 | 40 |

**Cir.py module:**
- Functions: delay_dictionary, estimate_cir_taps (ISTA), cir_from_csi, cir_features
- Tap detector: local maxima + nearest-peak basin (per ADR-134 §2.9 tolerance-aware detector)
- TestCir.py: 5 tests, all passing — two-tap recovery, sub-Nyquist separation, dictionary conditioning, gapped band, error cases
- Verified: recovers synthetic 2-tap channel to within one delay bin, κ(Φ)≈1

**Test on real HT20 data (72 labeled recordings):**
- Group-aware CV (no leakage): CIR delay features AUC≈0.60; ic27 variance + RF AUC≈0.45; CNN on CSI images AUC≈0.32
- All at or below chance — HT20 hand-held/omni data carries no separable weapon signal under any method
- Diagnosis: papers that work used static subject + chest placement + directional antenna + weapon in the LOS

**CIR utility by band:**
- 2.4 GHz HT20: entire room collapses to ~one CIR cluster (~5m path separation with CS) — weapon may not resolve from body's reflection
- HT40: ~2.7m path separation — marginal
- 5 GHz HT80: ~1.2m — genuinely useful for separating body tap from object tap

**HT40 switch:**
- Changed WT_BW_HT40 0→1 in firmware/esp32_node/main/config.h
- HT-LTF = 128 tone slots (−64…+63); ~108–114 carry real channel info; ~14–20 are structural nulls (DC, central gap, edge guards)
- Must recapture calibration + retrain after switch (HT40 changes byte-width and subcarrier count)

---

## 8. Firmware

### 8.1 Mesh Architecture

- Unified firmware: every board runs the same binary; NODE_ID is the only per-board difference
- esp32_rx / esp32_tx are legacy firmware; do NOT set CSI bandwidth
- Token-ring TDMA: BURST_LEN=10 frames, BURST_MS=2 → 20 ms/burst per turn
- Leader = lowest active node ID; TURN_TIMEOUT_MS=80 self-heal; LIVE_TIMEOUT_MS=1500 eviction; 5s re-discovery (admitted nodes), DISCOVERY_MS=300 aggressive announce (joiners)
- Dynamic ring: no hardcoded node count; active count learned live; MAX_NODES=16 is array capacity only
- TOKEN_REPEAT=3: handoff carried on the last 3 burst frames → survives single-frame loss
- WT_BW_HT40=1 activates HT40; requires router also on 40 MHz (silent fallback to legacy otherwise)
- Rate config: esp_now_set_peer_rate_config({.phymode=WIFI_PHY_MODE_HT40, .rate=MCS0}) — critical; without HT40 rate, the frame carries L-LTF (20 MHz, ~52 tones) even on a 40 MHz channel

The full config.h settings table (BURST_LEN, BURST_MS, MAX_NODES, LIVE_TIMEOUT_MS, etc.) is in firmware/README.md.

### 8.2 Bugs Found and Fixed

**ENOMEM backoff bug:**
- On sendto ENOMEM, code threw away the entire batch and slept 100ms
- At ~250–400 fps, a 256-entry queue overflows in ~0.6s → multi-hundred-frame hole from a single transient blip
- Fix: retry-don't-drop — keep unsent bytes, vTaskDelay(2), retry up to UDP_SEND_RETRIES=3; only then drop; backoff 100ms→15ms; poll interval 10ms→5ms

**CSI queue size (stale comment caused ENOMEM):**
- Queue was 256×396B ≈ 99KB at HT40; comment claimed "68KB" (the HT20 number, stale)
- Heap dipping to 37–60KB → queue causing heap-side ENOMEM on Node 2
- Fix: shrink to 128×396B; comment updated; still ~0.5s burst tolerance

**sdkconfig static TX buffers:**
- TX_BUFFER_TYPE=1 (dynamic): TX buffers compete with the CSI queue for DRAM → starved under heap pressure
- Fix: TX_BUFFER_TYPE=0 (static), STATIC_TX_BUFFER_NUM=16 (~26KB pre-allocated); cannot be starved by heap squeeze
- Must `rm sdkconfig` before build to force sdkconfig.defaults to re-apply (ESP-IDF will not override existing keys)

**SNTP dynamic sync gap:**
- Startup uses fallback SNTP_SERVER=PC_IP from config; if the PC IP changes after boot, clocks never re-sync → frame alignment failure
- Fix: discovery_task calls esp_sntp_stop() + esp_sntp_setservername() + esp_sntp_init() on new PC IP discovery

**IP print type mismatch:**
- s_pc_addr is struct in_addr (has .s_addr); IPSTR/IP2STR expect esp_ip4_addr_t → compile error
- Fix: use inet_ntoa(s_pc_addr)

**Token ring startup order bug:**
- Starting nodes in order 1→3→2 with MESH_NODES=3: ring 1→2→3→1; token passed to absent node 3 → token dies → leader-heal after 300ms → TX only 10–20/s instead of ~50+
- Fix: power all three boards simultaneously; ring closes and tx/csi_hz jumps

**LED GPIO bug:**
- An incorrect fix had been applied: "force GPIO 38 HIGH to power the RGB LED" — wrong on DevKitC-1 v1.1; GPIO 38 IS the WS2812 data line
- Forcing GPIO 38 HIGH blocked all serial data commands → LED dark
- Correct fix: STATUS_LED_GPIO=38 (data line); no separate power enable pin on v1.1; RGB_PWR_GPIO commented out

**CMake dependency error:**
- Build failed: `driver/gpio.h: No such file or directory`
- Fix: add esp_driver_gpio to REQUIRES in CMakeLists.txt

**HT40 rate config moved to post-association event hook (2026-06-26):**
- Original: esp_now_set_peer_rate_config() called in app_main
- Root cause: HT40 requires a secondary 40 MHz bonding channel that does not exist until after the router handshake; calling in app_main fails silently with ESP_ERR_ESPNOW_ARG, falling back to legacy 1 Mbps baseband → wrong subcarrier count or severe packet loss
- Fix: rate config call moved into wifi_event_handler on IP_EVENT_STA_GOT_IP; rate re-evaluated on every successful association

### 8.3 Health Monitor Results (3-node observed)

- Healthy mesh sustained csi_hz≈200–300 for 10+ minutes with all 3 nodes
- Node 1 (rssi≈−58): runs clean, consistent
- Node 2 (rssi≈−44, closer to AP): hits ENOMEM first — counterintuitive; better SNR → decodes more frames → more to offload → TX path saturates sooner
- Node 2 symptoms: intermittent sendto ENOMEM, heap dipping 37–60KB, periodic peers=0/csi_hz=0 dropouts
- Health output format: node ip age csi_hz tx_hz peers leader gain agc rssi heap(KB) up(s) clk
- clk=ok means NTP-synced; clk=no means sync failed → tx_hz=0 permanently (cannot schedule TX slot)

**4-node diagnosis (2026-06-25):**
- Node 2 (10.8.1.100): tx_hz=0, clk=no, peers=3, RSSI=−36/−37 dBm
- Node 1 (10.8.1.101): crash loop every ~20s; csi_hz frozen at 151 (cached value, not live)
- Node 2 root cause: NTP sync failed → token scheduling impossible → tx_hz=0 permanently; node 2 was physically too close to the router (RSSI −36 vs target −50 to −60 dBm)
- Node 1 root cause: promiscuous flood crash (wDev_SnifferRxData crash under promiscuous packet flood); fix identified (narrow RX filter to DATA + bump RX buffers) but not yet applied
- Both nodes resolved through physical fixes: boards replugged, antenna connections reseated, modem distance corrected

### 8.4 Host-Side Issues

**Single-threaded recv loop:**
- UDP socket with no SO_RCVBUF bump; drain recv in the same thread as inference → during inference tick (~1.5s), the kernel UDP buffer overflows and drops datagrams
- Fix: SO_RCVBUF bump + dedicated reader thread

**AGC skip corrupts σ²:**
- Firmware skips gain lock when AGC<30 (main.cpp:233)
- On strongest/closest nodes (Node 2 @ −44 dBm), AGC floats ±20–30% → σ² corrupted; best-placed hardware = least-reliable weapon hardware

**Zero-config PC discovery:**
- health_monitor.py broadcasts b"WAVETRACE_PING" to 255.255.255.255:9878 every 2s
- discovery_task binds DISCOVERY_PORT, matches ping, adopts from.sin_addr → nodes auto-discover PC IP without manual config.h edits

**Port 5566→9876 mismatch (fixed 2026-06-26):**
- All real-world scripts, Pi firmware, and DevicePanel docs locked to port 9876; web/streamer.py UdpSource was hardcoded to 5566 → live web UI received zero hardware frames; silent, no error
- Fix: dynamic udp_port parameter added to StartRequest, web/app.py, and sock.bind; UDP Port field added to Controls.tsx

**ESP32 throughput ceiling:**
- 200 pps configured → ~182 received; serial saturates at ~30 pps
- Capping rate to 120–150 Hz loses no signal (human motion bandwidth <50 Hz); saves ~50% uplink
- ESP-NOW multi-node scaling: each node transmits 1/N of the time; aggregate uplink = 500·(N-1) frames/s; at N=7 → ~1.2 MB/s on one half-duplex 2.4 GHz channel

---

## 9. Experiments and Results

### 9.1 Weapon Detection — Body-Worn (2026-06-23)

Setup: person standing still, weapon concealed on body, 3 sessions, 2.4 GHz HT40, LOGO cross-validation

| Node | LOGO | TPR | FPR | Verdict |
|---|---|---|---|---|
| Node 1 | 0.416 | 68.5% | 82.5% | Worse than chance |
| Node 2 | 0.588 | 42.3% | 26.2% | Marginal |
| Node 3 | 0.464 | 39.3% | 47.3% | Worse than chance |

Phase variance litmus (σ²[p]):

| Node | AUC | Result |
|---|---|---|
| Node 1 | 0.594 | Weak |
| Node 2 | 0.502 | Inverted |
| Node 3 | 0.639 | Weak |

Root cause: human body movement/breathing overwhelms metal's radio reflection at 2.4 GHz; the weapon adds a perturbation 2+ orders of magnitude smaller than body motion.

Data hygiene issue: first run missing --root → data saved to data/weapon_ds/ instead of data/2g4_ht40/; manually relocated; second run mixed clean + noisy sessions → degraded results further.

### 9.2 Weapon Detection — Desk-Based, No Person (2026-06-23)

Setup: metal object stationary on desk, person left room, 3 sessions, --subject weapon --carry onDesk

| Node | LOGO | TPR | FPR | Verdict |
|---|---|---|---|---|
| Node 1 | 0.455 | 64.8% | 71.8% | Poor |
| Node 2 | 0.782 | 54.6% | 1.4% | Strong |
| Node 3 | 0.800 | 98.6% | 37.3% | Strong |

Phase variance litmus:

| Node | AUC | Result |
|---|---|---|
| Node 1 | 0.502 | No separation |
| Node 2 | 0.537 | Inverted |
| Node 3 | 0.567 | Weak |

Key finding: litmus (σ² phase variance) showed no separation; ML (ic27) reached LOGO=0.800 — they measure different physical properties.
- Phase variance litmus: measures whether overall room phase stability shifts (metal stabilizing multipath)
- ic27 ML: measures frequency-domain relationships across subcarriers (amplitude flatness across 114 subcarriers)
- A poor litmus score does NOT necessarily predict ML failure

Leakage caveat: 3 back-to-back sessions captured minutes apart; LOGO may be measuring "room at 3:05 vs 3:10", not the metal; Node 3's 37% FPR is the tell.

### 9.3 Per-Link Litmus Discovery (2026-06-23)

experiments/weapon_litmus.py had been grouping σ² by RX node, pooling all TX directions into one node head. Per-node pooling averages 1 good NLOS direction with noise links → sign-flip and washout.

Per-link results (6 active links):

| Link | AUC | Direction |
|---|---|---|
| 4f9c→2 | 0.652 | ok (weapon→lower σ²) |
| 64b8→3 | 0.636 | ok |
| 4568→2 | 0.545 | inverted |
| 4568→3 | 0.516 | inverted |
| 64b8→1 | 0.509 | inverted |
| 4f9c→1 | 0.504 | ok |

Node 2 pooled: AUC=0.537 inverted — sign-flipped by mixing links; signal was present, pooling destroyed it. Fix: added --per-link flag to experiments/weapon_litmus.py; groups by (rx_node, tx_tag).

### 9.4 Backend Bake-Off — Experiments A–E (2026-06-25)

Central harness: experiments/weapon_experiments.py. Evaluated on session-LOGO only (3 sessions → 3 folds).

**Experiment A — Per-node, four backends:**

| Node | variance AUC | mlp AUC | svm AUC | cnn AUC |
|---|---|---|---|---|
| Node 1 | 0.528 | 0.492 | 0.492 | 0.162 |
| Node 2 | 0.611 | 0.502 | 0.502 | 0.438 |
| Node 3 | 0.528 | 0.469 | 0.469 | 0.347 |
| Node 4 | 0.630 | 0.562 | 0.531 | 0.162 |

- Best backend: ic27/variance everywhere; MLP/SVM worse than variance on real data at this scale
- CNN worst: ~166 aligned windows × 1 channel — too few samples; overfits to session identity, not weapon physics
- Decision: variance/threshold is the correct baseline at this data scale; CNN only viable when n is in the thousands

**Experiment B — Per-link variance (12 directed links):**

| Link | AUC | Majority |
|---|---|---|
| node3 ← 64b8 | 0.737 | 0.485 |
| node2 ← 4f9c | 0.647 | ~0.49 |
| node2 ← 4568 | 0.611 | ~0.49 |
| node3 per-node pooled | 0.528 | ~0.49 |
| node1 ← all links | 0.424–0.528 | ~0.49 |

- Per-link >> per-node; node3←64b8 = 0.737 while node3 pooled = 0.528
- Root cause: pooling averages 1 NLOS-good link with 2 noise links → dilutes and sign-flips signal
- 6 of 12 links beat majority class (AUC > 0.50)

**Experiment C — Combined 12-link multi-channel CNN:**
- All 12 links stacked as channels; result: AUC = 0.500 ≈ chance; 166 windows × 12 channels — even fewer effective samples per channel
- Conclusion: stacked multi-channel CNN needs n in the thousands; shelved

**Experiment D — Flat per-link weighted fusion:**

Design: weight = max(LOGO_acc − 0.5, 0) × 2 from TRAIN folds only; final score = Σ p·w / Σw

| Link | Weight |
|---|---|
| n3/64b8 | 0.379 |
| n4/4f9c | 0.210 |
| n2/4c1c | 0.171 |
| n4/64b8 | 0.153 |
| n1/* | 0.006–0.018 |

- Auto-muting works: node1 links (consistently near-chance) → w≈0.006–0.018, contribute almost nothing
- Fused system score: LOGO=0.494, TPR=0.325, FPR=0.337 — below majority
- Contradiction: individual link node3←64b8=0.737 but fused=0.494 — a high-weighted link fires high-p on BOTH classes in the test session → fold-luck in the per-link number
- The honest deployment number is the fused one (0.494), not the best individual link
- ROC-AUC = 0.626 — weak ranking signal exists; this is the leak-free equivalent of "optimal threshold"

**Experiment E — Drift fix ablation:**

| Config | AUC | acc@0.5 |
|---|---|---|
| Baseline | 0.626 | 0.494 |
| + Hampel/MAD denoise (k=3) | 0.630 | 0.494 |
| + per-session z-norm | 0.468 | 0.477 |
| + augment ×5 (jitter+mag-warp) | 0.673 | 0.494 |

- Hampel: no effect; signal is not spike-limited
- Augmentation: AUC 0.626→0.673 (regularizes ranking); acc unchanged; keep for future but cannot create cross-session threshold transfer alone
- Per-session z-norm: HURTS — removing absolute σ² level collapses the signal below chance. This is a confound test: the baseline AUC 0.626 rides on absolute σ² level that is confounded with session/capture conditions, not a level-invariant weapon signature. For real cross-session deployment there is ~no signal at current data scale.

### 9.5 Weapon Root Cause Analysis

**Physics (dominant cause):**
- 2.4 GHz λ≈12.5 cm ≥ most concealed object sizes → weak interaction
- Wu Wi-Metal 2016: objects < 12 cm "invisible" at 2.4 GHz
- Omni antenna + LOS: direct path and room multipath dwarf the few-percent metal echo
- Moving body: breathing/micro-sway modulation is orders of magnitude larger than the concealed object echo
- Yousaf 2025: gun undetectable in ALL 3 LOS geometries; only appeared in non-LOS scattering with directional RX + omni TX

**Geometry:**
- All captures were LOS or hand-held
- Yousaf scenario 4 (directional RX, omni TX, no direct path) → 87.5% gun detection on body; hardware: 1.2m satellite dish, >25 dBi, 7° beam

**Data hygiene:**
- Mixed physical conditions in the same pool → model learns room state, not weapon
- Back-to-back sessions → LOGO measures time confound not metal

**Code architecture:**
- Per-node pooling proven harmful by per-link litmus
- Paper accuracies are leakage-inflated: reference code does shuffle-then-KFold over windows; 95–99% accuracy collapses under proper subject/session separation

**AGC/amplitude corruption:**
- Firmware skips gain lock when AGC<30 → on strongest/closest nodes (Node 2 @ −44 dBm) AGC floats → σ² corrupted; best-placed hardware = least-reliable weapon hardware

### 9.6 NLOS Geometry Plan

**Physical configuration:**
- Aim all 6 antennas at the central target zone
- Every tx→rx pair becomes bistatic NLOS-scatter: direct node path falls into sidelobes (~15–25 dB suppression)
- 30 angle-pairs = 30 bistatic views

**Experiment protocol (gated):**
1. Center-aim antennas
2. `scripts/collect_baseline.py --root data/2g4_ht40` (recalibrate at new geometry)
3. One condition at a time; large metal plate (30–45 cm) first; sessions on different days
4. `scripts/collect_weapon.py --root data/2g4_ht40 --subject plate --carry chest --sessions 5`
5. `python experiments/weapon_litmus.py --root data/2g4_ht40 --per-link --plot` — GATE: AUC >= 0.65 on at least 1 direction before any ML
6. If plate separates → shrink object stepwise: plate → laptop → large knife → pistol; find the boundary

**Planned per-link model refactor (gated on litmus):**
- Train one head per (tx_tag→rx_node) direction
- Drop links below AUC gate and links with inverted physics direction
- Weight LinkVoter on per-link LOGO AUC (not per-RX-node)

### 9.7 Honest Weapon Evaluation Results (2026-06-26, group-aware, no leakage)

ic27 RF (AUC per-recording, grouped 5-fold):

| Dataset | Sessions | Recs | ic27 AUC | CNN AUC |
|---|---|---|---|---|
| ilker_hand HT20 Jun-18 cal | 3 | 72 | 0.455 inverted | 0.372 |
| p0_na body-worn HT40 Jun-23 cal | 3 | 36 | 0.290 inverted | — |
| ilker_metal_walk HT40/ui Jun-25 cal | 4 | 96 | 0.300 inverted | 0.215 |

No above-chance body-worn weapon signal found in any dataset. The 0.566 result from a previous session was an artifact of mixing weapon_onDesk (empty room, metal object) with p0_na (person, no weapon).

The Jun-23 desk-detection result (LOGO 0.782 Node 2, 0.800 Node 3) remains valid but is a different physical problem: static object in empty room, no person present.

### 9.8 Literature Research — Improvement Levers (Ranked by Evidence)

1. **Motion, not static (most impactful):** Zhou 2020 walking pedestrians → 95.6% accuracy; discriminative signal = body+object perturbation during movement. Adding a walk-through tier is the single highest-leverage data collection change.

2. **Data scale (quantified gap):** Yousaf 2025 used 7000 heatmap images, multiple subjects. Current: ~166–560 windows, 1 subject, 3 sessions. Need ≥5–10 subjects, multiple carry positions, separate days.

3. **CSI ratio / antenna-pair division:** Divides adjacent antenna pairs — cancels CFO/SFO and static channel → higher SNR, drift-robust. Achievable with a code change only (FarSense, sensors-24-07195).

4. **Richer features:** PSD + wavelet (Yousaf), amplitude-ratio + phase-diff (material-ID literature), cross-link correlation patterns.

5. **Directional antenna:** Yousaf: ~25 dBi / 7° beamwidth; current: 8 dBi omni collinear dipoles. Directional suppresses multipath clutter and amplifies weapon reflection in NLOS geometry.

Preprocessing tricks alone cannot rescue the data-scale ceiling.

### 9.9 Moving-Subject Weapon Collection Protocol

- New baseline required before any new weapon collection
- `scripts/collect_baseline.py --frames 3000 --root data/2g4_ht40/ui`
- `scripts/collect_weapon.py --subject ilker --carry metal_walk --sessions 3 --frames 1500 --per-link --root data/2g4_ht40/ui`
- Script prompts twice per session: "clear" (walk without weapon), "weapon" (walk with metal object)
- 4 nodes auto-detected from data/2g4_ht40/ui/cal/node*/

### 9.10 People Count Pipeline

**scripts/collect_count.py:**
- Prompts per count level (0, 1, 2, 3+) per session
- Cumulative index i is within the current run; back-to-back 3-session runs overwrite count_ds_0/1/2 from prior runs

**Walking vs. standing:**
- Initial approach (wrong): stand still during count capture
- Corrected: people must walk; count signal = dynamic multipath perturbations from N people moving simultaneously
- Session diversity: vary walking paths (parallel → perpendicular → crossing)

**Count Run 1 results (3 sessions × 3000 frames):**

| Node | LOGO | Majority |
|---|---|---|
| 1 | 0.573 | 0.253 |
| 2 | 0.554 | 0.251 |
| 3 | 0.464 | 0.253 |
| 4 | 0.543 | 0.241 |

- train_acc=1.000 → severe overfitting to session
- 0-people: 71–97% (empty vs. motion = strong binary signal); 2-people: 29–35% (hardest)

**Count Run 2 results (6 sessions × 6000 frames):**

| Node | Old LOGO | New LOGO | Delta |
|---|---|---|---|
| 1 | 0.573 | 0.597 | +0.024 |
| 2 | 0.554 | 0.611 | +0.057 |
| 3 | 0.464 | 0.528 | +0.064 |
| 4 | 0.543 | 0.582 | +0.039 |

- train_acc dropped from 1.000 to 0.863–0.912 → generalization gap narrowing
- 2-people still hardest (36–40%); diminishing returns: needs physical diversity

### 9.11 Camera-Supervised Heatmap Pipeline

Note: camera-supervised collection has never been run end-to-end. All code is designed only.

**Training phase:**
- MacBook FaceTime webcam → YOLO-seg → 16×16 per-frame occupancy mask
- Masks supervise CSI image tensor (nodes × subcarriers × window) → HeatmapHead CNN; camera not needed at runtime
- Fallback (_occupancy_fallback): if heatmap.joblib missing → spectral amplitude blob (not learned positions)

**Critical limitation — single-person only:**
- VisionLabeler._detect() (CameraLabeler.py:125–126): picks only the highest-confidence detected person; second person's mask is silently dropped
- With 2 people in frame: CSI sees both bodies, mask shows only 1 → contradictory training labels
- Fix identified: union all detected person masks (not implemented)

**Collection guidelines:**
- Walk grid pattern (corners → edges → center); not near center only
- Slow pace (<~0.5 m/s); pause 2–3s at corners; stay fully in camera frame
- 120s minimum recommended (~1800 labeled frames); multiple sessions on different days preferred

### 9.12 Pi 5 / 5 GHz Architecture

Note: Pi 5 Nexmon has not been set up. All of the following is designed only.

**Hardware:**
- Pi 5 onboard chip: CYW43455 (same family as Pi 3B+/4B BCM43455c0); Nexmon CSI via community patch
- 1×1 radio (single-antenna); does not unlock multi-antenna methods
- Supports 5 GHz up to 80 MHz

**Illuminator architecture:**
- Pi in monitor mode cannot self-illuminate (cannot associate to AP)
- Mac runs sustained traffic to router (ping -i 0.003 → ~300 Hz) → modem transmits replies → Pi sniffs AP→client frames and extracts CSI
- Pi purely receives; Mac is the illuminator

**Wire format:**
- `_BIN_HDR = struct.Struct("<BBBQH")`, magic=0x57, ver=2, int8 parsing `d[1::2] + 1j*d[0::2]`
- Same port 9876 as ESPs; host auto-discovers node 5 from node_id in frames

**CSI_SCALE bug (fixed in design):**
- Original CSI_SCALE=None: auto-scales every frame to peak 127 → destroys inter-frame amplitude = the metal signature (fatal for weapon)
- Fixed to CSI_SCALE=1.0

**5 GHz bandwidth decision:**
- HT80 chosen: delay resolution ~4 ns at 80 MHz vs ~9 ns HT40 → critical for CIR separating body vs object taps
- Router: must fix to non-DFS channel 36, disable auto-channel/DFS/band-steering

**firmware/pi/config.py key settings:**

| Setting | Value | Significance |
|---|---|---|
| NODE_ID | 5 | Appears alongside ESP nodes 1–4 |
| CHANNEL_SPEC | "36/80" | Channel 36, HT80 (256 subcarriers) |
| WIRE_VER | 3 | int16 I/Q; keeps absolute amplitude for the weapon feature |
| CSI_SCALE | 1.0 | Fixed; NEVER per-frame auto-scale for weapon |
| UDP_PORT | 9876 | Same as ESP mesh; host auto-discovers |
| PC_IP | "TODO_MAC_LAN_IP" | validate() fails loudly if still TODO |
| AP_BSSID | "TODO_MODEM_B_5G_BSSID" | validate() fails loudly if still TODO |

---

## 10. Web Dashboard and Frontend

### 10.1 Bugs Fixed

| Bug | Fix |
|---|---|
| FlashRequest missing clean attribute | Added clean: bool = False to Pydantic model |
| ANSI color codes rendering as raw escapes | ansi-to-react library + CommonJS/ESM .default export fix |
| Stop buttons disappearing on reload | Added /api/device/state + /api/pipeline/state endpoints; hooks query on load |
| Tabs not manually closeable | X button added; closing monitor tab stops process |
| Tab ordering unstable | Fixed with alphabetical .sort() |
| Single global 500-line log pool | Per-tab limit: 1000 lines per source |
| Auto-scroll unconditional | Smart auto-scroll: stays at bottom unless deltaY < 0 (explicit scroll up) |
| Interactive Python scripts blocking | Continue button injects Enter into subprocess stdin; scripts updated with -u flag + explicit flush() |
| HT bandwidth unknown at runtime | Auto-computed badge from subcarrier count: 64=HT20, 128=HT40, 256=HT80 |
| --root parameter wrong path | Auto-populated from subcarrier count: 128 subcarriers → data/2g4_ht40 |
| Model upload 422 error | Introduced ModelUploadRequest(BaseModel) with file_b64 and dest fields |
| btoa stack overflow on large files | Chunked loop reading 8192 bytes at a time |
| Train backend dropdown crashes | Removed invalid options; replaced with four valid backends: cnn, mlp, svm, variance |
| Sub baseline checkbox dead | Single-node inference path was hardcoding image_baseline=None; conditional baseline construction |
| WebSocket stale closure on reconnect | connectRef = useRef updated via useEffect([connect]); exponential backoff 2s→30s |

**Dead code removed:**
- CalibrationHealth.tsx — 68 lines, imported nowhere
- alertActive state — returned from useWaveTrace.ts but never consumed
- Antennas UI field — hardcoded to 2 internally; removed from Controls.tsx grid

### 10.2 New Features

**Devices tab:**
- web/device_ctl.py (DeviceHub): list_ports, serial monitor (pyserial), flash, pi-ssh
- /ws/device WebSocket (separate from pipeline /ws/logs)
- flash.sh: NO_MONITOR=1 drops blocking monitor; CLEAN=1 wipes sdkconfig+build for full rebuild

**Antenna power panel fixed:**
- Was: single bar "Antenna 1" flickering to whichever node's packet arrived last; arbitrary ×50 scale
- Fix: FrameSnooper tracks latest magnitude per node_id; backend emits per-node array + node_ids; UI renders one stable bar per board labeled "Node N", normalized to strongest node

**DevicePanel structured arg schema:**
- Previous: single freeform scriptArgs text input with factual errors (e.g. asserted mesh_verify.py took --port; it takes positional arguments)
- New: per-option typed fields — number inputs, text inputs, checkboxes for Boolean flags; live command preview string; empty values dropped automatically

**DecisionContribution panel:**
- Repurposed to show live per-class confidence (P(empty), P(present), etc.)
- Verified end-to-end: contribution: {'empty': 1.0, 'present': 0.0}

**Event channel (pipeline_done):**
- Replaced regex log-scraping (DONE_PATTERN) with explicit {"event": "pipeline_done"} emitted from streamer.py finally blocks
- Frontend checks data.event === 'pipeline_done' before treating message as inference result

**Feature surface additions (2026-06-25):**

| Feature | What changed |
|---|---|
| σ²[p] PDF histogram in Litmus Card | json_hist(clear, weapon, bins=20) added to experiments/weapon_litmus.py; SigmaHist SVG component (220×52px, teal/red overlay) in WeaponLitmus.tsx |
| subtract_ic_baseline badge | Amber badge in TrainingDashboard.tsx when model was trained with background subtraction |
| Carry axis in confusion-matrix selector | buildMatrix(logo, classCounts, axis?) accepts optional axis; availableAxes auto-detected from metrics |
| Missing-rate per link in NodeHealth | loss_pct = (target_hz - hz) / target_hz × 100; red when >20%, amber when >5% |
| Litmus card section | Moved to own "Weapon Litmus — Offline" section in App.tsx |

**Calibration auto-detect:**
- /api/calib/info?path= endpoint: reads meta.json, maps to bandwidth label
- Controls.tsx shows read-only calibBadge (e.g. "HT40 · 128 subcarriers") instead of a manual numeric input

**Cumulative weapon pool training:**
- start_training_managed now globs dataset_path subdirs containing X_features.npy
- Dataset Path text input added to Train settings section in Controls.tsx

### 10.3 Security Audit

**RCE via joblib.load() (high severity):**
- app.py:252–256: fusion_weights(path: str) GET endpoint deserializes any path via joblib (pickle); combined with allow_origins=["*"]
- Fix: _safe_output_path() resolves and confines to output/; rejects absolute paths and .. escapes

**Arbitrary file write → chained RCE (high severity):**
- app.py:264–270: writes base64 to any dest path, even creating dirs; chains with above (write a model then load it = RCE)
- Fix: same _safe_output_path() applied to model_upload dest

**app.py:~287 model_weights loads arbitrary client model path via mode_session — same RCE class; NOT YET FIXED**

**BackgroundTasks dead handle (low severity):**
- runner_task = background_tasks.add_task(run_blocking) returns None; stop path was dead
- Fix: asyncio.create_task(asyncio.to_thread(run_blocking)); stored handle; joined in stop with 2s timeout

**Bare except swallowing CancelledError (low severity):**
- 5 sites in broadcast loops using bare except: — swallows CancelledError and KeyboardInterrupt
- Fix: changed to except Exception:

**Verdict type unsafe (medium severity):**
- useWaveTrace.ts:107–112: JSON.parse(...) untyped; control events stored as verdict → UI reads verdict.conf/.t → undefined/NaN
- Fix: control events early-return; only true verdicts call setVerdict, cast to InferenceResult

**Per-frame canvas allocation (medium severity):**
- Spectrogram.tsx:21–27: new canvas and createImageData(W,K) allocated inside useEffect([data]) on every frame
- Fix: offscreen canvas + ImageData held in refs; reallocated only when W/K changes; Uint8ClampedArray updated in-place each frame

---

## 11. Data Management

### 11.1 Directory Taxonomy

Data is split by capture profile: data/2g4_ht20/, data/2g4_ht40/, data/5g_ht40/, data/5g_ht80/. All scripts take --root.

**Final clean state (as of 2026-06-25):**
- data/2g4_ht20/: baseline_raw/ cal/ sess_*/ ds_*/ model_presence/ model_weapon/ weapon_rec/ weapon_ds/
- data/2g4_ht40/: baseline_raw/ cal/ sess_0/ ds_0/ model_weapon/ weapon_rec/ weapon_ds/ ui/
- data/5g_ht40/, data/5g_ht80/: empty (future Pi/Nexmon captures)
- data/_archive_20260625/: all archived/superseded data

**Root-level strays cleaned up (2026-06-25):**
```
MOVE  data/baseline_raw/             → data/2g4_ht40/baseline_raw/
MOVE  data/ds_0/                     → data/2g4_ht40/ds_0/
MOVE  data/sess_0/                   → data/2g4_ht40/sess_0/
DEL   data/model/                    (empty dir)
ARCH  data/2g4_ht40/cal/             → data/_archive_20260625/cal_incomplete/ (no meta.json)
MOVE  data/cal/                      → data/2g4_ht40/cal/ (complete; has meta.json)
ARCH  data/2g4_ht40/model_weapon_bg/ → data/_archive_20260625/model_weapon_bg/ (LOGO 0.469 < majority 0.527)
ARCH  data/2g4_ht40/weapon_ds_bg/    → data/_archive_20260625/weapon_ds_bg/
ARCH  data/2g4_ht20/model_presence/  → data/_archive_20260625/model_presence_flat_bad/ (LOGO 0.470)
REN   data/2g4_ht20/model/           → data/2g4_ht20/model_presence/
```

### 11.2 Data Hygiene Issues

**Pool contamination (2026-06-25):**
- Old desk-object-only data (weapon_onDesk: metal on desk, nobody present) silently merged with body-worn data (ilker4node_metal: metal worn on body, person present) — completely different physical conditions labeled identically
- This produced AUC=0.566 which looked like a valid result but was an artifact

**Quarantined datasets (2026-06-26):**
- weapon_onDesk (36 entries) → _archive/weapon_ds_onDesk_rebuilt/ — "metal on desk, nobody present" pooled with "person, no weapon" is a person-presence detector, not a weapon detector
- ilker4node_metal (72 entries) → _archive/weapon_ds_metal_lostcal/ — correct Jun-24 17:16 cal was overwritten by Jun-25 13:08 recal; no honest eval possible

### 11.3 Baseline Environments

Weapon datasets were collected in three physically distinct environments (different rooms/days → different channel baselines). These cannot be combined; each must be evaluated independently.

| Dataset | Cal date | Cal notes |
|---|---|---|
| ilker_hand (HT20) | Jun-18 | flat cal, same room |
| p0_na body-worn (HT40) | Jun-23 16:15 | 3 nodes |
| ilker_metal_walk (HT40/ui) | Jun-25 13:08 | 4 nodes |

**Final clean data layout (as of 2026-06-26):**
- data/2g4_ht20/weapon_ds/ — ilker_hand × 3 sess, Jun-18 cal (72 entries)
- data/2g4_ht40/weapon_ds/ — p0_na body-worn × 3 sess, Jun-23 cal (36 entries)
- data/2g4_ht40/ui/weapon_ds/ — ilker_metal_walk × 4 sess, Jun-25 13:08 cal (96 entries)

### 11.4 Binary Parser Fixes

**uint32 ts_us wrap:**
- uint32 ts_us wraps every ~71.6 min; a batch straddling wrap → ~71 min apparent time jump
- Fix: mask subtraction & 0xFFFFFFFF; regression test added

**Per-record mac_str on hot path:**
- _iter_bin_records yielded a formatted 17-char MAC string per record — allocation on every record
- Fix: yield raw 6 MAC bytes; tx_mac filter compares bytes; parse_batch_links formats only the 2-octet bucket key

**Capture loop deadline:**
- Capture loops could hang indefinitely if one node is quiet
- Fix: max_capture_s=60 wall-clock deadline; breaks with WARN listing which nodes fell short

---

## 12. Open Items and Blockers

### 12.1 Critical (blocking next experiment)

- **NLOS litmus gate not cleared.** Antennas are aimed toward center, but AUC >= 0.65 on at least one link has not been verified on new geometry data. No ML training should happen until this gate clears.
- **app.py:~287 security hole.** model_weights loads an arbitrary client model path via mode_session — same RCE class as fusion_weights; not yet fixed.

### 12.2 Known Bugs

- **AGC<30 filter on training side:** training logic needs to drop data sequences where AGC<30 (gain-skip corruption); not yet implemented
- **Camera under FastAPI asyncio:** still requires manual `python3 -c "import cv2; cv2.VideoCapture(0)"` workaround; correct fix (ffmpeg subprocess) identified but not implemented
- **col_spans still visible in Collect tab** when camera_collect=true; should be hidden
- **ntp_server.py alignment:** must be running at the start of every session; whether it materially improves cross-node timestamp alignment in practice has not been measured

### 12.3 Designed Only (not tested)

- **Pi 5 / 5 GHz HT80:** community Nexmon patch required; architecture designed (firmware/pi/config.py, illuminator via Mac ping, wire format v3), zero hardware testing done
- **Camera-supervised heatmap:** all code exists; collect_camera.py --train has never been run end-to-end; no heatmap.joblib has ever been produced; fallback to _occupancy_fallback() is correct and works, but the trained-model path is unverified

### 12.4 Missing Features

- No custom YOLO weights field in Controls (needed for gun detection beyond COCO knife class 43)
- No live per-link weapon direction breakdown in dashboard; run_weapon.py emits per-link probabilities but only the fused verdict is shown
- Bounded Pi publisher queue: collections.deque(maxlen=300) FIFO cap not yet applied; prevents unbounded memory growth if UDP drain falls behind Pi transmit rate
- Host-side reader decoupling: multi-threaded socket ingestion undecided; partially mitigated by SO_RCVBUF bump but not fully resolved
- Multi-person heatmap: VisionLabeler._detect() picks only highest-confidence person; second person's mask is silently dropped

---