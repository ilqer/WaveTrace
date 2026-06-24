# `tests/`

Automated test suite. All tests run on synthetic or recorded data — no hardware required.

```bash
pytest tests/ -q    # run everything; expect ~271 tests all passing
```

## What is tested

| File(s) | Coverage |
|---|---|
| `TestCore.py`, `TestFrameParser.py` | C++ `CsiFrame` type and UDP datagram parsing |
| `TestPreprocess.py`, `TestFeatures.py` | Conjugate multiply, Hampel filter, phase unwrap, gain lock, NBVI, feature extraction |
| `TestGainLock.py` | Gain lock apply / relock / coefficient-of-variation fallback |
| `TestCalibration.py` | Save/load calibration, NBVI subcarrier selection |
| `TestResample.py`, `TestSubcarrierSelect.py` | Timing-jitter resampler, subcarrier ranking |
| `TestSpectrogram.py` | Spectrogram builder shape and values |
| `TestRecognition.py`, `TestPipeline.py` | Presence model train + infer on synthetic data |
| `TestWeapon.py`, `TestWeaponPipeline.py` | Weapon head train + σ²[p] baseline |
| `TestWeaponBgSubtract.py`, `TestWeaponConfound.py`, `TestWeaponLitmus.py` | Weapon-specific edge cases |
| `TestCount.py` | People-count model train + infer |
| `TestCir.py` | CIR super-resolution (L1/ISTA) |
| `TestUdpSource.py`, `TestMeshLinks.py` | UDP source and multi-node link handling |
| `TestMultiNode.py`, `TestNodeWeights.py`, `TestLinkVoter.py` | Per-node voting and per-link vote weights |
| `TestFusion.py` | Feature-level multi-node fusion |
| `TestGroundTruth.py`, `TestSegmentationLabeler.py`, `TestVisionLabeler.py` | Camera labeler and dataset builder |
| `TestPresenceSegment.py` | Motion-segment gate used inside the voter |
| `TestLocalize.py`, `TestFrontendExt.py` | Localization scaffold and frontend extensions |
| `TestRegression.py` | End-to-end regression: full pipeline on a synthetic recording |
| `TestPiPublisher.py` | Round-trips the Pi wire format through `wavetrace/Source.py` |
| `conftest.py` | Shared fixtures (synthetic CSI, synthetic recordings) |
