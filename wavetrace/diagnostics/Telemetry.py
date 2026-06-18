"""Per-node link health + dataset diagnostics. All numpy, all off the hot path.

A `NodeHealthMeter` ingests the same CsiFrame stream the front-end sees (tee it through), tracks a
rolling window per node_id, and emits a health dict per node on demand. The cluster_sync / drift /
separation helpers are one-shot computations for the UI's calibration & training panels.

node_id < 100 => 2.4 GHz (ESP32), node_id >= 100 => 5 GHz (nexmon Pi).
"""

import time
from collections import defaultdict, deque

import numpy as np


class NodeHealthMeter:
    """Rolling per-node link health. Feed every CsiFrame via `observe`; read `snapshot()` whenever
    the UI wants an update (e.g. once per second). O(1) per frame, O(nodes·win) per snapshot."""

    def __init__(self, *, target_hz=100.0, min_hz_frac=0.8, win=200, gain_ref_by_node=None):
        self.target_hz = float(target_hz)
        self.min_hz = float(target_hz) * float(min_hz_frac)
        self.win = int(win)
        self._recv = defaultdict(lambda: deque(maxlen=self.win))  # wall recv times
        self._amp = defaultdict(lambda: deque(maxlen=self.win))   # mean |H| per frame
        self._cv = defaultdict(lambda: deque(maxlen=self.win))    # cross-subcarrier CV
        self._count = defaultdict(int)
        self._last_ts = {}
        self._gain_ref = dict(gain_ref_by_node or {})  # node_id -> calib mean amplitude
        self._subc = {}    # node_id -> NBVI subcarrier indices (for display)

    def set_reference(self, node_id, ref_scale=None, subcarriers=None):
        """Record a node's calibration reference amplitude + NBVI subcarriers for display."""
        if ref_scale is not None:
            self._gain_ref[int(node_id)] = float(ref_scale)
        if subcarriers is not None:
            self._subc[int(node_id)] = list(subcarriers)

    def observe(self, frame) -> None:
        """One CsiFrame. O(A·S)."""
        nid = int(frame.node_id)
        now = time.time()
        g = np.asarray(frame.grid)               # (A, S) complex
        amp = np.abs(g)                          # (A, S)
        sub_mean = float(amp.mean())             # bulk level
        # cross-subcarrier CV (std/mean over per-antenna-averaged subcarriers) = turbulence/motion
        per_sub = amp.mean(axis=0)               # (S,) antenna-averaged
        cv = float(per_sub.std() / per_sub.mean()) if per_sub.mean() > 0 else 0.0
        self._recv[nid].append(now)
        self._amp[nid].append(sub_mean)
        self._cv[nid].append(cv)
        self._count[nid] += 1
        self._last_ts[nid] = float(frame.timestamp)

    def _hz(self, nid) -> float:
        rs = self._recv[nid]
        if len(rs) < 2:
            return 0.0
        dt = rs[-1] - rs[0]
        return (len(rs) - 1) / dt if dt > 0 else 0.0

    def snapshot(self) -> list[dict]:
        """Per-node health dict list. Call on UI cadence (~1 Hz), not per frame."""
        out = []
        for nid in sorted(self._recv.keys()):
            amp = np.asarray(self._amp[nid], dtype=np.float64)
            mean_amp = float(amp.mean()) if amp.size else 0.0
            floor = float(np.percentile(amp, 5)) if amp.size >= 20 else mean_amp * 0.1
            snr_db = float(20.0 * np.log10(mean_amp / floor)) if floor > 1e-12 else 0.0
            hz = self._hz(nid)
            ref = self._gain_ref.get(nid)
            gain_drift = round(mean_amp / ref, 3) if (ref and ref > 0) else None
            out.append({
                "node_id": nid,
                "band": "5GHz" if nid >= 100 else "2.4GHz",
                "hz": round(hz, 1),
                "hz_ok": hz >= self.min_hz,
                "frames": self._count[nid],
                "mean_amp": round(mean_amp, 4),
                "snr_db": round(snr_db, 1),
                "cv": round(float(np.mean(list(self._cv[nid]))) if self._cv[nid] else 0.0, 4),
                "gain_drift": gain_drift,
                "subcarriers": self._subc.get(nid, []),
                "last_ts": round(self._last_ts.get(nid, 0.0), 4),
            })
        return out


def cluster_sync(meter: NodeHealthMeter) -> dict:
    """Time-alignment health: spread of the latest per-node frame timestamps.
    Large spread => nodes are de-synced => stacking will drop windows. O(nodes)."""
    last = {nid: meter._last_ts.get(nid, 0.0) for nid in meter._recv.keys()}
    if len(last) < 2:
        return {"spread_s": 0.0, "ok": True, "per_node_last": last}
    vals = np.asarray(list(last.values()))
    spread = float(vals.max() - vals.min())
    return {"spread_s": round(spread, 4), "ok": spread <= 0.05, "per_node_last": last}


def baseline_drift(calib_result, recent_frames) -> dict:
    """How far the current quiet room has drifted from the calibrated baseline.
    >~20% mean ratio suggests recalibration. O(F·S)."""
    if not recent_frames:
        return {"mean_ratio": None, "max_ratio": None, "recalibrate": False}
    amps = np.stack([np.abs(np.asarray(f.grid)).mean(axis=0) for f in recent_frames])  # (F, S)
    cur = amps.mean(axis=0)
    base = np.asarray(calib_result.baseline_mag, dtype=np.float64)
    n = min(cur.size, base.size)
    ratio = cur[:n] / np.where(base[:n] > 1e-9, base[:n], 1e-9)
    mean_r = float(np.mean(ratio))
    max_r = float(np.max(np.abs(ratio - 1.0)) + 1.0)
    return {"mean_ratio": round(mean_r, 3), "max_ratio": round(max_r, 3),
            "recalibrate": abs(mean_r - 1.0) > 0.2}


def feature_separation(x_intercarrier, y, *, variance_col=9) -> dict:
    """Is the σ²[p] weapon signature separable? Returns two histograms (weapon vs none) of the
    inter-carrier variance feature + a rank-AUC separability score. O(n log n).

    auc = 0.5 means no separation, 1.0 means perfect. Threshold 0.65 is a rough 'worth training' gate."""
    x = np.asarray(x_intercarrier, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 2 or x.shape[0] != y.size:
        return {"error": "shape"}
    feat = x[:, variance_col]
    pos, neg = feat[y == 1], feat[y == 0]
    if pos.size == 0 or neg.size == 0:
        return {"error": "single_class"}
    # rank-AUC (Mann-Whitney U / n*m): P(random positive < random negative)
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    auc = (ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    auc = float(max(auc, 1 - auc))  # fold: 0.5=random, 1.0=perfect
    edges = np.linspace(float(feat.min()), float(feat.max()), 33)
    h_pos, _ = np.histogram(pos, bins=edges)
    h_neg, _ = np.histogram(neg, bins=edges)
    return {
        "auc": round(auc, 3),
        "edges": edges.tolist(),
        "weapon_hist": h_pos.tolist(),
        "none_hist": h_neg.tolist(),
        "separable": auc > 0.65,
    }


def dataset_report(dataset) -> dict:
    """4-quadrant style summary of a built dataset for the training UI panel."""
    y = np.asarray(dataset.y)
    classes, counts = np.unique(y, return_counts=True)
    return {
        "n_samples": int(y.size),
        "distribution": {str(int(c)): int(n) for c, n in zip(classes, counts)},
        "class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
        "balance": round(float(counts.min() / counts.max()), 3) if counts.size > 1 else 0.0,
        "sessions": sorted({str(s) for s in dataset.session_ids}),
        "subjects": sorted({str(s) for s in dataset.subject_ids}),
        "sync_error": dataset.meta.get("sync_error", {}),
        "num_nodes": dataset.meta.get("num_nodes", 1),
        "K": dataset.meta.get("K"),
        "K_img": dataset.meta.get("K_img"),
    }
