"""Independent live WEAPON detection — every (tx->rx) link served through its RX node's cal + weapon
head (inter-carrier features), fused into one armed/clear verdict. Standalone from run_live_mesh /
run_count (imports only library code), reads the weapon models from data/model_weapon.

Each link's RX-node WeaponHead emits P(weapon); LinkVoter blends them weighted by static reliability
(per-node LOGO accuracy via accuracyWeights) x live decision margin. A node validated at/below chance
gets weight 0 and drops out.

    .venv/bin/python scripts/run_weapon.py
"""

import argparse
import collections
import glob
import json
import os
import socket
import time

import numpy as np

from wavetrace.Source import parseBatchLinks, resampleUniform, bindUdp
from wavetrace.Calibration import loadCalibration
from wavetrace.Frontend import iterWindows
from wavetrace.recognition import modeSession
from wavetrace.recognition.Link import LinkVoter, accuracyWeights
from wavetrace.Cli import _servingPlan


def _minWidth(result):
    """Subcarrier width the calibration needs = highest index it references + 1."""
    idx = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(idx)


def _logoAcc(metrics_path):
    """A node head's honest (LOGO) accuracy — session axis preferred, subject fallback; None if absent."""
    try:
        with open(metrics_path) as f:
            logo = json.load(f).get("logo", {})
    except (OSError, ValueError):
        return None
    for axis in ("session", "subject"):
        acc = logo.get(axis, {}).get("accuracy")
        if acc is not None:
            return float(acc)
    return None


def dwellProbaDetailed(frames, fs, m):
    """SINGLE SOURCE OF TRUTH for one link's serving math, shared by run_weapon and the web streamer
    (P5: don't reimplement the vote loop twice). `m` is a serving entry — the dict from
    loadWeaponLinks OR a mesh node dict; both carry result/lock/cfg/intercarrier/pick/session and an
    optional ic_baseline.

    Resamples to fs, windows the dwell, and returns the TEMPORAL VOTE — the mean class-proba over every
    window in the buffer (BUFFER_S of history), not just the last (diagnosis CAUSE 5C: Zhou's
    per-crossing aggregation lifted single-window 51% -> 93%). Soft mean (not hard majority) so it
    composes with the soft cross-link LinkVoter. Also returns the LAST window's (image, features, ic)
    for the web spectrogram, and the window count. Returns (None, None, None, None, 0) if no full
    window fits. ic_baseline (Item 10/CAUSE 2B): subtracted from the IC path when this head was trained
    that way — MUST match training (carried on the entry)."""
    res = resampleUniform(frames, fs)
    cfg = m["cfg"]
    if len(res) < cfg.window:
        return None, None, None, None, 0
    probas = []
    image = features = ic = None
    for _t, features, image, ic in iterWindows(
        res, m["result"].subcarriers, m["lock"],
        window=cfg.window, hop=cfg.hop, intercarrier=m["intercarrier"],
        image_subcarriers=m["result"].image_subcarriers, ic_baseline=m.get("ic_baseline"),
    ):
        probas.append(m["session"].predictProbaWindow(m["pick"](features, image, ic)))
    if not probas:
        return None, None, None, None, 0
    return np.mean(probas, axis=0), image, features, ic, len(probas)  # temporal (soft) vote


def loadWeaponLinks(cal_root, model_root):
    """Discover weapon heads -> {(tx_tag|None, rx_node): entry}. Two layouts, auto-detected per node:
      * PER-LINK (WEAPON_NLOS_PLAN §4): model_weapon/node<id>/link<tag>/model.joblib -> one entry per
        directed (tx->rx) view, keyed (tag, nid). The weapon signal is per-direction; this lets each
        NLOS-scatter link vote on its own merit instead of being pooled (and sign-flipped) per node.
      * PER-NODE (fallback): model_weapon/node<id>/model.joblib -> a single (None, nid) entry that
        serves ANY tx into that node. Used when a node wasn't trained --per-link.
    Calibration is per RX node (shared across that node's link heads). Each entry carries a static
    LinkVoter `weight` from its OWN LOGO accuracy (accuracyWeights; chance->0 so a weak direction
    drops out without being explicitly removed) and the serving plan matching how it was trained."""
    entries = {}
    accs = {}
    for model_dir in sorted(glob.glob(os.path.join(model_root, "node*"))):
        base = os.path.basename(model_dir)
        if not base[len("node"):].isdigit():
            continue
        nid = int(base[len("node"):])
        calDir = os.path.join(cal_root, base)
        if not os.path.isdir(calDir):
            continue
        linkModels = sorted(glob.glob(os.path.join(model_dir, "link*", "model.joblib")))
        if linkModels:  # per-link: tag from the link<tag> dir name
            found = [(os.path.basename(os.path.dirname(p))[len("link"):], p) for p in linkModels]
        elif os.path.exists(os.path.join(model_dir, "model.joblib")):
            found = [(None, os.path.join(model_dir, "model.joblib"))]  # per-node fallback
        else:
            continue
        result, gainLock = loadCalibration(calDir)
        for tag, modelPath in found:
            session = modeSession("weapon", modelPath)
            applyLock, intercarrier, pick = _servingPlan("weapon", session.head)
            classes = list(session.head.classes_)
            # rebuilt from the node's calibration so train/serve subtract the same baseline (Item 10).
            icBaseline = (result.baseline_mag
                           if getattr(session.head.config, "subtract_ic_baseline", False) else None)
            key = (tag, nid)
            entries[key] = dict(
                result=result, lock=gainLock if applyLock else None,
                intercarrier=intercarrier, pick=pick, session=session, cfg=session.head.config,
                min_width=_minWidth(result), weapon_i=classes.index(1) if 1 in classes else -1,
                ic_baseline=icBaseline,
            )
            accs[key] = _logoAcc(os.path.join(os.path.dirname(modelPath), "metrics.json"))
    weights = accuracyWeights({k: a for k, a in accs.items() if a is not None})
    for key in entries:
        entries[key]["weight"] = weights.get(key, 1.0)
    orders = {tuple(int(c) for c in e["session"].head.classes_) for e in entries.values()}
    if len(orders) > 1:
        raise ValueError(f"weapon heads disagree on class ordering {orders}; retrain consistently")
    return entries


def _entryFor(entries, buf_key):
    """Match a live buffer key (tx_short, rx_node) to a serving entry: the per-link (tag, nid) head
    first (tx_short '4f:9c' -> tag '4f9c'), then the per-node fallback (None, nid). None if neither."""
    txShort, nid = buf_key
    return entries.get((txShort.replace(":", ""), nid)) or entries.get((None, nid))


def _linkHealth(frames):
    """(delivered_hz, missing_fraction) from frame timestamps — the per-link Missing-Rate metric
    (diagnosis C9b: measure DELIVERED rate, not configured). Median inter-arrival is the nominal
    period; a gap of ~k periods counts k-1 missing frames. O(n). (0,0) if too few frames."""
    if len(frames) < 3:
        return 0.0, 0.0
    ts = np.array([f.timestamp for f in frames], dtype=np.float64)
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0, 0.0
    span = ts[-1] - ts[0]
    hz = (len(frames) - 1) / span if span > 0 else 0.0
    med = float(np.median(dt))
    if med <= 0:
        return hz, 0.0
    missing = float(np.clip(np.round(dt / med) - 1.0, 0.0, None).sum())
    expected = missing + dt.size
    return hz, (missing / expected if expected > 0 else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Live ALL-PAIRS weapon detection (per-link, per-RX-node cal+head).")
    parser.add_argument("--port", type=int, default=9876, help="UDP port (default: 9876)")
    parser.add_argument("--root", default="data",
                        help="Capture-profile root, e.g. data/2g4_ht40 or data/5g_ht80 (default: data)")
    parser.add_argument("--cal", default=None, help="Calibration root (default: <root>/cal)")
    parser.add_argument("--model", default=None, help="Weapon model root (default: <root>/model_weapon)")
    args = parser.parse_args()
    if args.cal is None:
        args.cal = f"{args.root}/cal"
    if args.model is None:
        args.model = f"{args.root}/model_weapon"

    TARGET_FS = 100.0      # uniform resample grid; MUST match collect_weapon.TARGET_FS
    CHUNK_S = 1.5          # fuse + print at this cadence
    LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
    BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing

    entries = loadWeaponLinks(args.cal, args.model)
    if not entries:
        print(f"[ERROR] no weapon heads under {args.model}/node*/[link*/]model.joblib with a matching "
              f"{args.cal}/node*/. Run collect_baseline.py then collect_weapon.py first.")
        return
    weaponI = next(iter(entries.values()))["weapon_i"]  # ordering validated equal in loadWeaponLinks

    buffers = collections.defaultdict(collections.deque)  # keyed by (tx_short, rx_node)
    lastSeen = {}
    linkIds = {}

    sock = bindUdp(args.port, timeout=0.5)
    perLink = any(tag is not None for tag, _ in entries)
    def _wlabel(key):
        tag, nid = key
        return f"{tag}->{nid}" if tag is not None else f"*->{nid}"
    wsummary = "  ".join(f"{_wlabel(k)}:w={entries[k]['weight']:.2f}" for k in sorted(entries))
    print(f"WEAPON detection on udp/{args.port} (fs={TARGET_FS:g}Hz, "
          f"{'per-link' if perLink else 'per-node'} heads; vote weights {wsummary}). Ctrl+C to stop.\n")

    nextFuse = time.time() + CHUNK_S
    try:
        while True:
            now = time.time()
            try:
                payload, _ = sock.recvfrom(65535)
                for key, frames in parseBatchLinks(payload).items():
                    m = _entryFor(entries, key)  # key=(tx_short, rx_node) -> per-link, then per-node
                    if m is not None and frames[0].num_subcarriers >= m["min_width"]:
                        buffers[key].extend(frames)
                        lastSeen[key] = now
                        linkIds.setdefault(key, len(linkIds))
            except socket.timeout:
                pass

            if now < nextFuse:
                continue
            nextFuse = now + CHUNK_S

            for buf in buffers.values():
                if buf:
                    cutoff = buf[-1].timestamp - BUFFER_S
                    while buf and buf[0].timestamp < cutoff:
                        buf.popleft()

            # static per-link reliability x live margin (LinkVoter multiplies them); uniform fallback.
            linkStatic = {lid: _entryFor(entries, key)["weight"] for key, lid in linkIds.items()}
            static = linkStatic if any(w > 0 for w in linkStatic.values()) else None
            voter = LinkVoter(static)
            breakdown = []
            for key in sorted(buffers):
                if now - lastSeen.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                m = _entryFor(entries, key)
                proba, *_ = dwellProbaDetailed(list(buffers[key]), TARGET_FS, m)
                if proba is None:
                    continue
                wi = m["weapon_i"]
                pWeapon = float(proba[wi]) if wi >= 0 else 0.0
                quality = abs(pWeapon - 0.5) * 2.0  # decision margin -> 0 (unsure) .. 1 (confident)
                voter.add(linkIds[key], proba, quality=quality)
                hz, miss = _linkHealth(buffers[key])  # delivered rate + missing-frame fraction (C9b)
                tail = f"@{hz:.0f}Hz" + (f"!{miss:.0%}drop" if miss > 0.1 else "")
                breakdown.append(f"{key[0]}->{key[1]}:{pWeapon:.2f}{tail}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            try:
                _cls, blended = voter.finalize()
            except ValueError:
                print("\r(live links present, but all from chance-level nodes)   ", end="", flush=True)
                continue
            pWeapon = float(blended[weaponI]) if weaponI >= 0 else 0.0
            label = "WEAPON" if pWeapon >= 0.5 else "clear "
            bar = "#" * int(pWeapon * 20)
            print(f"{label}  P {pWeapon:0.2f}  {bar:<20}  [{len(breakdown)} links] "
                  + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
