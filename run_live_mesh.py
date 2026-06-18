"""Live ALL-PAIRS presence: fuse every (tx->rx) link with ONE shared head via LinkVoter.

Splits the batched-UDP mesh stream by transmitter (parse_batch_links), resamples each (tx,rx) link
onto a uniform grid, windows each through the SAME trained presence head, and blends the per-link
probabilities with LinkVoter. Live quality = the head's decision margin, so a blocked/confused link
down-weights itself (plan §2.9.3 blockage recovery). Dynamic ring: links appear and vanish; the vote
uses whatever is currently live. Reuses the exact serving path the trainer used. Ctrl+C to stop.

    .venv/bin/python run_live_mesh.py
"""

import collections
import socket
import time

import numpy as np

from wavetrace.Source import parse_batch_links, resample_uniform
from wavetrace.Calibration import load_calibration
from wavetrace.Frontend import iter_windows
from wavetrace.recognition import mode_session
from wavetrace.recognition.Link import LinkVoter
from wavetrace.Cli import _serving_plan

UDP_PORT = 9876
CAL = "data/cal"
MODEL = "data/model_presence/model.joblib"
MODE = "presence"
TARGET_FS = 100.0      # uniform resample grid (the locked live cadence the collect scripts assume)
CHUNK_S = 1.5          # fuse + print at this cadence
LINK_TIMEOUT_S = 3.0   # drop a link from the vote if unheard this long
BUFFER_S = 3.0         # per-link rolling history kept for resampling/windowing


def _min_width(result):
    """Subcarrier width the calibration needs = highest index it references + 1."""
    idx = [int(i) for i in list(result.subcarriers) + list(result.image_subcarriers)]
    return 1 + max(idx)


def _last_window_proba(frames, fs, result, gain_lock, cfg, intercarrier, pick, session):
    """Resample one link's frames to fs, window them, return the LAST window's class-proba or None
    (None when there aren't enough frames to fill a window)."""
    res = resample_uniform(frames, fs)
    if len(res) < cfg.window:
        return None
    last = None
    for _t, features, image, ic in iter_windows(
        res, result.subcarriers, gain_lock,
        window=cfg.window, hop=cfg.hop, intercarrier=intercarrier,
        image_subcarriers=result.image_subcarriers,
    ):
        last = session.predict_proba_window(pick(features, image, ic))
    return last


def main():
    result, gain_lock = load_calibration(CAL)
    min_width = _min_width(result)
    session = mode_session(MODE, MODEL)
    apply_lock, intercarrier, pick = _serving_plan(MODE, session.head)
    cfg = session.head.config
    fs = TARGET_FS
    lock = gain_lock if apply_lock else None

    classes = list(session.head.classes_)
    present_i = classes.index(1) if 1 in classes else -1

    # per-link rolling frame buffers + last-heard wall time, and a stable int id for LinkVoter
    buffers: dict = collections.defaultdict(collections.deque)
    last_seen: dict = {}
    link_ids: dict = {}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.5)
    print(f"ALL-PAIRS presence on udp/{UDP_PORT} (fs={fs:g}Hz, window={cfg.window}). "
          f"move in and out of the links. Ctrl+C to stop.\n")

    next_fuse = time.time() + CHUNK_S
    try:
        while True:
            now = time.time()
            try:
                payload, _ = sock.recvfrom(65535)
                for key, frames in parse_batch_links(payload).items():
                    if frames[0].num_subcarriers < min_width:
                        continue  # too-narrow link (beacon/legacy) — cal would index off the end
                    buffers[key].extend(frames)
                    last_seen[key] = now
                    link_ids.setdefault(key, len(link_ids))
            except socket.timeout:
                pass

            if now < next_fuse:
                continue
            next_fuse = now + CHUNK_S

            # trim each buffer to the last BUFFER_S seconds (by frame timestamp)
            for key, buf in buffers.items():
                if buf:
                    cutoff = buf[-1].timestamp - BUFFER_S
                    while buf and buf[0].timestamp < cutoff:
                        buf.popleft()

            voter = LinkVoter()  # shared head -> static prior uniform; weight = live quality only
            breakdown = []
            for key in sorted(buffers):
                if now - last_seen.get(key, 0) > LINK_TIMEOUT_S or len(buffers[key]) < 2:
                    continue
                proba = _last_window_proba(list(buffers[key]), fs, result, lock, cfg,
                                           intercarrier, pick, session)
                if proba is None:
                    continue
                p_present = float(proba[present_i]) if present_i >= 0 else 0.0
                quality = abs(p_present - 0.5) * 2.0  # decision margin -> 0 (unsure) .. 1 (confident)
                voter.add(link_ids[key], proba, quality=quality)
                breakdown.append(f"{key[0]}->{key[1]}:{p_present:.2f}")

            if not breakdown:
                print("\r(no live links with a full window yet)            ", end="", flush=True)
                continue
            _cls, blended = voter.finalize()
            p_present = float(blended[present_i]) if present_i >= 0 else 0.0
            label = "PRESENT" if p_present >= 0.5 else "absent "
            bar = "#" * int(p_present * 20)
            print(f"{label}  P {p_present:0.2f}  {bar:<20}  [{len(breakdown)} links] "
                  + " ".join(breakdown))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
