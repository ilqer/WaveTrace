"""Weapon model bake-off on the CLEANED latest-only pool (ilker4node_metal, onDesk quarantined).

Reports leave-one-SESSION-out (LOGO) for three families:
  A. PER-NODE pooled head (a node's 3 links pooled)  x {variance, mlp, svm on ic27 ; cnn on image}
  B. PER-LINK head (each tx->rx direction on its own)  x {variance, mlp on ic27}
  C. COMBINED 12-link multi-channel CNN — every link stacked as a CHANNEL: X=(n,12,K,window),
     so one CNN sees the whole mesh at once (NOT feature vectors glued side by side).

Only SESSION-LOGO is meaningful here: there is 1 subject and 1 carry pose, so subject/carry folds
degenerate and are skipped by _logo_metrics. Static-subject capture -> the per-link channels are
stationary, so the combined stack aligns the 12 links index-wise per (session,condition) (identical
100 Hz grid + hop); a 1-2 window phase slip is immaterial for a stationary signal.

    .venv/bin/python experiments/weapon_experiments.py --root data/2g4_ht40/ui
"""

import argparse
import collections
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from wavetrace.Config import ModelConfig
from wavetrace.groundtruth import load_dataset
from wavetrace.recognition.Weapon import WeaponHead
from wavetrace.recognition.Train import _logo_metrics, train_weapon

WINDOW, HOP, FS = 128, 32, 100.0


def _cfg(backend, k, **kw):
    return ModelConfig(stage="weapon", k=k, backend=backend, window=WINDOW, hop=HOP, **kw)


def _session_logo(X, y, sess, subj, make_head):
    """Run _logo_metrics and return only the session fold (the valid one) or None."""
    m = _logo_metrics(np.asarray(X), np.asarray(y), np.asarray(sess), np.asarray(subj), make_head)
    return m.get("session")


def _fmt(tag, n, s):
    if s is None:
        return f"   {tag:<26} n={n:<4} (LOGO needs >=2 sessions)"
    return (f"   {tag:<26} n={n:<4} LOGO={s['accuracy']:.3f}  maj={s['majority_accuracy']:.3f}  "
            f"TPR={s.get('tpr', 0):.3f}  FP={s.get('fp_rate', 0):.3f}  "
            f"{'BEATS' if s['accuracy'] > s['majority_accuracy'] + 1e-9 else 'below'} majority")


def _link_tag(d):
    b = os.path.basename(d)
    return b.split("_link")[-1] if "_link" in b else None


def discover(root):
    """{node -> {tag -> [ds dirs]}} for the latest ilker4node_metal pool, plus the 12-link order."""
    nodes = {}
    for nd in sorted(glob.glob(f"{root}/weapon_ds/node*")):
        nid = int(os.path.basename(nd)[4:])
        bytag = collections.defaultdict(list)
        for d in sorted(glob.glob(f"{nd}/*")):
            t = _link_tag(d)
            if t:
                bytag[t].append(d)
        if bytag:
            nodes[nid] = dict(bytag)
    links = sorted((nid, t) for nid in nodes for t in nodes[nid])
    return nodes, links


# ---- A. per-node pooled -------------------------------------------------------------------------
def exp_per_node(nodes, root):
    print("\n=== A. PER-NODE pooled heads (session-LOGO) ===")
    deploy = {}
    for nid in sorted(nodes):
        dirs = [d for t in nodes[nid] for d in nodes[nid][t]]
        dss = [load_dataset(d) for d in dirs]
        X_ic = np.concatenate([d.X_intercarrier for d in dss]).astype(np.float32)
        X_im = np.concatenate([d.X_image for d in dss]).astype(np.float32)
        y = np.concatenate([d.y for d in dss])
        sess = np.concatenate([d.session_ids for d in dss])
        subj = np.concatenate([d.subject_ids for d in dss])
        k = int(dss[0].meta["K"])
        print(f" node{nid}:")
        for name, backend, X in (("ic27/variance", "variance", X_ic),
                                 ("ic27/mlp", "mlp", X_ic),
                                 ("ic27/svm", "svm", X_ic),
                                 ("cnn/image", "cnn", X_im)):
            s = _session_logo(X, y, sess, subj, lambda b=backend, kk=k: WeaponHead(_cfg(b, kk)))
            print(_fmt(name, len(y), s))
        deploy[nid] = dirs  # for the canonical ic27 retrain
    return deploy


# ---- B. per-link --------------------------------------------------------------------------------
def exp_per_link(nodes):
    print("\n=== B. PER-LINK heads — each tx->rx direction (session-LOGO) ===")
    rows = []
    for nid in sorted(nodes):
        for tag in sorted(nodes[nid]):
            dss = [load_dataset(d) for d in nodes[nid][tag]]
            X_ic = np.concatenate([d.X_intercarrier for d in dss]).astype(np.float32)
            y = np.concatenate([d.y for d in dss])
            sess = np.concatenate([d.session_ids for d in dss])
            subj = np.concatenate([d.subject_ids for d in dss])
            k = int(dss[0].meta["K"])
            sv = _session_logo(X_ic, y, sess, subj, lambda kk=k: WeaponHead(_cfg("variance", kk)))
            sm = _session_logo(X_ic, y, sess, subj, lambda kk=k: WeaponHead(_cfg("mlp", kk)))
            print(_fmt(f"node{nid} link{tag} var", len(y), sv))
            print(_fmt(f"node{nid} link{tag} mlp", len(y), sm))
            if sv:
                rows.append((nid, tag, sv["accuracy"], sv["majority_accuracy"]))
    beats = [r for r in rows if r[2] > r[3] + 1e-9]
    print(f"\n   per-link (variance) links beating majority: {len(beats)}/{len(rows)}")
    for nid, tag, a, mj in sorted(beats, key=lambda r: -r[2]):
        print(f"      node{nid} link{tag}: {a:.3f} (maj {mj:.3f})")


# ---- C. combined 12-link multi-channel CNN ------------------------------------------------------
def build_aligned(nodes, links, field):
    """(n,12,*) X, y, session — each sample's 12 link-channels are aligned at the same window.
    `field` = "X_image" -> (n,12,K,window) for the combined CNN ; "X_intercarrier" -> (n,12,27) for
    the weighted fusion. Index-wise stack per (session,condition) trimmed to the min window count
    (static-subject, shared 100 Hz grid -> window k of every link is the same instant; any slip is
    immaterial for a stationary signal)."""
    chan = {lk: i for i, lk in enumerate(links)}          # (node,tag) -> channel index 0..11
    sessions = sorted({os.path.basename(d).split("_metal_")[1].split("_")[0]
                       for nid in nodes for t in nodes[nid] for d in nodes[nid][t]})
    Xs, ys, ss = [], [], []
    for s in sessions:
        for cond, lab in (("clear", 0), ("weapon", 1)):
            blocks = {}                                    # channel -> field array (m,*)
            for (nid, tag), ci in chan.items():
                hit = [d for d in nodes[nid][tag]
                       if f"_metal_{s}_{cond}_link{tag}" in os.path.basename(d)]
                if hit:
                    blocks[ci] = getattr(load_dataset(hit[0]), field).astype(np.float32)
            if len(blocks) != len(links):                  # need all 12 channels present
                print(f"   [skip] {s}/{cond}: only {len(blocks)}/{len(links)} links")
                continue
            m = min(b.shape[0] for b in blocks.values())
            stack = np.stack([blocks[ci][:m] for ci in range(len(links))], axis=1)  # (m,12,*)
            Xs.append(stack)
            ys.append(np.full(m, lab, dtype=np.int64))
            ss.append(np.array([s] * m))
    X = np.concatenate(Xs).astype(np.float32)
    return X, np.concatenate(ys), np.concatenate(ss)


def exp_combined(nodes, links, root, save_to):
    print("\n=== C. COMBINED 12-link multi-channel CNN (session-LOGO) ===")
    X, y, sess = build_aligned(nodes, links, "X_image")
    subj = np.array(["ilker4node"] * len(y))
    k = X.shape[2]
    print(f"   tensor X={X.shape} (n, channels=12, K={k}, window={X.shape[3]})  "
          f"class_counts={dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)]))}")
    s = _session_logo(X, y, sess, subj, lambda: WeaponHead(_cfg("cnn", k)))
    print(_fmt("12-link CNN", len(y), s))
    # persist a full-data fit for the record
    head = WeaponHead(_cfg("cnn", k)); head.feature_mode = "cnn"; head.fit(X, y)
    head.save(os.path.join(save_to, "model.joblib"))
    print(f"   saved combined CNN -> {save_to}/model.joblib")
    return s


# ---- D. weighted fusion: every link votes, weighted by reliability -> one 0..1 score -------------
def _reliability(X, y, sess):
    """LinkVoter weight = max(session-LOGO acc - 0.5, 0)*2 (same formula run_weapon uses). A link that
    can't beat chance on the TRAINING sessions gets ~0 vote; computed only on train folds (no leak)."""
    s = _session_logo(X, y, sess, np.array(["_"] * len(y)),
                      lambda: WeaponHead(_cfg("variance", 12)))
    return max((s["accuracy"] if s else 0.5) - 0.5, 0.0) * 2.0


def _fused_metrics(tru, fus):
    pred = (fus >= 0.5).astype(int)
    acc = float((pred == tru).mean()); maj = float(max(tru.mean(), 1 - tru.mean()))
    tpr = float(pred[tru == 1].mean()) if (tru == 1).any() else 0.0
    fp = float(pred[tru == 0].mean()) if (tru == 0).any() else 0.0
    return acc, maj, tpr, fp


def _best_threshold(scores, y):
    """Threshold (>=) maximizing accuracy on the GIVEN scores. Leak-free ONLY when fed TRAIN data —
    the legit version of Gemini's idea (its code searched the TEST scores, which leaks)."""
    best_t, best_a = 0.5, -1.0
    for t in np.unique(scores):
        a = float(((scores >= t).astype(int) == y).mean())
        if a > best_a:
            best_a, best_t = a, float(t)
    return best_t


def exp_weighted_fusion(nodes, links):
    """FLAT per-link fusion (leak-free, session-LOGO): each of the 12 links is an INDEPENDENT voter —
    no node grouping. Each link head emits p(weapon)∈[0,1]; fuse as a reliability-weighted average
    (link -> link-weight -> result) into ONE 0..1 score per window. Weights come from each link's OWN
    reliability on the TRAIN sessions only; the fused score is judged on the held-out session. This is
    exactly what run_weapon serves (LinkVoter keyed per link)."""
    print("\n=== D. FLAT PER-LINK FUSION — 12 independent link voters -> one 0..1 score (session-LOGO) ===")
    IC, y, sess = build_aligned(nodes, links, "X_intercarrier")  # (n,12,27)
    names = [f"n{nid}/{tag}" for nid, tag in links]              # link id = (tx_tag -> rx_node)
    folds = sorted(set(sess.tolist()))
    L = len(links)
    tru = []; fus = []
    wsum = np.zeros(L)
    aucs = []; acc05 = []; acctr = []; thrs = []          # Gemini-hypothesis test (leak-free)
    last = None
    for st in folds:
        tr, te = sess != st, sess == st
        W = np.zeros(L)                                  # per-LINK weights (independent)
        Ptr = np.zeros((int(tr.sum()), L))               # per-LINK train probs (for the train threshold)
        Pte = np.zeros((int(te.sum()), L))               # per-LINK test probs
        for i in range(L):
            head = WeaponHead(_cfg("variance", 12)).fit(IC[tr, i, :], y[tr])
            W[i] = _reliability(IC[tr, i, :], y[tr], sess[tr])
            Ptr[:, i] = head.predict_proba(IC[tr, i, :])[:, 1]
            Pte[:, i] = head.predict_proba(IC[te, i, :])[:, 1]
        wsum += W
        ws = W.sum() if W.sum() > 0 else 1.0             # all-junk fold -> plain mean fallback
        ftr = Ptr @ W / ws if W.sum() > 0 else Ptr.mean(axis=1)
        fte = Pte @ W / ws if W.sum() > 0 else Pte.mean(axis=1)
        thr = _best_threshold(ftr, y[tr])                # threshold from TRAIN only (no leak)
        thrs.append(thr)
        acc05.append(float(((fte >= 0.5).astype(int) == y[te]).mean()))
        acctr.append(float(((fte >= thr).astype(int) == y[te]).mean()))
        if np.unique(y[te]).size == 2:
            aucs.append(float(roc_auc_score(y[te], fte)))  # threshold-free: is there ANY ranking signal?
        tru.append(y[te]); fus.append(fte)
        last = (st, Pte, W, fte, y[te])

    tru = np.concatenate(tru); fus = np.concatenate(fus)
    avgW = wsum / len(folds)

    print("   per-LINK vote WEIGHT (avg over folds, reliability = max(LOGO-0.5,0)*2; links independent):")
    for i in np.argsort(-avgW):
        print(f"      {names[i]:<10} w={avgW[i]:.3f}  {'#' * int(round(avgW[i] * 40))}")

    a, m, t, fp = _fused_metrics(tru, fus)
    print(f"\n   FUSED SYSTEM score (0..1) -> threshold 0.5  (link->link-weight->result):")
    print(f"      clear={fus[tru == 0].mean():.3f} weapon={fus[tru == 1].mean():.3f}"
          f" (sep {fus[tru == 1].mean() - fus[tru == 0].mean():+.3f})  "
          f"LOGO={a:.3f} maj={m:.3f} TPR={t:.3f} FP={fp:.3f}  {'BEATS' if a > m + 1e-9 else 'below'} majority")

    # ---- Gemini hypothesis: "hidden signal squashed by a misplaced 0.5 threshold"? Settle it honestly.
    auc = float(np.mean(aucs)) if aucs else float("nan")
    print(f"\n   GEMINI TEST — is signal hidden behind a bad threshold, or just absent?")
    print(f"      ROC-AUC (threshold-FREE, mean/fold) = {auc:.3f}   (0.50 = no ranking signal at all)")
    print(f"      acc @ fixed 0.5         = {np.mean(acc05):.3f}")
    print(f"      acc @ TRAIN-picked thr  = {np.mean(acctr):.3f}   (per-fold thr {[round(x, 2) for x in thrs]})")
    if auc <= 0.55:
        verdict = "NO ranking signal (AUC≈0.5) — data ceiling, not a threshold problem"
    elif np.mean(acctr) > np.mean(acc05) + 0.02:
        verdict = f"threshold WAS misplaced — AUC {auc:.2f}>0.5 and a TRAIN-derived threshold recovers it"
    else:
        verdict = (f"weak ranking signal (AUC {auc:.2f}) EXISTS but NO fixed threshold (0.5 or "
                   "train-derived) transfers — the fused-score DISTRIBUTION drifts across sessions "
                   "(domain shift). Re-thresholding does NOT rescue it; need per-session score "
                   "normalization/calibration or more data. (Gemini's mechanism right, fix wrong.)")
    print(f"      -> {verdict}")

    # ---- see all: per-sample breakdown for a few held-out windows (last fold) ----
    st, P, W, f_link, yte = last
    ws = W.sum() if W.sum() > 0 else 1.0
    cols = np.argsort(-W)[:6]
    print(f"\n   SEE-ALL (LINK path) — held-out session {st}, top-{len(cols)} voters (cell = p*weight):")
    hdr = "      " + "true  fused | " + " ".join(f"{names[i]:>10}" for i in cols)
    print(hdr); print("      " + "-" * (len(hdr) - 6))
    for r in list(np.where(yte == 1)[0][:4]) + list(np.where(yte == 0)[0][:4]):
        cells = " ".join(f"{P[r, i]:.2f}*{W[i]:.2f}" for i in cols)
        print(f"      {'WPN' if yte[r] == 1 else 'clr':>4}  {f_link[r]:.3f} | {cells}")
    print(f"      (fused = Σ p·w / Σw, Σw={ws:.2f}; a voter with w≈0 contributes nothing)")
    return a


# ---- E. drift-fix ablation: do the literature-backed preprocessing fixes help threshold transfer? ---
def _mad_clip(IC, sess, k=3.0):
    """Hampel-style robust clip per (link,feature) WITHIN each session: clamp to median ± k·1.4826·MAD.
    Label-free (uses only the feature distribution → leak-safe on the test session). Kills CSI spikes
    that inflate the variance feature and worsen cross-session drift (Non-Obtrusive/espectre)."""
    out = IC.copy()
    for s in set(sess.tolist()):
        m = sess == s
        sub = out[m]                                   # (ns, L, 27)
        med = np.median(sub, axis=0, keepdims=True)
        mad = np.median(np.abs(sub - med), axis=0, keepdims=True) * 1.4826 + 1e-9
        out[m] = np.clip(sub, med - k * mad, med + k * mad)
    return out


def _zscore_sess(IC, sess):
    """Per-session z-norm per (link,feature): subtract the session mean, divide by session std. Label-
    FREE — the test session is normalized by its OWN unlabeled stats (transductive, no label leak). This
    removes the absolute σ² baseline drift that breaks a fixed threshold across sessions (the diagnosed
    failure; cross-domain survey: subtract running baseline)."""
    out = IC.copy()
    for s in set(sess.tolist()):
        m = sess == s
        sub = out[m]
        out[m] = (sub - sub.mean(axis=0, keepdims=True)) / (sub.std(axis=0, keepdims=True) + 1e-9)
    return out


def _fusion_eval(IC, y, sess, links, *, augment=0, seed=0):
    """Flat per-link fusion under session-LOGO; returns pooled AUC + acc@0.5 + acc@train-threshold + sep.
    Heads fit on (optionally) augmented TRAIN; weights come from the ORIGINAL train (no aug leak); the
    augmented copies are train-only (never test). augment = #jittered copies appended (factor augment+1;
    jitter + magnitude warp, sensors-25-03955)."""
    rng = np.random.default_rng(seed)
    folds = sorted(set(sess.tolist())); L = len(links)
    tru = []; fus = []; aucs = []; acc05 = []; acctr = []
    for st in folds:
        tr, te = sess != st, sess == st
        Xtr0, ytr0, str0 = IC[tr], y[tr], sess[tr]
        Xte, yte = IC[te], y[te]
        if augment > 0:
            sd = np.std(Xtr0, axis=0, keepdims=True) + 1e-6
            blocks = [Xtr0]; ys = [ytr0]
            for _ in range(augment):
                jit = Xtr0 + rng.normal(0, 0.05, Xtr0.shape) * sd        # jitter ~5% robust std
                jit = jit * (1 + rng.normal(0, 0.03, (Xtr0.shape[0], 1, 1)))  # magnitude warp
                blocks.append(jit); ys.append(ytr0)
            Xtr = np.concatenate(blocks); ytr = np.concatenate(ys)
        else:
            Xtr, ytr = Xtr0, ytr0
        W = np.zeros(L); Ptr = np.zeros((len(ytr), L)); Pte = np.zeros((int(te.sum()), L))
        for i in range(L):
            head = WeaponHead(_cfg("variance", 12)).fit(Xtr[:, i, :], ytr)
            W[i] = _reliability(Xtr0[:, i, :], ytr0, str0)   # reliability on ORIGINAL train (no aug leak)
            Ptr[:, i] = head.predict_proba(Xtr[:, i, :])[:, 1]
            Pte[:, i] = head.predict_proba(Xte[:, i, :])[:, 1]
        ws = W.sum() if W.sum() > 0 else 1.0
        ftr = Ptr @ W / ws if W.sum() > 0 else Ptr.mean(axis=1)
        fte = Pte @ W / ws if W.sum() > 0 else Pte.mean(axis=1)
        thr = _best_threshold(ftr, ytr)
        acc05.append(float(((fte >= 0.5).astype(int) == yte).mean()))
        acctr.append(float(((fte >= thr).astype(int) == yte).mean()))
        if np.unique(yte).size == 2:
            aucs.append(float(roc_auc_score(yte, fte)))
        tru.append(yte); fus.append(fte)
    tru = np.concatenate(tru); fus = np.concatenate(fus)
    return dict(auc=float(np.mean(aucs)) if aucs else float("nan"),
                acc05=float(np.mean(acc05)), acctr=float(np.mean(acctr)),
                sep=float(fus[tru == 1].mean() - fus[tru == 0].mean()))


def exp_drift_ablation(nodes, links):
    """Test the 3 literature-backed fixes (Hampel/MAD denoise, per-session baseline z-norm, train-only
    augmentation) against the drift problem, leak-free, on the flat per-link fusion. AUC = ranking
    signal; acc@trThr rising toward/above acc@0.5 with the gap closed = a threshold finally TRANSFERS."""
    print("\n=== E. DRIFT-FIX ABLATION — do denoise / per-session norm / augment help? (session-LOGO) ===")
    IC0, y, sess = build_aligned(nodes, links, "X_intercarrier")
    den = _mad_clip(IC0, sess)
    configs = [
        ("baseline (current)", IC0, 0),
        ("+Hampel/MAD denoise", den, 0),
        ("+denoise +per-session z-norm", _zscore_sess(den, sess), 0),
        ("+denoise +z-norm +augment x5", _zscore_sess(den, sess), 4),
    ]
    print(f"   {'config':<32}  AUC    acc@0.5  acc@trThr   sep")
    print("   " + "-" * 64)
    for name, IC, aug in configs:
        r = _fusion_eval(IC, y, sess, links, augment=aug)
        print(f"   {name:<32} {r['auc']:.3f}   {r['acc05']:.3f}    {r['acctr']:.3f}    {r['sep']:+.3f}")
    print("   (AUC↑ = more ranking signal; acc@trThr↑ with small gap to acc@0.5 = threshold TRANSFERS)")


def main():
    ap = argparse.ArgumentParser(description="Weapon model bake-off (per-node / per-link / combined CNN).")
    ap.add_argument("--root", default="data/2g4_ht40/ui")
    ap.add_argument("--skip-cnn", action="store_true", help="Skip exp_combined CNN (slow on CPU)")
    args = ap.parse_args()

    nodes, links = discover(args.root)
    if not nodes:
        print(f"[ERROR] no weapon datasets under {args.root}/weapon_ds/node*"); return
    print(f"Pool: nodes {sorted(nodes)}; {len(links)} links {['n%d/%s' % (n, t) for n, t in links]}")

    deploy = exp_per_node(nodes, args.root)
    exp_per_link(nodes)
    if not args.skip_cnn:
        exp_combined(nodes, links, args.root, os.path.join(args.root, "model_weapon_combined"))
    else:
        print("\n=== C. COMBINED CNN — skipped (--skip-cnn) ===")
    exp_weighted_fusion(nodes, links)
    exp_drift_ablation(nodes, links)

    # canonical deployable: retrain the per-node ic27/variance heads run_weapon serves
    print("\n=== Retraining canonical per-node ic27 heads -> model_weapon/node* (deployable) ===")
    for nid in sorted(deploy):
        _, m = train_weapon(deploy[nid], out_dir=f"{args.root}/model_weapon/node{nid}",
                            feature_mode="ic27")
        lg = (m.get("logo") or {}).get("session", {})
        print(f"   node{nid}: LOGO={lg.get('accuracy', float('nan')):.3f} "
              f"(maj {lg.get('majority_accuracy', float('nan')):.3f})")


if __name__ == "__main__":
    main()
