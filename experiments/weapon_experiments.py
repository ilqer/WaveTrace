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

from wavetrace.adapters.recognition.heads import build_weapon_head
from wavetrace.adapters.store.dataset import load_dataset
from wavetrace.domain.model_config import ModelConfig
from wavetrace.recognition import parse_link_tag
from wavetrace.recognition.train import _logo_metrics, train_weapon

WINDOW, HOP = 128, 32


def _weapon_config(backend, subcarrier_count, **options):
    return ModelConfig(stage="weapon", k=subcarrier_count, backend=backend, window=WINDOW, hop=HOP, **options)


def _session_logo(X, y, session_ids, subject_ids, make_head):
    """Run _logo_metrics and return only the session fold (the valid one) or None."""
    metrics = _logo_metrics(np.asarray(X), np.asarray(y), np.asarray(session_ids), np.asarray(subject_ids),
                            make_head)
    return metrics.get("session")


def _format_logo_row(tag, sample_count, fold):
    if fold is None:
        return f"   {tag:<26} n={sample_count:<4} (LOGO needs >=2 sessions)"
    return (f"   {tag:<26} n={sample_count:<4} LOGO={fold['accuracy']:.3f}  "
            f"maj={fold['majority_accuracy']:.3f}  "
            f"TPR={fold.get('tpr', 0):.3f}  FP={fold.get('fp_rate', 0):.3f}  "
            f"{'BEATS' if fold['accuracy'] > fold['majority_accuracy'] + 1e-9 else 'below'} majority")


def discover(root):
    """{node -> {tag -> [dataset dirs]}} for the latest ilker4node_metal pool, plus the 12-link order."""
    nodes = {}
    for node_dir in sorted(glob.glob(f"{root}/weapon_ds/node*")):
        node_id = int(os.path.basename(node_dir)[4:])
        dirs_by_tag = collections.defaultdict(list)
        for dataset_dir in sorted(glob.glob(f"{node_dir}/*")):
            tag = parse_link_tag(dataset_dir)
            if tag:
                dirs_by_tag[tag].append(dataset_dir)
        if dirs_by_tag:
            nodes[node_id] = dict(dirs_by_tag)
    links = sorted((node_id, tag) for node_id in nodes for tag in nodes[node_id])
    return nodes, links


def exp_per_node(nodes):
    print("\n=== A. PER-NODE pooled heads (session-LOGO) ===")
    deploy = {}
    for node_id in sorted(nodes):
        dataset_dirs = [dataset_dir for tag in nodes[node_id] for dataset_dir in nodes[node_id][tag]]
        datasets = [load_dataset(dataset_dir) for dataset_dir in dataset_dirs]
        X_intercarrier = np.concatenate([dataset.X_intercarrier for dataset in datasets]).astype(np.float32)
        X_image = np.concatenate([dataset.X_image for dataset in datasets]).astype(np.float32)
        y = np.concatenate([dataset.y for dataset in datasets])
        session_ids = np.concatenate([dataset.session_ids for dataset in datasets])
        subject_ids = np.concatenate([dataset.subject_ids for dataset in datasets])
        subcarrier_count = int(datasets[0].meta["K"])
        print(f" node{node_id}:")
        for name, backend, X in (("ic27/variance", "variance", X_intercarrier),
                                 ("ic27/mlp", "mlp", X_intercarrier),
                                 ("ic27/svm", "svm", X_intercarrier),
                                 ("cnn/image", "cnn", X_image)):
            fold = _session_logo(X, y, session_ids, subject_ids,
                                 lambda head_backend=backend, head_k=subcarrier_count:
                                 build_weapon_head(_weapon_config(head_backend, head_k)))
            print(_format_logo_row(name, len(y), fold))
        deploy[node_id] = dataset_dirs  # feeds the canonical ic27 retrain
    return deploy


def exp_per_link(nodes):
    print("\n=== B. PER-LINK heads — each tx->rx direction (session-LOGO) ===")
    rows = []
    for node_id in sorted(nodes):
        for tag in sorted(nodes[node_id]):
            datasets = [load_dataset(dataset_dir) for dataset_dir in nodes[node_id][tag]]
            X_intercarrier = np.concatenate(
                [dataset.X_intercarrier for dataset in datasets]).astype(np.float32)
            y = np.concatenate([dataset.y for dataset in datasets])
            session_ids = np.concatenate([dataset.session_ids for dataset in datasets])
            subject_ids = np.concatenate([dataset.subject_ids for dataset in datasets])
            subcarrier_count = int(datasets[0].meta["K"])
            variance_fold = _session_logo(X_intercarrier, y, session_ids, subject_ids,
                                          lambda head_k=subcarrier_count:
                                          build_weapon_head(_weapon_config("variance", head_k)))
            mlp_fold = _session_logo(X_intercarrier, y, session_ids, subject_ids,
                                     lambda head_k=subcarrier_count:
                                     build_weapon_head(_weapon_config("mlp", head_k)))
            print(_format_logo_row(f"node{node_id} link{tag} var", len(y), variance_fold))
            print(_format_logo_row(f"node{node_id} link{tag} mlp", len(y), mlp_fold))
            if variance_fold:
                rows.append((node_id, tag, variance_fold["accuracy"], variance_fold["majority_accuracy"]))
    beats = [row for row in rows if row[2] > row[3] + 1e-9]
    print(f"\n   per-link (variance) links beating majority: {len(beats)}/{len(rows)}")
    for node_id, tag, accuracy, majority_accuracy in sorted(beats, key=lambda row: -row[2]):
        print(f"      node{node_id} link{tag}: {accuracy:.3f} (maj {majority_accuracy:.3f})")


def build_aligned(nodes, links, field):
    """(n,12,*) X, y, session — each sample's 12 link-channels are aligned at the same window.
    `field` = "X_image" -> (n,12,K,window) for the combined CNN ; "X_intercarrier" -> (n,12,27) for
    the weighted fusion. Index-wise stack per (session,condition) trimmed to the min window count
    (static-subject, shared 100 Hz grid -> window k of every link is the same instant; any slip is
    immaterial for a stationary signal)."""
    channel_of_link = {link: i for i, link in enumerate(links)}
    sessions = sorted({os.path.basename(dataset_dir).split("_metal_")[1].split("_")[0]
                       for node_id in nodes for link_tag in nodes[node_id]
                       for dataset_dir in nodes[node_id][link_tag]})
    X_blocks, label_blocks, session_blocks = [], [], []
    for session in sessions:
        for condition, label in (("clear", 0), ("weapon", 1)):
            blocks = {}                                    # channel -> field array (window_count,*)
            for (node_id, tag), channel in channel_of_link.items():
                matches = [dataset_dir for dataset_dir in nodes[node_id][tag]
                           if f"_metal_{session}_{condition}_link{tag}" in os.path.basename(dataset_dir)]
                if matches:
                    blocks[channel] = getattr(load_dataset(matches[0]), field).astype(np.float32)
            if len(blocks) != len(links):
                print(f"   [skip] {session}/{condition}: only {len(blocks)}/{len(links)} links")
                continue
            window_count = min(block.shape[0] for block in blocks.values())
            stack = np.stack([blocks[channel][:window_count] for channel in range(len(links))],
                             axis=1)  # (window_count,12,*)
            X_blocks.append(stack)
            label_blocks.append(np.full(window_count, label, dtype=np.int64))
            session_blocks.append(np.array([session] * window_count))
    X = np.concatenate(X_blocks).astype(np.float32)
    return X, np.concatenate(label_blocks), np.concatenate(session_blocks)


def exp_combined(nodes, links, save_to):
    print("\n=== C. COMBINED 12-link multi-channel CNN (session-LOGO) ===")
    X, y, session_ids = build_aligned(nodes, links, "X_image")
    subject_ids = np.array(["ilker4node"] * len(y))
    subcarrier_count = X.shape[2]
    print(f"   tensor X={X.shape} (n, channels=12, K={subcarrier_count}, window={X.shape[3]})  "
          f"class_counts={dict(zip(*[array.tolist() for array in np.unique(y, return_counts=True)]))}")
    fold = _session_logo(X, y, session_ids, subject_ids,
                         lambda: build_weapon_head(_weapon_config("cnn", subcarrier_count)))
    print(_format_logo_row("12-link CNN", len(y), fold))
    head = build_weapon_head(_weapon_config("cnn", subcarrier_count))
    head.feature_mode = "cnn"
    head.fit(X, y)
    head.save(os.path.join(save_to, "model.joblib"))
    print(f"   saved combined CNN -> {save_to}/model.joblib")
    return fold


def _reliability(X, y, session_ids):
    """LinkVoter weight = max(session-LOGO acc - 0.5, 0)*2 (same formula run_weapon uses). A link that
    can't beat chance on the TRAINING sessions gets ~0 vote; computed only on train folds (no leak)."""
    fold = _session_logo(X, y, session_ids, np.array(["_"] * len(y)),
                         lambda: build_weapon_head(_weapon_config("variance", 12)))
    return max((fold["accuracy"] if fold else 0.5) - 0.5, 0.0) * 2.0


def _fused_metrics(true_labels, fused_scores):
    predictions = (fused_scores >= 0.5).astype(int)
    accuracy = float((predictions == true_labels).mean())
    majority_accuracy = float(max(true_labels.mean(), 1 - true_labels.mean()))
    tpr = float(predictions[true_labels == 1].mean()) if (true_labels == 1).any() else 0.0
    false_positive_rate = float(predictions[true_labels == 0].mean()) if (true_labels == 0).any() else 0.0
    return accuracy, majority_accuracy, tpr, false_positive_rate


def _best_threshold(scores, y):
    """Threshold (>=) maximizing accuracy on the GIVEN scores. Leak-free ONLY when fed TRAIN data."""
    best_threshold, best_accuracy = 0.5, -1.0
    for threshold in np.unique(scores):
        accuracy = float(((scores >= threshold).astype(int) == y).mean())
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, float(threshold)
    return best_threshold


def exp_weighted_fusion(nodes, links):
    """FLAT per-link fusion (leak-free, session-LOGO): each of the 12 links is an INDEPENDENT voter —
    no node grouping. Each link head emits p(weapon)∈[0,1]; fuse as a reliability-weighted average
    (link -> link-weight -> result) into ONE 0..1 score per window. Weights come from each link's OWN
    reliability on the TRAIN sessions only; the fused score is judged on the held-out session. This is
    exactly what run_weapon serves (LinkVoter keyed per link)."""
    print("\n=== D. FLAT PER-LINK FUSION — 12 independent link voters -> one 0..1 score (session-LOGO) ===")
    X_intercarrier, y, session_ids = build_aligned(nodes, links, "X_intercarrier")  # (n,12,27)
    names = [f"n{node_id}/{tag}" for node_id, tag in links]              # link id = (tx_tag -> rx_node)
    folds = sorted(set(session_ids.tolist()))
    link_count = len(links)
    true_labels = []; fused_scores = []
    weight_sums = np.zeros(link_count)
    auc_per_fold = []; accuracy_at_half = []; accuracy_at_train_threshold = []; thresholds = []
    last_fold = None
    for held_out_session in folds:
        is_train, is_test = session_ids != held_out_session, session_ids == held_out_session
        weights = np.zeros(link_count)
        train_probas = np.zeros((int(is_train.sum()), link_count))
        test_probas = np.zeros((int(is_test.sum()), link_count))
        for i in range(link_count):
            head = build_weapon_head(_weapon_config("variance", 12)).fit(X_intercarrier[is_train, i, :],
                                                                          y[is_train])
            weights[i] = _reliability(X_intercarrier[is_train, i, :], y[is_train], session_ids[is_train])
            train_probas[:, i] = head.predict_proba(X_intercarrier[is_train, i, :])[:, 1]
            test_probas[:, i] = head.predict_proba(X_intercarrier[is_test, i, :])[:, 1]
        weight_sums += weights
        # all-junk fold -> plain mean fallback
        weight_total = weights.sum() if weights.sum() > 0 else 1.0
        fused_train = (train_probas @ weights / weight_total if weights.sum() > 0
                       else train_probas.mean(axis=1))
        fused_test = test_probas @ weights / weight_total if weights.sum() > 0 else test_probas.mean(axis=1)
        threshold = _best_threshold(fused_train, y[is_train])  # from TRAIN only (no leak)
        thresholds.append(threshold)
        accuracy_at_half.append(float(((fused_test >= 0.5).astype(int) == y[is_test]).mean()))
        accuracy_at_train_threshold.append(
            float(((fused_test >= threshold).astype(int) == y[is_test]).mean()))
        if np.unique(y[is_test]).size == 2:
            # threshold-free: is there ANY ranking signal?
            auc_per_fold.append(float(roc_auc_score(y[is_test], fused_test)))
        true_labels.append(y[is_test]); fused_scores.append(fused_test)
        last_fold = (held_out_session, test_probas, weights, fused_test, y[is_test])

    true_labels = np.concatenate(true_labels); fused_scores = np.concatenate(fused_scores)
    mean_link_weights = weight_sums / len(folds)

    print("   per-LINK vote WEIGHT (avg over folds, reliability = max(LOGO-0.5,0)*2; links independent):")
    for i in np.argsort(-mean_link_weights):
        print(f"      {names[i]:<10} w={mean_link_weights[i]:.3f}  "
              f"{'#' * int(round(mean_link_weights[i] * 40))}")

    accuracy, majority_accuracy, tpr, false_positive_rate = _fused_metrics(true_labels, fused_scores)
    print("\n   FUSED SYSTEM score (0..1) -> threshold 0.5  (link->link-weight->result):")
    print(f"      clear={fused_scores[true_labels == 0].mean():.3f} "
          f"weapon={fused_scores[true_labels == 1].mean():.3f}"
          f" (sep {fused_scores[true_labels == 1].mean() - fused_scores[true_labels == 0].mean():+.3f})  "
          f"LOGO={accuracy:.3f} maj={majority_accuracy:.3f} TPR={tpr:.3f} FP={false_positive_rate:.3f}  "
          f"{'BEATS' if accuracy > majority_accuracy + 1e-9 else 'below'} majority")

    auc = float(np.mean(auc_per_fold)) if auc_per_fold else float("nan")
    print("\n   GEMINI TEST — is signal hidden behind a bad threshold, or just absent?")
    print(f"      ROC-AUC (threshold-FREE, mean/fold) = {auc:.3f}   (0.50 = no ranking signal at all)")
    print(f"      acc @ fixed 0.5         = {np.mean(accuracy_at_half):.3f}")
    print(f"      acc @ TRAIN-picked thr  = {np.mean(accuracy_at_train_threshold):.3f}   "
          f"(per-fold thr {[round(value, 2) for value in thresholds]})")
    if auc <= 0.55:
        verdict = "NO ranking signal (AUC≈0.5) — data ceiling, not a threshold problem"
    elif np.mean(accuracy_at_train_threshold) > np.mean(accuracy_at_half) + 0.02:
        verdict = f"threshold WAS misplaced — AUC {auc:.2f}>0.5 and a TRAIN-derived threshold recovers it"
    else:
        verdict = (f"weak ranking signal (AUC {auc:.2f}) EXISTS but NO fixed threshold (0.5 or "
                   "train-derived) transfers — the fused-score DISTRIBUTION drifts across sessions "
                   "(domain shift). Re-thresholding does NOT rescue it; need per-session score "
                   "normalization/calibration or more data. (Gemini's mechanism right, fix wrong.)")
    print(f"      -> {verdict}")

    held_out_session, held_out_probas, weights, held_out_fused, held_out_y = last_fold
    weight_total = weights.sum() if weights.sum() > 0 else 1.0
    top_links = np.argsort(-weights)[:6]
    print(f"\n   SEE-ALL (LINK path) — held-out session {held_out_session}, "
          f"top-{len(top_links)} voters (cell = p*weight):")
    header = "      " + "true  fused | " + " ".join(f"{names[i]:>10}" for i in top_links)
    print(header); print("      " + "-" * (len(header) - 6))
    for row in list(np.where(held_out_y == 1)[0][:4]) + list(np.where(held_out_y == 0)[0][:4]):
        cells = " ".join(f"{held_out_probas[row, i]:.2f}*{weights[i]:.2f}" for i in top_links)
        print(f"      {'WPN' if held_out_y[row] == 1 else 'clr':>4}  {held_out_fused[row]:.3f} | {cells}")
    print(f"      (fused = Σ p·w / Σw, Σw={weight_total:.2f}; a voter with w≈0 contributes nothing)")
    return accuracy


def _mad_clip(X_intercarrier, session_ids, clip_multiple=3.0):
    """Hampel-style robust clip per (link,feature) WITHIN each session: clamp to median ± k·1.4826·MAD
    with k = clip_multiple. Label-free (uses only the feature distribution → leak-safe on the test
    session). Kills CSI spikes that inflate the variance feature and worsen cross-session drift
    (Non-Obtrusive/espectre)."""
    clipped = X_intercarrier.copy()
    for session in set(session_ids.tolist()):
        is_session = session_ids == session
        session_rows = clipped[is_session]                                  # (ns, L, 27)
        median = np.median(session_rows, axis=0, keepdims=True)
        mad = np.median(np.abs(session_rows - median), axis=0, keepdims=True) * 1.4826 + 1e-9
        clipped[is_session] = np.clip(session_rows, median - clip_multiple * mad,
                                      median + clip_multiple * mad)
    return clipped


def _zscore_per_session(X_intercarrier, session_ids):
    """Per-session z-norm per (link,feature): subtract the session mean, divide by session std. Label-
    FREE — the test session is normalized by its OWN unlabeled stats (transductive, no label leak). This
    removes the absolute σ² baseline drift that breaks a fixed threshold across sessions (the diagnosed
    failure; cross-domain survey: subtract running baseline)."""
    normalized = X_intercarrier.copy()
    for session in set(session_ids.tolist()):
        is_session = session_ids == session
        session_rows = normalized[is_session]
        normalized[is_session] = ((session_rows - session_rows.mean(axis=0, keepdims=True))
                                  / (session_rows.std(axis=0, keepdims=True) + 1e-9))
    return normalized


def _evaluate_fusion(X_intercarrier, y, session_ids, links, *, augment=0, seed=0):
    """Flat per-link fusion under session-LOGO; returns pooled AUC + acc@0.5 + acc@train-threshold + sep.
    Heads fit on (optionally) augmented TRAIN; weights come from the ORIGINAL train (no aug leak); the
    augmented copies are train-only (never test). augment = #jittered copies appended (factor augment+1;
    jitter + magnitude warp, sensors-25-03955)."""
    rng = np.random.default_rng(seed)
    folds = sorted(set(session_ids.tolist())); link_count = len(links)
    true_labels = []; fused_scores = []; auc_per_fold = []; accuracy_at_half = []
    accuracy_at_train_threshold = []
    for held_out_session in folds:
        is_train, is_test = session_ids != held_out_session, session_ids == held_out_session
        X_train_raw, y_train_raw, session_ids_train = (X_intercarrier[is_train], y[is_train],
                                                       session_ids[is_train])
        X_test, y_test = X_intercarrier[is_test], y[is_test]
        if augment > 0:
            feature_standard_deviation = np.std(X_train_raw, axis=0, keepdims=True) + 1e-6
            blocks = [X_train_raw]; label_blocks = [y_train_raw]
            for _ in range(augment):
                # jitter ~5% robust std
                jittered = X_train_raw + rng.normal(0, 0.05, X_train_raw.shape) * feature_standard_deviation
                # magnitude warp
                jittered = jittered * (1 + rng.normal(0, 0.03, (X_train_raw.shape[0], 1, 1)))
                blocks.append(jittered); label_blocks.append(y_train_raw)
            X_train = np.concatenate(blocks); y_train = np.concatenate(label_blocks)
        else:
            X_train, y_train = X_train_raw, y_train_raw
        weights = np.zeros(link_count)
        train_probas = np.zeros((len(y_train), link_count))
        test_probas = np.zeros((int(is_test.sum()), link_count))
        for i in range(link_count):
            head = build_weapon_head(_weapon_config("variance", 12)).fit(X_train[:, i, :], y_train)
            # reliability on ORIGINAL train (no aug leak)
            weights[i] = _reliability(X_train_raw[:, i, :], y_train_raw, session_ids_train)
            train_probas[:, i] = head.predict_proba(X_train[:, i, :])[:, 1]
            test_probas[:, i] = head.predict_proba(X_test[:, i, :])[:, 1]
        weight_total = weights.sum() if weights.sum() > 0 else 1.0
        fused_train = (train_probas @ weights / weight_total if weights.sum() > 0
                       else train_probas.mean(axis=1))
        fused_test = test_probas @ weights / weight_total if weights.sum() > 0 else test_probas.mean(axis=1)
        threshold = _best_threshold(fused_train, y_train)
        accuracy_at_half.append(float(((fused_test >= 0.5).astype(int) == y_test).mean()))
        accuracy_at_train_threshold.append(float(((fused_test >= threshold).astype(int) == y_test).mean()))
        if np.unique(y_test).size == 2:
            auc_per_fold.append(float(roc_auc_score(y_test, fused_test)))
        true_labels.append(y_test); fused_scores.append(fused_test)
    true_labels = np.concatenate(true_labels); fused_scores = np.concatenate(fused_scores)
    return dict(auc=float(np.mean(auc_per_fold)) if auc_per_fold else float("nan"),
                acc05=float(np.mean(accuracy_at_half)), acctr=float(np.mean(accuracy_at_train_threshold)),
                sep=float(fused_scores[true_labels == 1].mean() - fused_scores[true_labels == 0].mean()))


def exp_drift_ablation(nodes, links):
    """Test the 3 literature-backed fixes (Hampel/MAD denoise, per-session baseline z-norm, train-only
    augmentation) against the drift problem, leak-free, on the flat per-link fusion. AUC = ranking
    signal; acc@trThr rising toward/above acc@0.5 with the gap closed = a threshold finally TRANSFERS."""
    print("\n=== E. DRIFT-FIX ABLATION — do denoise / per-session norm / augment help? (session-LOGO) ===")
    X_intercarrier_raw, y, session_ids = build_aligned(nodes, links, "X_intercarrier")
    denoised = _mad_clip(X_intercarrier_raw, session_ids)
    configs = [
        ("baseline (current)", X_intercarrier_raw, 0),
        ("+Hampel/MAD denoise", denoised, 0),
        ("+denoise +per-session z-norm", _zscore_per_session(denoised, session_ids), 0),
        ("+denoise +z-norm +augment x5", _zscore_per_session(denoised, session_ids), 4),
    ]
    print(f"   {'config':<32}  AUC    acc@0.5  acc@trThr   sep")
    print("   " + "-" * 64)
    for name, X_intercarrier, augment_copies in configs:
        scores = _evaluate_fusion(X_intercarrier, y, session_ids, links, augment=augment_copies)
        print(f"   {name:<32} {scores['auc']:.3f}   {scores['acc05']:.3f}    {scores['acctr']:.3f}    "
              f"{scores['sep']:+.3f}")
    print("   (AUC↑ = more ranking signal; acc@trThr↑ with small gap to acc@0.5 = threshold TRANSFERS)")


def main():
    parser = argparse.ArgumentParser(
        description="Weapon model bake-off (per-node / per-link / combined CNN).")
    parser.add_argument("--root", default="data/2g4_ht40/ui")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip exp_combined CNN (slow on CPU)")
    args = parser.parse_args()

    nodes, links = discover(args.root)
    if not nodes:
        print(f"[ERROR] no weapon datasets under {args.root}/weapon_ds/node*"); return
    print(f"Pool: nodes {sorted(nodes)}; {len(links)} links "
          f"{['n%d/%s' % (link_node, tag) for link_node, tag in links]}")

    deploy = exp_per_node(nodes)
    exp_per_link(nodes)
    if not args.skip_cnn:
        exp_combined(nodes, links, os.path.join(args.root, "model_weapon_combined"))
    else:
        print("\n=== C. COMBINED CNN — skipped (--skip-cnn) ===")
    exp_weighted_fusion(nodes, links)
    exp_drift_ablation(nodes, links)

    print("\n=== Retraining canonical per-node ic27 heads -> model_weapon/node* (deployable) ===")
    for node_id in sorted(deploy):
        _, metrics = train_weapon(deploy[node_id], out_dir=f"{args.root}/model_weapon/node{node_id}",
                                  feature_mode="ic27")
        session_logo = (metrics.get("logo") or {}).get("session", {})
        print(f"   node{node_id}: LOGO={session_logo.get('accuracy', float('nan')):.3f} "
              f"(maj {session_logo.get('majority_accuracy', float('nan')):.3f})")


if __name__ == "__main__":
    main()
