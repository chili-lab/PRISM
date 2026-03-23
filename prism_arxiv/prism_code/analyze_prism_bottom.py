#!/usr/bin/env python3
"""
PRISM Bottom-Level Analysis

Usage:
    python analyze_prism_bottom.py --model_npz /path/to/prism_order1.npz --pt_file /path/to/data.pt --output_dir /path/to/output
"""

import os
import argparse
import numpy as np
import torch
from collections import defaultdict, Counter
from typing import Dict, List
import json
import pickle


from prism_lib import (
    CANON_TAGS, coerce_labels_to_ids, load_pt_records,
    load_prism_model,
    preprocess_hidden_states as _lib_preprocess,
    _emission_log_probs as _lib_emission_log_probs,
    gmm_layer_posterior as _lib_posterior,
)

# ---------- Constants ----------
SHORT_TAGS = ["FA", "SR", "AC", "UV", "UN"]
UNKNOWN_ID = 4
EPS = 1e-12


# ---------- Data Loading ----------
def load_prism_model_dict(npz_path: str, device: str = "cpu") -> Dict:
    """Load PRISM model from .npz via prism_lib.load_prism_model.
    """
    model_obj, prep = load_prism_model(npz_path)

    model = {}
    model["C"] = model_obj.C
    model["K"] = model_obj.K
    model["D"] = model_obj.D
    model["top_order"] = model_obj.top_order
    model["top_start"] = model_obj.top.startprob
    model["top_trans"] = model_obj.top.transmat

    model["bottom"] = []
    for c in range(model_obj.C):
        gmm = model_obj.gmm_bottom[c]
        model["bottom"].append({
            "weights": gmm.weights_.cpu().numpy(),
            "means": gmm.means_.cpu().numpy(),
            "variances": gmm.covariances_.cpu().numpy(),
        })

    model["implicit_bridge"] = model_obj.implicit_bridge
    model["explicit_bridge"] = model_obj.explicit_bridge
    model["joint_bridge"] = model_obj.joint_bridge
    model["num_layers"] = prep["num_layers"]

    dev = torch.device(device)
    model["_gmm_bottom"] = [gmm.to(dev) for gmm in model_obj.gmm_bottom]
    model["_prep"] = prep
    model["_device"] = dev

    # Load PCA explained variance ratio from npz
    npz_data = np.load(npz_path, allow_pickle=True)
    if "prep_global_pca_explained_variance_ratio" in npz_data:
        model["prep_global_pca_explained_variance_ratio"] = npz_data["prep_global_pca_explained_variance_ratio"]

    return model


def preprocess_hidden_states(hs, model: Dict,
                             skip_embedding: bool = True,
                             cat_id: int = None) -> np.ndarray:
    """Preprocess hidden states via prism_lib."""
    return _lib_preprocess(hs, model["_prep"], skip_embedding=skip_embedding)


# ---------- Batch emission precomputation ----------
def precompute_all_emissions(records, model, label_key="sentence_labels"):
    """Batch-precompute emission log-probs.
    """
    dev = model["_device"]
    C = model["C"]

    # Phase 1: Preprocess hidden states, group by category
    items = {c: [] for c in range(C)}  # c -> [(rec_id, t, x_np)]
    labels_map = {}   # rec_id -> list[int]
    x_map = {}        # (rec_id, t) -> x_np [L, D]

    for rec in records:
        if "error" in rec:
            continue
        labels_raw = rec.get(label_key, [])
        hs_list = rec.get("step_hidden_states", [])
        if not labels_raw or not hs_list:
            continue
        rec_id = id(rec)
        labels = coerce_labels_to_ids(labels_raw)
        if len(labels) != len(hs_list):
            continue
        labels_map[rec_id] = labels
        for t, (label, hs) in enumerate(zip(labels, hs_list)):
            if hs is None or label == UNKNOWN_ID or label >= C:
                continue
            if isinstance(hs, torch.Tensor):
                hs = hs.cpu().float().numpy()
            try:
                x = preprocess_hidden_states(hs, model, cat_id=label)
            except Exception:
                continue
            x_map[(rec_id, t)] = x
            items[label].append((rec_id, t, x))

    # Phase 2: Log-weights per category
    log_weights = []
    for c in range(C):
        gmm = model["_gmm_bottom"][c]
        log_weights.append(torch.log(gmm.weights_.to(dev) + 1e-20))

    # Phase 3: Emission computation per category
    emissions = {}  # (rec_id, t) -> B tensor [L, K] on device
    for c in range(C):
        if not items[c]:
            continue
        gmm = model["_gmm_bottom"][c]
        refs = [(rid, t) for rid, t, _ in items[c]]
        all_x = [x for _, _, x in items[c]]
        lengths = [x.shape[0] for x in all_x]

        X_flat = np.concatenate(all_x, axis=0)
        X_gpu = torch.from_numpy(X_flat).float().to(dev)

        B_all = _lib_emission_log_probs(gmm, X_gpu)

        offset = 0
        for idx, (rid, t) in enumerate(refs):
            L = lengths[idx]
            emissions[(rid, t)] = B_all[offset:offset + L]
            offset += L
        del X_gpu, B_all

    n_steps = len(emissions)
    print(f"  Precomputed {n_steps} step emissions on {dev}")
    return {
        "emissions": emissions,
        "log_weights": log_weights,
        "x_preprocessed": x_map,
        "labels": labels_map,
    }


# ── Cache save / load helpers ─────────────────────────────────────────────────

def _save_emissions_cache(cache, records, path):
    """Save emissions cache."""
    id_to_idx = {id(rec): i for i, rec in enumerate(records)}

    emissions_stable = {}
    for (rec_id, t), tensor in cache["emissions"].items():
        idx = id_to_idx.get(rec_id)
        if idx is not None:
            emissions_stable[(idx, t)] = tensor.cpu()

    x_stable = {}
    for (rec_id, t), arr in cache["x_preprocessed"].items():
        idx = id_to_idx.get(rec_id)
        if idx is not None:
            x_stable[(idx, t)] = arr

    labels_stable = {}
    for rec_id, labels in cache["labels"].items():
        idx = id_to_idx.get(rec_id)
        if idx is not None:
            labels_stable[idx] = labels

    log_weights_cpu = [lw.cpu() for lw in cache["log_weights"]]

    torch.save({
        "emissions": emissions_stable,
        "log_weights": log_weights_cpu,
        "x_preprocessed": x_stable,
        "labels": labels_stable,
    }, path)
    print(f"  Saved emissions cache to {path}")


def _load_emissions_cache(path, records, device):
    """Load emissions cache."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    idx_to_id = {i: id(rec) for i, rec in enumerate(records)}

    emissions = {}
    for (idx, t), tensor in data["emissions"].items():
        rec_id = idx_to_id.get(idx)
        if rec_id is not None:
            emissions[(rec_id, t)] = tensor.to(device)

    x_preprocessed = {}
    for (idx, t), arr in data["x_preprocessed"].items():
        rec_id = idx_to_id.get(idx)
        if rec_id is not None:
            x_preprocessed[(rec_id, t)] = arr

    labels = {}
    for idx, lab in data["labels"].items():
        rec_id = idx_to_id.get(idx)
        if rec_id is not None:
            labels[rec_id] = lab

    log_weights = [lw.to(device) for lw in data["log_weights"]]

    n_steps = len(emissions)
    print(f"  Loaded emissions cache from {path} ({n_steps} steps, device={device})")
    return {
        "emissions": emissions,
        "log_weights": log_weights,
        "x_preprocessed": x_preprocessed,
        "labels": labels,
    }


def _save_decoded_seqs(decoded_seqs, path):
    """Save decoded sequences."""
    with open(path, "wb") as f:
        pickle.dump(decoded_seqs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {len(decoded_seqs)} decoded sequences to {path}")


def _load_decoded_seqs(path):
    """Load decoded sequences."""
    with open(path, "rb") as f:
        decoded_seqs = pickle.load(f)
    print(f"  Loaded {len(decoded_seqs)} decoded sequences from {path}")
    return decoded_seqs


def _argmax_from_emissions(B, log_w, sp_override=None):
    """Per-layer argmax from pre-computed emissions on GPU."""
    log_pi_0 = (torch.log(sp_override + 1e-20)
                if sp_override is not None else log_w)
    log_resp = torch.empty_like(B)
    log_resp[0] = log_pi_0 + B[0]
    if B.shape[0] > 1:
        log_resp[1:] = log_w.unsqueeze(0) + B[1:]
    z = torch.argmax(log_resp, dim=1)
    best_ll = float(log_resp.max(dim=1).values.sum())
    return best_ll, z.cpu().numpy()


def _posterior_from_emissions(B, log_w, sp_override=None):
    """Per-layer GMM posterior."""
    logZ, gamma = _lib_posterior(B, log_w, startprob_override=sp_override)
    return float(logZ), gamma.cpu().numpy()


def _bridge_sp_override(prev_exit_gamma, prev_c, label, K, dev,
                        joint_bridge=None):
    """Compute joint bridge startprob override.
    """
    if prev_exit_gamma is None or prev_c is None:
        return None
    if joint_bridge is not None:
        raw = prev_exit_gamma @ joint_bridge[prev_c, :, label, :]
    else:
        import warnings
        warnings.warn("joint_bridge is None — bridge override disabled, falling back to standard GMM weights.")
        return None
    raw_sum = raw.sum()
    sp = raw / raw_sum if raw_sum > EPS else np.ones(K, dtype=np.float64) / K
    return torch.from_numpy(sp).float().to(dev)


def _decode_one_record(rec: Dict, model: Dict, label_key: str, cache=None) -> Dict:
    """Decode bottom-level regimes for a single record.
    """
    if "error" in rec:
        return None

    rec_id = id(rec)
    use_cache = cache is not None and rec_id in cache["labels"]

    if use_cache:
        labels = cache["labels"][rec_id]
    else:
        labels_raw = rec.get(label_key, [])
        hidden_states_list = rec.get("step_hidden_states", [])
        if not labels_raw or not hidden_states_list:
            return None
        labels = coerce_labels_to_ids(labels_raw)
        if len(labels) != len(hidden_states_list):
            raise ValueError(
                f"Record: len(labels)={len(labels)} != "
                f"len(step_hidden_states)={len(hidden_states_list)}")

    seq_result = {
        "labels": labels,
        "is_correct": rec.get("is_correct", None),
        "question": rec.get("question", ""),
        "sample_idx": rec.get("sample_idx"),
        "regimes_per_step": [],
        "log_likelihoods": [],
    }

    joint_bridge = model.get("joint_bridge", None)
    dev = model.get("_device", "cpu")
    K = model["K"]
    prev_exit_gamma = None
    prev_c = None

    for t, label in enumerate(labels):
        if label == UNKNOWN_ID or label >= model["C"]:
            prev_exit_gamma = None
            prev_c = None
            continue

        # Get emission log-probs
        if use_cache:
            key = (rec_id, t)
            if key not in cache["emissions"]:
                prev_exit_gamma = None
                prev_c = None
                continue
            B = cache["emissions"][key]
            log_w = cache["log_weights"][label]
        else:
            hs = hidden_states_list[t]
            if hs is None:
                prev_exit_gamma = None
                prev_c = None
                continue
            if isinstance(hs, torch.Tensor):
                hs = hs.cpu().float().numpy()
            try:
                x = preprocess_hidden_states(hs, model, cat_id=label)
            except Exception:
                prev_exit_gamma = None
                prev_c = None
                continue
            x_t = torch.from_numpy(x).float().to(dev)
            B = _lib_emission_log_probs(model["_gmm_bottom"][label], x_t)
            log_w = torch.log(model["_gmm_bottom"][label].weights_.to(dev) + 1e-20)

        # Bridge startprob override
        sp = _bridge_sp_override(prev_exit_gamma, prev_c, label, K, dev,
                                 joint_bridge=joint_bridge)

        # MAP argmax for hard regime assignment
        ll, regimes = _argmax_from_emissions(B, log_w, sp)
        regimes_list = regimes.tolist()
        L_layers = len(regimes_list)

        final_regime = regimes_list[-1]
        conv_layer = L_layers - 1
        for l in range(L_layers - 2, -1, -1):
            if regimes_list[l] == final_regime:
                conv_layer = l
            else:
                break

        n_regime_changes = sum(1 for l in range(L_layers - 1)
                               if regimes_list[l] != regimes_list[l + 1])

        seq_result["regimes_per_step"].append({
            "step": t,
            "category": label,
            "regimes": regimes_list,
            "log_likelihood": float(ll),
            "convergence_layer": conv_layer,
            "n_regime_changes": n_regime_changes,
        })
        seq_result["log_likelihoods"].append(float(ll))

        if joint_bridge is not None:
            _, gamma = _posterior_from_emissions(B, log_w, sp)
            prev_exit_gamma = gamma[-1].copy()
        prev_c = label

    return seq_result if seq_result["regimes_per_step"] else None


def decode_sequences(records: List[Dict], model: Dict, label_key: str = "sentence_labels",
                     n_jobs: int = -1, cache=None) -> List[Dict]:
    """Decode bottom-level regimes for all records.
    """
    if cache is None:
        cache = precompute_all_emissions(records, model, label_key)
    results = [_decode_one_record(rec, model, label_key, cache=cache)
               for rec in records]
    return [r for r in results if r is not None]


# ============================================================
# 1. Regime Characteristics
# ============================================================

def analyze_regime_characteristics(model: Dict,
                                    records: List[Dict] = None,
                                    label_key: str = "sentence_labels",
                                    cache=None,
                                    soft_posteriors: List[List[Dict]] = None) -> Dict:
    """Analyze regime means, variances, distances per category.
    """
    results = {}
    K = model["K"]
    D = model["D"]

    for c in range(model["C"]):
        if c == UNKNOWN_ID:
            continue

        bottom = model["bottom"][c]
        means = bottom["means"]
        variances = bottom["variances"]

        distances = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                distances[i, j] = np.linalg.norm(means[i] - means[j])

        centroid = np.mean(means, axis=0)

        avg_std = float(np.mean(np.sqrt(np.mean(variances, axis=1))))
        avg_dist = float(np.mean(distances[np.triu_indices(K, 1)])) if K > 1 else 0.0
        results[c] = {
            "category": CANON_TAGS[c],
            "num_regimes": K,
            "dimension": D,
            "regime_mean_norms": np.linalg.norm(means, axis=1).tolist(),
            "regime_avg_variance": np.mean(variances, axis=1).tolist(),
            "avg_inter_regime_distance": avg_dist,
            "regime_spread": float(np.mean(np.linalg.norm(means - centroid, axis=1))),
            "separation_ratio": round(avg_dist / (2 * max(avg_std, 1e-6)), 3),
        }

    # ── 2D — per-category ─────────────────────────────────────
    cats_active = [c for c in range(model["C"]) if c != UNKNOWN_ID and c in results]
    if not (cats_active and K >= 2 and D >= 2):
        return results

    MAX_SAMPLES = 5000

    cat_samples: Dict[int, List] = {c: [] for c in cats_active}
    assert soft_posteriors is not None, "soft_posteriors must be provided"
    for rec_posteriors in soft_posteriors:
        for sd in rec_posteriors:
            label = sd["category"]
            if label == UNKNOWN_ID or label not in cat_samples:
                continue
            if len(cat_samples[label]) >= MAX_SAMPLES:
                continue
            x = sd["x"]          
            gamma = sd["gamma"]  
            L = x.shape[0]
            for ell in range(L):
                if len(cat_samples[label]) >= MAX_SAMPLES:
                    break
                feat = x[ell]  
                regime = int(np.argmax(gamma[ell]))
                cat_samples[label].append((feat, regime))

    for c in cats_active:
        regime_means = model["bottom"][c]["means"]    

        samples_c = cat_samples[c]
        if samples_c:
            sample_feats   = np.array([s[0] for s in samples_c])   
            sample_regimes = np.array([s[1] for s in samples_c])   
        else:
            sample_feats = np.empty((0, D))
            sample_regimes = np.array([], dtype=int)

        if D < 2:
            continue

        results[c]["pca2d_x"] = regime_means[:, 0].tolist()
        results[c]["pca2d_y"] = regime_means[:, 1].tolist()
        ev_ratio = model.get("prep_global_pca_explained_variance_ratio")
        if ev_ratio is not None and len(ev_ratio) >= 2:
            results[c]["pca2d_ev"] = [round(float(ev_ratio[0]), 4), round(float(ev_ratio[1]), 4)]

        if len(sample_feats) >= K:
            results[c]["pca_sample_x"] = sample_feats[:, 0].tolist()
            results[c]["pca_sample_y"] = sample_feats[:, 1].tolist()
            results[c]["pca_sample_regime"] = sample_regimes.tolist()

        pca_regime_std = []
        if len(sample_feats) >= K:
            for k in range(K):
                mask = sample_regimes == k
                pca_regime_std.append(round(float(sample_feats[mask, :2].std(axis=0).mean()) if mask.sum() > 1 else 0.0, 5))
        else:
            pca_regime_std = [0.0] * K
        results[c]["pca2d_regime_std"] = pca_regime_std

    return results


def analyze_regime_top_transitions(model: Dict) -> Dict:
    """Explicit bridge P(c2 | c1, exit_regime=k).
    """
    explicit_bridge = model.get("explicit_bridge")
    if explicit_bridge is None:
        return {}

    K, C_full, C_full2 = explicit_bridge.shape
    assert C_full == C_full2
    cats = [c for c in range(C_full) if c != UNKNOWN_ID]

    per_regime = []
    for k in range(K):
        regime_data = {}
        for c1 in cats:
            c1_tag = SHORT_TAGS[c1]
            dist = explicit_bridge[k, c1]
            regime_data[c1_tag] = {
                "dist": {SHORT_TAGS[c2]: round(float(dist[c2]), 4) for c2 in cats},
            }
        per_regime.append(regime_data)

    return {
        "per_regime": per_regime,
        "K": K,
    }


# ============================================================
# 2. Category Activation Signatures
# ============================================================

def analyze_category_distributions(records: List[Dict], model: Dict,
                                   label_key: str = "sentence_labels",
                                   max_samples: int = 500, cache=None) -> Dict:
    """Compute per-category PCA activation distributions.
    """
    cats = [c for c in range(model["C"]) if c != UNKNOWN_ID]
    cat_set = set(cats)

    cat_vecs = defaultdict(list)

    for rec in records:
        if "error" in rec:
            continue
        rec_id = id(rec)
        use_cache = cache is not None and rec_id in cache["labels"]

        if use_cache:
            labels = cache["labels"][rec_id]
        else:
            labels_raw = rec.get(label_key, [])
            hs_list = rec.get("step_hidden_states", [])
            if len(labels_raw) < 1 or len(hs_list) < 1:
                continue
            labels = coerce_labels_to_ids(labels_raw)

        for t in range(len(labels)):
            cat = labels[t]
            if cat not in cat_set:
                continue
            if use_cache:
                key = (rec_id, t)
                if key not in cache["x_preprocessed"]:
                    continue
                vec = cache["x_preprocessed"][key].mean(axis=0)
            else:
                hs = hs_list[t]
                if hs is None:
                    continue
                try:
                    if isinstance(hs, torch.Tensor):
                        hs = hs.cpu().float().numpy()
                    x_pca = preprocess_hidden_states(hs, model, cat_id=cat)
                    vec = x_pca.mean(axis=0)
                except Exception:
                    continue
            cat_vecs[cat].append(vec)

    results = {}
    for cat in cats:
        tag = SHORT_TAGS[cat] if cat < len(SHORT_TAGS) else f"C{cat}"
        vecs = cat_vecs[cat]
        if len(vecs) < 5:
            continue

        all_vecs = np.array(vecs)  
        cat_result = {}

        cat_result["mean_0"] = round(float(all_vecs[:, 0].mean()), 4)
        cat_result["mean_1"] = round(float(all_vecs[:, 1].mean()), 4)

        n = len(vecs)
        if n > max_samples:
            idx = np.random.choice(n, max_samples, replace=False)
        else:
            idx = np.arange(n)
        cat_result["pca_samples_0"] = [round(float(all_vecs[i, 0]), 3) for i in idx]
        cat_result["pca_samples_1"] = [round(float(all_vecs[i, 1]), 3) for i in idx]
        cat_result["n_steps"] = n

        D = all_vecs.shape[1]
        for dim in range(min(D, 3)):
            col = all_vecs[:, dim]
            counts, edges = np.histogram(col, bins=50)
            cat_result[f"hist_edges_{dim}"] = [round(float(e), 4) for e in edges]
            cat_result[f"hist_counts_{dim}"] = counts.tolist()

        results[tag] = cat_result

    return results


# ============================================================
# 3. Transition Direction Vectors
# ============================================================

def analyze_transition_directions(records: List[Dict], model: Dict,
                                  label_key: str = "sentence_labels",
                                  correct_only: bool = False,
                                  cache=None) -> Dict:
    """Compute direction vectors for category transitions.
    """
    cats = [c for c in range(model["C"]) if c != UNKNOWN_ID]
    cat_set = set(cats)

    exit_sums = defaultdict(lambda: {"sum": np.zeros(model["D"]), "n": 0})
    entry_sums = defaultdict(lambda: {"sum": np.zeros(model["D"]), "n": 0})
    individual_dirs = defaultdict(list)

    for rec in records:
        if "error" in rec:
            continue
        if correct_only and rec.get("is_correct") is not True:
            continue
        labels_raw = rec.get(label_key, [])
        hs_list = rec.get("step_hidden_states", [])
        if len(labels_raw) < 2 or len(hs_list) < 2:
            continue

        labels = coerce_labels_to_ids(labels_raw)
        n = len(labels)

        for i in range(1, min(n, len(hs_list))):
            prev_cat = labels[i-1]
            curr_cat = labels[i]

            if prev_cat == curr_cat:
                continue
            if prev_cat not in cat_set or curr_cat not in cat_set:
                continue

            key = (prev_cat, curr_cat)

            exit_vec = None
            if hs_list[i-1] is not None:
                try:
                    rec_id = id(rec)
                    cache_key = (rec_id, i-1)
                    if cache is not None and cache_key in cache["x_preprocessed"]:
                        x = cache["x_preprocessed"][cache_key]
                    else:
                        hs = hs_list[i-1]
                        if isinstance(hs, torch.Tensor):
                            hs = hs.cpu().float().numpy()
                        x = preprocess_hidden_states(hs, model, cat_id=prev_cat)
                    exit_vec = x.mean(axis=0)
                    exit_sums[key]["sum"] += exit_vec
                    exit_sums[key]["n"] += 1
                except Exception:
                    pass

            entry_vec = None
            if hs_list[i] is not None:
                try:
                    rec_id = id(rec)
                    cache_key = (rec_id, i)
                    if cache is not None and cache_key in cache["x_preprocessed"]:
                        x = cache["x_preprocessed"][cache_key]
                    else:
                        hs = hs_list[i]
                        if isinstance(hs, torch.Tensor):
                            hs = hs.cpu().float().numpy()
                        x = preprocess_hidden_states(hs, model, cat_id=curr_cat)
                    entry_vec = x.mean(axis=0)
                    entry_sums[key]["sum"] += entry_vec
                    entry_sums[key]["n"] += 1
                except Exception:
                    pass

            if exit_vec is not None and entry_vec is not None:
                d = entry_vec - exit_vec
                norm = np.linalg.norm(d)
                if norm > 0:
                    individual_dirs[key].append(d / norm)

    transition_vectors = {}
    direction_map = {}

    for c1 in cats:
        for c2 in cats:
            if c1 == c2:
                continue
            key = (c1, c2)
            tag = f"{SHORT_TAGS[c1]}->{SHORT_TAGS[c2]}"

            n_exit = exit_sums[key]["n"]
            n_entry = entry_sums[key]["n"]
            if n_exit == 0 or n_entry == 0:
                continue

            exit_mean = exit_sums[key]["sum"] / n_exit
            entry_mean = entry_sums[key]["sum"] / n_entry
            direction = entry_mean - exit_mean
            magnitude = float(np.linalg.norm(direction))

            if magnitude < EPS:
                continue

            direction_norm = direction / magnitude
            direction_map[tag] = direction_norm

            entry_result = {
                "magnitude": magnitude,
                "n_transitions": min(n_exit, n_entry),
            }

            dirs = individual_dirs[key]
            if len(dirs) >= 5:
                cosines = [float(np.dot(d, direction_norm)) for d in dirs]
                entry_result["consistency"] = float(np.mean(cosines))
                entry_result["consistency_std"] = float(np.std(cosines))

            ranked = np.argsort(np.abs(direction))[::-1]
            entry_result["top5_dims"] = ranked[:5].tolist()
            entry_result["top5_values"] = direction[ranked[:5]].tolist()

            transition_vectors[tag] = entry_result

    results = {"transition_vectors": transition_vectors}

    tags = list(direction_map.keys())
    cosine_matrix = {}
    for i, t1 in enumerate(tags):
        for t2 in tags[i+1:]:
            cos = float(np.dot(direction_map[t1], direction_map[t2]))
            cosine_matrix[f"{t1} vs {t2}"] = cos
    results["cross_transition_cosine"] = cosine_matrix

    results["_direction_map"] = direction_map

    return results


def analyze_direction_correctness_comparison(records: List[Dict], model: Dict,
                                             label_key: str = "sentence_labels",
                                             cache=None,
                                             td_correct_precomputed=None) -> Dict:
    """Compare mean activation direction at each transition: correct vs incorrect sequences.
    """
    incorrect_recs = [r for r in records if r.get("is_correct") is False and "error" not in r]
    if not incorrect_recs:
        return {}

    if td_correct_precomputed is not None:
        td_corr = td_correct_precomputed
    else:
        correct_recs = [r for r in records if r.get("is_correct") is True and "error" not in r]
        if not correct_recs:
            return {}
        td_corr = analyze_transition_directions(correct_recs, model, label_key=label_key, cache=cache)
    td_incorr = analyze_transition_directions(incorrect_recs, model, label_key=label_key, cache=cache)

    dir_map_corr = td_corr.get("_direction_map", {})
    dir_map_incorr = td_incorr.get("_direction_map", {})
    tvecs_corr = td_corr.get("transition_vectors", {})
    tvecs_incorr = td_incorr.get("transition_vectors", {})

    results = {}
    all_tags = set(dir_map_corr.keys()) | set(dir_map_incorr.keys())

    for tag in sorted(all_tags):
        d_corr = dir_map_corr.get(tag)
        d_incorr = dir_map_incorr.get(tag)
        n_corr = tvecs_corr.get(tag, {}).get("n_transitions", 0)
        n_incorr = tvecs_incorr.get(tag, {}).get("n_transitions", 0)

        entry = {
            "n_correct": n_corr,
            "n_incorrect": n_incorr,
            "magnitude_correct": round(float(tvecs_corr[tag]["magnitude"]), 4) if tag in tvecs_corr else None,
            "magnitude_incorrect": round(float(tvecs_incorr[tag]["magnitude"]), 4) if tag in tvecs_incorr else None,
        }

        if d_corr is not None and d_incorr is not None:
            entry["dir_cosine"] = round(float(np.dot(d_corr, d_incorr)), 4)
        else:
            entry["dir_cosine"] = None
            entry["only_in"] = "correct" if d_corr is not None else "incorrect"

        results[tag] = entry

    cosines = [v["dir_cosine"] for v in results.values() if v.get("dir_cosine") is not None]
    if cosines:
        results["_summary"] = {
            "mean_dir_cosine": round(float(np.mean(cosines)), 4),
            "min_dir_cosine": round(float(np.min(cosines)), 4),
            "n_transitions_compared": len(cosines),
        }

    return results


def convert_for_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(v) for v in obj]
    return obj


# ============================================================
# 4. Per-step hidden-state details for sampled sequences
# ============================================================

def _match_sampled_seq(rep: Dict, decoded_seqs: List[Dict]) -> int:
    """Find decoded sequence index matching by question + sample_idx + labels.
    """
    rep_labels = rep.get("labels", [])
    rep_q = rep.get("question", "").strip()
    rep_idx = rep.get("sample_idx")

    def _check_unique(matches, level):
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous match ({level}): {len(matches)} decoded sequences match "
                f"sample_idx={rep_idx}, question='{rep_q[:50]}...', "
                f"labels_len={len(rep_labels)}")
        return matches[0] if matches else None

    strict = [i for i, ds in enumerate(decoded_seqs)
              if (ds["labels"] == rep_labels
                  and ds.get("question", "").strip() == rep_q
                  and ds.get("sample_idx") == rep_idx)]
    result = _check_unique(strict, "question+sample_idx+labels")
    if result is not None:
        return result

    return -1


def _process_one_sequence(ds, model, correct_regime_dist,
                          correct_path_stats):
    """Process a single decoded sequence.
    """
    steps = []
    for step_info in ds["regimes_per_step"]:
        t = step_info["step"]
        cat = step_info["category"]
        regimes = step_info["regimes"]

        dominant = Counter(regimes).most_common(1)[0][0]

        rp_zscore = None
        if correct_path_stats.get(cat):
            score = 0.0
            n_l = 0
            for l, r in enumerate(regimes):
                dist = correct_regime_dist.get((cat, l))
                if dist is not None and r < len(dist):
                    score += np.log(dist[r])
                    n_l += 1
            if n_l > 0:
                rp_score = score / n_l
                ps = correct_path_stats[cat]
                rp_zscore = (rp_score - ps["mean"]) / ps["std"]

        step_data = {
            "step": t,
            "category": int(cat),
            "category_tag": SHORT_TAGS[cat] if cat < len(SHORT_TAGS) else "?",
            "regime_dominant": int(dominant),
            "regime_path_zscore": round(float(rp_zscore), 3) if rp_zscore is not None else None,
            "regimes": regimes,
            "layer_data": {"regimes_per_layer": regimes},
        }
        steps.append(step_data)

    if not steps:
        return None

    cat_tags = [s["category_tag"] for s in steps]
    cat_counts = Counter(cat_tags)
    total_steps = len(cat_tags)
    category_composition = {tag: round(cnt / total_steps, 4)
                            for tag, cnt in cat_counts.items()}
    first_order_transitions = {}
    for i in range(len(cat_tags) - 1):
        key = f"{cat_tags[i]}->{cat_tags[i+1]}"
        first_order_transitions[key] = first_order_transitions.get(key, 0) + 1
    most_frequent_transition = (
        max(first_order_transitions, key=first_order_transitions.get)
        if first_order_transitions else None
    )

    metadata = {
        "category_composition": category_composition,
        "first_order_transitions": first_order_transitions,
        "most_frequent_transition": most_frequent_transition,
    }
    return steps, metadata


def extract_step_details(
    sampled_sequences: Dict,
    decoded_seqs: List[Dict],
    model: Dict,
    baseline_decoded_seqs: List[Dict] = None,
) -> Dict:
    """Compute per-step regime metrics for sampled sequences.
    """
    baseline_seqs = baseline_decoded_seqs if baseline_decoded_seqs is not None else decoded_seqs

    correct_regime_by_layer = defaultdict(list)
    for ds in baseline_seqs:
        if ds.get("is_correct") is not True:
            continue
        for step_info in ds["regimes_per_step"]:
            for l, r in enumerate(step_info["regimes"]):
                correct_regime_by_layer[(step_info["category"], l)].append(r)

    K = model["K"]
    correct_regime_dist = {} 
    for (cat, layer), regime_list in correct_regime_by_layer.items():
        counts = np.bincount(regime_list, minlength=K).astype(np.float64)
        counts += 1e-6 
        correct_regime_dist[(cat, layer)] = counts / counts.sum()

    correct_path_scores = defaultdict(list)
    for ds in baseline_seqs:
        if ds.get("is_correct") is not True:
            continue
        for step_info in ds["regimes_per_step"]:
            cat = step_info["category"]
            regimes = step_info["regimes"]
            score = 0.0
            n_layers = 0
            for l, r in enumerate(regimes):
                dist = correct_regime_dist.get((cat, l))
                if dist is not None:
                    score += np.log(dist[r])
                    n_layers += 1
            if n_layers > 0:
                correct_path_scores[cat].append(score / n_layers)

    correct_path_stats = {}
    for cat, scores in correct_path_scores.items():
        arr = np.array(scores)
        correct_path_stats[cat] = {"mean": float(arr.mean()), "std": float(max(arr.std(), 1e-6))}

    result = {}
    groups = ["correct", "long_fail", "short_fail"]

    for group in groups:
        reps = sampled_sequences.get(group, [])
        group_details = []
        for rep in reps:
            rep_labels = rep.get("labels", [])
            if not rep_labels:
                continue

            match_idx = _match_sampled_seq(rep, decoded_seqs)
            if match_idx < 0:
                continue

            ds = decoded_seqs[match_idx]
            processed = _process_one_sequence(
                ds, model, correct_regime_dist, correct_path_stats,
            )
            if processed is not None:
                steps, metadata = processed
                group_details.append({
                    "labels": rep_labels,
                    "question_prefix": rep.get("question", "")[:80],
                    "steps": steps,
                    **metadata,
                })

        result[group] = group_details

    return result


# ============================================================
# 5. Step Trajectory Extraction
# ============================================================

def extract_step_trajectories(records: List[Dict], model: Dict,
                               label_key: str = "sentence_labels",
                               n_per_cat: int = 20, cache=None) -> Dict:
    """For a sample of steps with regime transitions, extract per-layer 2D trajectories.
    """
    if cache is None:
        cache = precompute_all_emissions(records, model, label_key)
    cats = [c for c in range(model["C"]) if c != UNKNOWN_ID]
    cat_set = set(cats)
    K = model["K"]
    joint_bridge = model.get("joint_bridge", None)
    dev = model.get("_device", "cpu")

    candidates: Dict[int, List] = {c: [] for c in cats}

    rec_path_lengths = {}
    for rec in records:
        if "error" in rec:
            continue
        rec_id = id(rec)
        labels_raw = rec.get(label_key, [])
        rec_path_lengths[rec_id] = len(labels_raw)

    for rec in records:
        if "error" in rec:
            continue
        rec_id = id(rec)
        if rec_id not in cache["labels"]:
            continue
        labels = cache["labels"][rec_id]
        is_correct = rec.get("is_correct")
        path_length = rec_path_lengths.get(rec_id, 0)

        prev_exit_gamma = None
        prev_c = None

        for t, label in enumerate(labels):
            if label == UNKNOWN_ID or label >= model["C"] or label not in cat_set:
                prev_exit_gamma = None
                prev_c = None
                continue

            key = (rec_id, t)
            if key not in cache["emissions"]:
                prev_exit_gamma = None
                prev_c = None
                continue
            B = cache["emissions"][key]
            log_w = cache["log_weights"][label]
            x = cache["x_preprocessed"][key]

            L = x.shape[0]

            sp = _bridge_sp_override(prev_exit_gamma, prev_c, label, K, dev,
                                     joint_bridge=joint_bridge)

            _, regimes = _argmax_from_emissions(B, log_w, sp)
            regimes = regimes.tolist()

            if joint_bridge is not None:
                _, gamma = _posterior_from_emissions(B, log_w, sp)
                prev_exit_gamma = gamma[-1].copy()
            prev_c = label

            n_trans = sum(1 for l in range(L - 1) if regimes[l] != regimes[l + 1])

            candidates[label].append({
                "step_idx": t,
                "n_layers": L,
                "n_transitions": n_trans,
                "regimes": regimes,
                "x_raw": x,         
                "is_correct": is_correct,
                "path_length": path_length,
            })

    results = {}
    for c in cats:
        cands = candidates[c]
        if not cands:
            continue

        regime_means = model["bottom"][c]["means"] 
        rm_x = regime_means[:, 0].tolist()
        rm_y = regime_means[:, 1].tolist() if regime_means.shape[1] >= 2 else [0.0] * K

        rng = np.random.RandomState(42)
        corr   = [e for e in cands if e["is_correct"] is True]
        incorr = [e for e in cands if e["is_correct"] is False]
        other  = [e for e in cands if e["is_correct"] is None]
        rng.shuffle(corr)
        rng.shuffle(incorr)
        rng.shuffle(other)
        corr   = corr[:10]
        incorr = incorr[:10]
        other  = other[:max(0, n_per_cat - len(corr) - len(incorr))]
        selected = (corr + incorr + other)[:n_per_cat]

        def _regime_segments(regimes):
            """Summarize regime runs."""
            segs = []
            start = 0
            for l in range(1, len(regimes)):
                if regimes[l] != regimes[l - 1]:
                    segs.append((regimes[start], start, l - 1))
                    start = l
            segs.append((regimes[start], start, len(regimes) - 1))
            return " → ".join(f"R{r}({s}-{e})" if s != e else f"R{r}({s})"
                              for r, s, e in segs)

        samples = []
        for entry in selected:
            x_raw = entry["x_raw"]  
            seg_str = _regime_segments(entry["regimes"])
            samples.append({
                "label": (f"step {entry['step_idx']} | "
                          f"{entry['n_transitions']}T {entry['n_layers']}L"),
                "regime_summary": seg_str,
                "is_correct":    entry["is_correct"],
                "step_idx":      entry["step_idx"],
                "n_layers":      entry["n_layers"],
                "n_transitions": entry["n_transitions"],
                "regimes":       entry["regimes"],
                "x":             x_raw[:, 0].round(4).tolist(),
                "y":             (x_raw[:, 1].round(4).tolist()
                                 if x_raw.shape[1] >= 2 else [0.0] * x_raw.shape[0]),
            })

        if not samples:
            continue

        cat_result = {
            "regime_means_x": rm_x,
            "regime_means_y": rm_y,
            "samples":        samples,
        }

        results[SHORT_TAGS[c]] = cat_result

    return results


# ============================================================
# 6. Soft-posterior analyses
# ============================================================

def _get_soft_posteriors_for_record(rec: Dict, model: Dict,
                                    label_key: str, cache=None) -> List[Dict]:
    """Compute soft posteriors.
    """
    if "error" in rec:
        return []

    rec_id = id(rec)
    use_cache = cache is not None and rec_id in cache["labels"]

    if use_cache:
        labels = cache["labels"][rec_id]
    else:
        labels_raw = rec.get(label_key, [])
        hs_list = rec.get("step_hidden_states", [])
        if not labels_raw or not hs_list:
            return []
        labels = coerce_labels_to_ids(labels_raw)

    joint_bridge = model.get("joint_bridge")
    dev = model.get("_device", "cpu")
    K = model["K"]
    results = []

    prev_exit_gamma = None
    prev_c = None

    for t, label in enumerate(labels):
        if label == UNKNOWN_ID or label >= model["C"]:
            prev_exit_gamma = None
            prev_c = None
            continue

        if use_cache:
            key = (rec_id, t)
            if key not in cache["emissions"]:
                prev_exit_gamma = None
                prev_c = None
                continue
            B = cache["emissions"][key]
            log_w = cache["log_weights"][label]
            x = cache["x_preprocessed"][key]
        else:
            hs = hs_list[t]
            if hs is None:
                prev_exit_gamma = None
                prev_c = None
                continue
            if isinstance(hs, torch.Tensor):
                hs = hs.cpu().float().numpy()
            try:
                x = preprocess_hidden_states(hs, model, cat_id=label)
            except Exception:
                prev_exit_gamma = None
                prev_c = None
                continue
            x_t = torch.from_numpy(x).float().to(dev)
            B = _lib_emission_log_probs(model["_gmm_bottom"][label], x_t)
            log_w = torch.log(model["_gmm_bottom"][label].weights_.to(dev) + 1e-20)

        sp = _bridge_sp_override(prev_exit_gamma, prev_c, label, K, dev,
                                 joint_bridge=joint_bridge)
        _, gamma = _posterior_from_emissions(B, log_w, sp)
        results.append({"category": label, "gamma": gamma, "x": x})

        prev_exit_gamma = gamma[-1].copy()
        prev_c = label

    return results


def precompute_soft_posteriors(records: List[Dict], model: Dict,
                              label_key: str = "sentence_labels",
                              cache=None) -> List[List[Dict]]:
    """Precompute soft posteriors.
    """
    all_posteriors = []
    for rec in records:
        step_data = _get_soft_posteriors_for_record(rec, model, label_key, cache=cache)
        is_correct = rec.get("is_correct")
        for sd in step_data:
            sd["is_correct"] = is_correct
        all_posteriors.append(step_data)
    return all_posteriors


def analyze_soft_profiles(records: List[Dict], model: Dict,
                          label_key: str = "sentence_labels",
                          cache=None, soft_posteriors=None,
                          long_fail_threshold: int = 100) -> Dict:
    """Compute regime activation profiles.
    """
    if cache is None:
        cache = precompute_all_emissions(records, model, label_key)
    C = model["C"]
    K = model["K"]

    ALL_GROUPS = ["correct", "incorrect", "long_fail", "short_fail"]
    cat_profiles = {c: {g: [] for g in ALL_GROUPS} for c in range(C)}

    def _classify_record(rec):
        ic = rec.get("is_correct")
        if ic is None:
            return []
        if ic:
            return ["correct"]
        labels = rec.get(label_key, [])
        sub = "long_fail" if len(labels) >= long_fail_threshold else "short_fail"
        return ["incorrect", sub]

    if soft_posteriors is not None:
        for rec, rec_steps in zip(records, soft_posteriors):
            groups = _classify_record(rec)
            if not groups:
                continue
            for sd in rec_steps:
                for g in groups:
                    cat_profiles[sd["category"]][g].append(sd["gamma"])
    else:
        for rec in records:
            groups = _classify_record(rec)
            if not groups:
                continue
            step_data = _get_soft_posteriors_for_record(rec, model, label_key, cache=cache)
            for sd in step_data:
                for g in groups:
                    cat_profiles[sd["category"]][g].append(sd["gamma"])

    results = {}
    for c in range(C):
        tag = SHORT_TAGS[c]
        cat_result = {}

        for group in ALL_GROUPS:
            gammas = cat_profiles[c][group]
            if not gammas:
                continue
            G = np.stack(gammas)   
            mean_profile = G.mean(axis=0)   
            dominant_regime = mean_profile.argmax(axis=1).tolist()  

            cat_result[group] = {
                "n_steps": len(gammas),
                "mean_profile": mean_profile.round(4).tolist(),
                "dominant_regime_per_layer": dominant_regime,
            }

        if "correct" in cat_result and "incorrect" in cat_result:
            prof_c = np.array(cat_result["correct"]["mean_profile"])
            prof_i = np.array(cat_result["incorrect"]["mean_profile"])
            diff = prof_c - prof_i   
            cat_result["diff_correct_minus_incorrect"] = diff.round(4).tolist()
            flat_idx = np.argsort(np.abs(diff).ravel())[::-1]
            top_diffs = []
            for idx in flat_idx[:5]:
                ell, k = divmod(int(idx), K)
                top_diffs.append({
                    "layer": ell, "regime": k,
                    "diff": round(float(diff[ell, k]), 4),
                })
            cat_result["top_discriminative"] = top_diffs

        results[tag] = cat_result

    return results


def run_analysis(model_path: str, pt_path, output_dir: str,
                 analysis_json: str = None,
                 label_key: str = "sentence_labels",
                 n_jobs: int = -1,
                 device: str = "cpu",
                 no_cache: bool = False) -> Dict:
    """Run full bottom-level analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = load_prism_model_dict(model_path, device=device)
    print(f"  Device: {model['_device']}")
    print(f"  C={model['C']}, K={model['K']}, D={model['D']}  (unknown excluded)")

    pt_paths = [pt_path] if isinstance(pt_path, str) else list(pt_path)
    records = []
    for p in pt_paths:
        r = load_pt_records(p)
        records.extend(r)
        print(f"  {p}: {len(r)} records")
    print(f"Loaded {len(records)} records total from {len(pt_paths)} file(s)")

    has_bridge = model.get("bridge") is not None

    results = {
        "model_info": {
            "C": model["C"], "K": model["K"], "D": model["D"],
            "top_order": model["top_order"],
            "num_layers": model.get("num_layers", 0),
            "has_bridge": has_bridge,
        }
    }

    emissions_cache_path = os.path.join(output_dir, "emissions_cache.pt")
    if not no_cache and os.path.exists(emissions_cache_path):
        print(f"Loading cached emissions from {emissions_cache_path}...")
        emissions_cache = _load_emissions_cache(emissions_cache_path, records, model["_device"])
    else:
        print("Precomputing GPU emissions for all records...")
        emissions_cache = precompute_all_emissions(records, model, label_key)
        if not no_cache:
            _save_emissions_cache(emissions_cache, records, emissions_cache_path)

    n_steps = 9

    step = 0
    step += 1
    print(f"\n[{step}/{n_steps}] Analyzing regime characteristics (deferred until soft posteriors ready)...")

    step += 1
    decoded_cache_path = os.path.join(output_dir, "decoded_seqs.pkl")
    if not no_cache and os.path.exists(decoded_cache_path):
        print(f"[{step}/{n_steps}] Loading cached decoded sequences...")
        decoded_seqs = _load_decoded_seqs(decoded_cache_path)
    else:
        print(f"[{step}/{n_steps}] Decoding sequences...")
        decoded_seqs = decode_sequences(records, model, n_jobs=n_jobs, cache=emissions_cache)
        print(f"  Decoded {len(decoded_seqs)} sequences")
        if not no_cache:
            _save_decoded_seqs(decoded_seqs, decoded_cache_path)

    step += 1
    print(f"[{step}/{n_steps}] Analyzing explicit bridge p(c2|c1,k)...")
    results["explicit_bridge"] = analyze_regime_top_transitions(model)
    rtt = results["explicit_bridge"]
    if rtt:
        print(f"  K={rtt['K']} regimes, {len(rtt['per_regime'])} regime entries")
    else:
        print("  No explicit_bridge in model, skipping")

    step += 1
    print(f"[{step}/{n_steps}] Extracting step trajectories...")
    results["step_trajectories"] = extract_step_trajectories(
        records, model, label_key=label_key, n_per_cat=20, cache=emissions_cache)
    n_traj = sum(len(v["samples"]) for v in results["step_trajectories"].values())
    print(f"  {n_traj} trajectory samples across {len(results['step_trajectories'])} categories")

    step += 1
    print(f"[{step}/{n_steps}] Analyzing transition directions...")
    td_result = analyze_transition_directions(records, model, cache=emissions_cache)
    results["transition_directions"] = td_result
    td_result.pop("_direction_map", {})  

    td_correct = analyze_transition_directions(records, model, correct_only=True, cache=emissions_cache)
    print(f"  Correct-only direction map (population): {len(td_correct.get('_direction_map', {}))} transitions")

    step += 1
    print(f"[{step}/{n_steps}] Comparing transition directions: correct vs incorrect...")
    dcc = analyze_direction_correctness_comparison(records, model, label_key=label_key, cache=emissions_cache, td_correct_precomputed=td_correct)
    td_direction_map_correct = td_correct.pop("_direction_map", {})
    results["direction_correctness_comparison"] = dcc
    if dcc:
        summary = dcc.get("_summary", {})
        print(f"  {summary.get('n_transitions_compared', 0)} transitions compared, "
              f"mean_cos={summary.get('mean_dir_cosine', float('nan')):.3f}, "
              f"min_cos={summary.get('min_dir_cosine', float('nan')):.3f}")

    step += 1
    print(f"[{step}/{n_steps}] Computing category activation distributions...")
    results["category_distributions"] = analyze_category_distributions(records, model, cache=emissions_cache)
    for tag, d in results["category_distributions"].items():
        print(f"  {tag}: {d['n_steps']} steps, sampled {len(d.get('pca_samples_0', []))} points")

    # --- Soft-posterior analysis ---
    print("  Precomputing soft posteriors for population records...")
    soft_posteriors = precompute_soft_posteriors(records, model, label_key=label_key, cache=emissions_cache)

    print(f"  Analyzing regime characteristics...")
    results["regime_characteristics"] = analyze_regime_characteristics(
        model, records=records, label_key=label_key, cache=emissions_cache,
        soft_posteriors=soft_posteriors)

    step += 1
    print(f"[{step}/{n_steps}] Computing soft activation profiles (correct vs incorrect)...")
    results["soft_profiles"] = analyze_soft_profiles(records, model, label_key=label_key, cache=emissions_cache, soft_posteriors=soft_posteriors)
    for tag, sp in results["soft_profiles"].items():
        groups = [g for g in ["correct", "incorrect"] if g in sp]
        grp_str = ", ".join(f"{g}: n={sp[g]['n_steps']}" for g in groups)
        disc = sp.get("top_discriminative", [])
        disc_str = ""
        if disc:
            disc_str = " | top diff: " + ", ".join(
                f"L{d['layer']}R{d['regime']}({d['diff']:+.3f})" for d in disc[:3])
        print(f"  {tag}: {grp_str}{disc_str}")

    step += 1
    if analysis_json and os.path.exists(analysis_json):
        print(f"[{step}/{n_steps}] Extracting per-step details for sampled sequences...")
        with open(analysis_json) as f:
            top_analysis = json.load(f)
        rep_seqs = top_analysis.get("sampled_sequences")
        if rep_seqs:
            match_decoded = decoded_seqs
            results["sampled_step_details"] = extract_step_details(
                rep_seqs, match_decoded, model,
                baseline_decoded_seqs=decoded_seqs,
            )
            n_details = sum(
                len(results["sampled_step_details"].get(g, []))
                for g in ["correct", "long_fail", "short_fail"]
            )
            print(f"  Extracted details for {n_details} sampled sequences")
        else:
            print("  No sampled_sequences in analysis.json, skipping.")
    else:
        if analysis_json:
            print(f"[SKIP] analysis.json not found: {analysis_json}")
        else:
            print("[SKIP] No --analysis_json provided, skipping step details.")

    results_path = os.path.join(output_dir, "bottom_analysis.json")
    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\n[SAVED] {results_path}")

    return results


def print_summary(results: Dict):
    """Print a summary."""
    print("\n" + "="*60)
    print("BOTTOM-LEVEL ANALYSIS SUMMARY")
    print("="*60)

    info = results["model_info"]
    print(f"\nModel: C={info['C']} categories, K={info['K']} regimes, D={info['D']} dims")
    print(f"  (unknown excluded from all results)")

    print("\n--- Regime Characteristics ---")
    for c, data in results["regime_characteristics"].items():
        if isinstance(c, int) and c != UNKNOWN_ID:
            print(f"  {data['category']}:")
            print(f"    Avg inter-regime distance: {data['avg_inter_regime_distance']:.3f}")
            print(f"    Regime spread: {data['regime_spread']:.3f}")

    if "transition_directions" in results:
        td = results["transition_directions"]
        print("\n--- Transition Direction Vectors ---")
        if "transition_vectors" in td:
            for tag, info in td["transition_vectors"].items():
                parts = [f"|d|={info['magnitude']:.2f}", f"n={info['n_transitions']}"]
                if "consistency" in info:
                    parts.append(f"consistency={info['consistency']:.3f}")
                print(f"  {tag}: {', '.join(parts)}")
        if "cross_transition_cosine" in td and td["cross_transition_cosine"]:
            cosines = td["cross_transition_cosine"]
            sorted_cos = sorted(cosines.items(), key=lambda x: x[1])
            print("  Cross-transition cosine (all pairs, sorted):")
            for pair, cos in sorted_cos:
                print(f"    {pair}: {cos:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PRISM bottom-level (regime) patterns")
    parser.add_argument("--model_npz", type=str, required=True, help="Path to PRISM .npz model file")
    parser.add_argument("--pt_file", type=str, required=True, nargs="+", help="Path(s) to .pt data file(s)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--label_key", type=str, default="sentence_labels", help="Key for labels in records")
    parser.add_argument("--analysis_json", type=str, default=None,
                        help="Path to analysis.json from analyze_prism_top.py (for sampled sequence details)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs for sequence decoding (-1 = all CPUs)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for computation (cpu/cuda)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Force recomputation of emissions and decoded sequences (ignore cached files)")
    args = parser.parse_args()

    results = run_analysis(args.model_npz, args.pt_file, args.output_dir,
                           analysis_json=args.analysis_json,
                           label_key=args.label_key,
                           n_jobs=args.n_jobs,
                           device=args.device,
                           no_cache=args.no_cache)
    print_summary(results)


if __name__ == "__main__":
    main()
