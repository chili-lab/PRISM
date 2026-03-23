#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRISM Library.
"""

import copy

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from tgmm import GaussianMixture as TorchGMM


# ---------- numerics ----------
EPS = 1e-8

def _to_np(x) -> np.ndarray:
    """Convert torch tensor or array-like to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# ---------- semantic tag space (anchored top) ----------
CANON_TAGS = [
    "final_answer",                 # 0
    "setup_and_retrieval",          # 1
    "analysis_and_computation",     # 2
    "uncertainty_and_verification", # 3
    "unknown",                      # 4
]
CANON_TAG2ID = {t: i for i, t in enumerate(CANON_TAGS)}
UNKNOWN_ID = CANON_TAG2ID["unknown"]   # 4 — skipped during training

def _label_str_to_canon_id(s: str) -> int:
    if s in CANON_TAG2ID:
        return CANON_TAG2ID[s]
    raise KeyError(f"Unknown label string: {s!r}. Valid tags: {list(CANON_TAG2ID.keys())}")

def coerce_labels_to_ids(labels: List[Any]) -> List[int]:
    """Convert string labels to integer IDs (0-4)."""
    out: List[int] = []
    for v in labels:
        if isinstance(v, str):
            out.append(_label_str_to_canon_id(v))
        elif isinstance(v, (int, np.integer)):
            iv = int(v)
            if iv < 0 or iv > 4:
                raise ValueError(f"Numeric label {v} out of range.")
            out.append(iv)
        else:
            raise TypeError(f"Unsupported label type: {type(v)} value={v}")
    return out

# ---------- data adapters ----------
def load_pt_records(pt_path: str):
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    return blob["records"]

def build_top_sequences(records, skip_embedding_layer: bool = True) -> List[Dict]:
    seqs: List[Dict] = []
    for ri, r in enumerate(records):
        hs_list = r.get("step_hidden_states", [])
        labels_raw = r.get("sentence_labels", None)
        sentences = r.get("sentences", None)

        # --- Data integrity checks ---
        n_hs = len(hs_list)
        if labels_raw is not None and len(labels_raw) != n_hs:
            raise ValueError(
                f"Record {ri}: len(step_hidden_states)={n_hs} != "
                f"len(sentence_labels)={len(labels_raw)}")
        if sentences is not None and len(sentences) != n_hs:
            raise ValueError(
                f"Record {ri}: len(step_hidden_states)={n_hs} != "
                f"len(sentences)={len(sentences)}")
        if any(H is None for H in hs_list):
            raise ValueError(
                f"Record {ri}: step_hidden_states contains None entries")

        steps: List[np.ndarray] = []
        for H in hs_list:
            if isinstance(H, torch.Tensor):
                arr = H.to(torch.float32).cpu().numpy()
            else:
                arr = torch.as_tensor(H, dtype=torch.float32).cpu().numpy()
            if skip_embedding_layer and arr.ndim == 2 and arr.shape[0] > 1:
                arr = arr[1:, :]
            steps.append(arr)
        if not steps:
            continue
        seq: Dict[str, Any] = {"steps": steps}
        if labels_raw is not None:
            seq["sentence_labels"] = labels_raw
        seqs.append(seq)
    return seqs

# ---------- Gaussian emission ----------
def _emission_log_probs(gmm: TorchGMM, x: torch.Tensor) -> torch.Tensor:
    """Compute emission log-probs.  [L, D] → [L, K].
    """
    device = x.device
    means = gmm.means_.to(device)       # (K, D)
    variances = gmm.covariances_.to(device)  # (K, D)
    _, D = means.shape

    precisions = 1.0 / (variances + 1e-20)
    log_det = torch.sum(torch.log(variances + 1e-20), dim=1)      # (K,)
    diff = x.unsqueeze(1) - means.unsqueeze(0)                    # (L, K, D)
    mahal = torch.sum(diff.pow(2) * precisions.unsqueeze(0), dim=2)  # (L, K)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=device))
    return -0.5 * (D * log_2pi + log_det.unsqueeze(0) + mahal)


# ======================================================================
#  GMM: tgmm GaussianMixture
# ======================================================================

def _extract_gmm_diag_vars(gmm: TorchGMM) -> np.ndarray:
    """Extract diagonal variances (K, D) from a fitted tgmm GMM (diag only)."""
    return _to_np(gmm.covariances_)


def _precompute_emissions(sequences, gmm_bottom, C, label_key, device):
    """Batch-compute emission log-probs.
    """
    cache = _build_step_cache(sequences, C, label_key, device)
    return _compute_emissions_cached(cache, gmm_bottom)


def _build_step_cache(sequences, C, label_key, device):
    """Upload step data to GPU once.
    """
    device = torch.device(device) if isinstance(device, str) else device

    n_seqs = len(sequences)
    seq_T = [None] * n_seqs  # T for each seq, or None if no labels

    # Group steps by category (skip unknown)
    items = {c: [] for c in range(C)}
    for si, seq in enumerate(sequences):
        labels_raw = seq.get(label_key)
        if labels_raw is None:
            continue
        y = coerce_labels_to_ids(labels_raw)
        if len(seq["steps"]) != len(y):
            raise ValueError(
                f"Sequence {si}: len(steps)={len(seq['steps'])} != "
                f"len(labels)={len(y)}")
        T = len(y)
        seq_T[si] = T
        for t in range(T):
            c = int(y[t])
            if c >= C or c == UNKNOWN_ID:
                continue
            items[c].append((si, t, seq["steps"][t]))

    # Upload per-category data to GPU once
    cat_cache = {}
    for c in range(C):
        if not items[c]:
            cat_cache[c] = None
            continue
        refs = [(si, t) for si, t, _ in items[c]]
        all_x = [item[2] for item in items[c]]
        lengths = [x.shape[0] for x in all_x]
        X_flat = np.concatenate(all_x, axis=0)
        X_gpu = torch.from_numpy(X_flat).float().to(device)
        cat_cache[c] = {"refs": refs, "lengths": lengths, "X_gpu": X_gpu}

    return {"seq_T": seq_T, "cat_cache": cat_cache, "device": device,
            "n_seqs": n_seqs, "C": C}


def _compute_emissions_cached(cache, gmm_bottom):
    """Compute emissions using cached GPU step data and current GMM params.
    """
    device = cache["device"]
    seq_T = cache["seq_T"]
    cat_cache = cache["cat_cache"]
    n_seqs = cache["n_seqs"]
    C = cache["C"]

    # Log weights
    log_weights = []
    for c in range(C):
        w = _to_np(gmm_bottom[c].weights_)
        log_weights.append(torch.from_numpy(np.log(w + 1e-20)).to(
            dtype=torch.float32, device=device))

    # Initialize output
    emissions = [None] * n_seqs
    step_tensors = [None] * n_seqs
    for si in range(n_seqs):
        if seq_T[si] is not None:
            T = seq_T[si]
            emissions[si] = [None] * T
            step_tensors[si] = [None] * T

    # Compute emissions per category
    for c in range(C):
        cc = cat_cache[c]
        if cc is None:
            continue
        gmm = gmm_bottom[c]
        means_gpu = gmm.means_.to(device)       # (K, D)
        vars_gpu = gmm.covariances_.to(device)   # (K, D)
        _, D = means_gpu.shape
        X_gpu = cc["X_gpu"]

        # log_gaussian_diag
        precisions = 1.0 / (vars_gpu + 1e-20)
        log_det = torch.sum(torch.log(vars_gpu + 1e-20), dim=1)        # (K,)
        diff = X_gpu.unsqueeze(1) - means_gpu.unsqueeze(0)             # (N, K, D)
        mahal = torch.sum(diff.pow(2) * precisions.unsqueeze(0), dim=2)  # (N, K)
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=device))
        B_all = -0.5 * (D * log_2pi + log_det.unsqueeze(0) + mahal)

        # Distribute results back to per-(si, t)
        offset = 0
        for idx, (si, t) in enumerate(cc["refs"]):
            L = cc["lengths"][idx]
            emissions[si][t] = B_all[offset:offset + L]
            step_tensors[si][t] = X_gpu[offset:offset + L]
            offset += L

    return emissions, log_weights, step_tensors


def gmm_layer_posterior(
    B: torch.Tensor,
    log_w: torch.Tensor,
    startprob_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GMM posterior.

    Args:
        B: [L, K] emission log-probs on device
        log_w: [K] log mixture weights on device
        startprob_override: [K] bridge-provided prior on device, or None

    Returns:
        logZ: scalar tensor (sum of log-partition)
        gamma: [L, K] float32 posterior responsibilities on device
    """
    L, K = B.shape

    log_pi_0 = (torch.log(startprob_override + 1e-20)
                if startprob_override is not None else log_w)

    log_resp = torch.empty_like(B)
    log_resp[0] = log_pi_0 + B[0]
    if L > 1:
        log_resp[1:] = log_w.unsqueeze(0) + B[1:]

    log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)  # [L, 1]
    gamma = torch.exp(log_resp - log_norm)                      # [L, K] float32
    logZ = log_norm.sum()  # scalar

    return logZ, gamma


def map_decode_gmm(gmm: TorchGMM, x: torch.Tensor,
                       startprob_override: Optional[torch.Tensor] = None,
) -> Tuple[float, np.ndarray]:
    """GMM per-layer argmax.
    """
    B = _emission_log_probs(gmm, x)        # [L, K] on device
    log_w = torch.log(gmm.weights_.to(x.device) + 1e-20)  # [K]
    log_pi_0 = (torch.log(startprob_override + 1e-20)
                if startprob_override is not None else log_w)

    log_resp = torch.empty_like(B)
    log_resp[0] = log_pi_0 + B[0]
    if B.shape[0] > 1:
        log_resp[1:] = log_w.unsqueeze(0) + B[1:]

    z = torch.argmax(log_resp, dim=1)
    best_ll = float(log_resp.max(dim=1).values.sum())
    return best_ll, z.cpu().numpy()


def train_gmm_bottoms(sequences: List[Dict], C: int, K: int,
                      label_key: str = "sentence_labels",
                      seed: int = 0, max_iter: int = 50,
                      n_init: int = 3, verbose: bool = True,
                      device: str = "cpu",
                      init_means: str = "random",
                      ) -> List[TorchGMM]:
    """Train one tgmm GMM per category.
    """
    cat_features: Dict[int, List[np.ndarray]] = {c: [] for c in range(C)}
    for seq in sequences:
        labels_raw = seq.get(label_key, None)
        if labels_raw is None:
            continue
        y = coerce_labels_to_ids(labels_raw)
        steps = seq["steps"]
        T = min(len(y), len(steps))
        for t in range(T):
            c = int(y[t])
            if c >= C or c == UNKNOWN_ID:
                continue
            cat_features[c].append(steps[t])

    gmm_list: List[TorchGMM] = []
    for c in range(C):
        if not cat_features[c]:
            gmm = TorchGMM(n_components=K, covariance_type="diag",
                             random_state=seed, device=device)
            for cc in range(C):
                if cat_features[cc]:
                    X_dummy = np.vstack(cat_features[cc])[:max(K, 1)]
                    gmm.fit(torch.from_numpy(X_dummy).float())
                    break
            gmm_list.append(gmm)
            if verbose:
                tag = CANON_TAGS[c] if c < len(CANON_TAGS) else f"C{c}"
                print(f"  GMM[{tag}]: no data — dummy init")
            continue

        X = np.vstack(cat_features[c])
        gmm = TorchGMM(
            n_components=K,
            covariance_type="diag",
            max_iter=max_iter,
            random_state=seed,
            n_init=n_init,
            init_means=init_means,
            device=device,
        )
        Xt = torch.from_numpy(X).float()
        gmm.fit(Xt)
        if verbose:
            tag = CANON_TAGS[c] if c < len(CANON_TAGS) else f"C{c}"
            score = float(gmm.score(Xt))
            from tgmm import ClusteringMetrics as _CM
            _m = _CM()
            bic_val = float(_m.bic_score(score, Xt, gmm.n_components, gmm.covariance_type))
            labels = gmm.predict(Xt)
            n_unique = len(set(labels.tolist() if hasattr(labels, 'tolist') else list(labels)))
            sil_str = ""
            if n_unique >= 2:
                try:
                    MAX_SIL = 10_000
                    if Xt.shape[0] > MAX_SIL:
                        idx = torch.randperm(Xt.shape[0])[:MAX_SIL]
                        sil_val = float(_m.silhouette_score(Xt[idx], labels[idx], gmm.n_components))
                    else:
                        sil_val = float(_m.silhouette_score(Xt, labels, gmm.n_components))
                    sil_str = f", sil={sil_val:.4f}"
                except Exception:
                    pass
            print(f"  GMM[{tag}]: {X.shape[0]} vectors, K={K}, "
                  f"converged={gmm.converged_}, "
                  f"LL/vec={score:.3f}, BIC={bic_val:.1f}{sil_str}")
        gmm_list.append(gmm)

    return gmm_list


# ======================================================================
#  Top-level transitions
# ======================================================================

def tuple_to_index(tup: Tuple[int, ...], C: int) -> int:
    idx = 0
    for v in tup:
        idx = idx * C + v
    return idx

def index_to_tuple(idx: int, C: int, order: int) -> Tuple[int, ...]:
    tup = []
    for _ in range(order):
        tup.append(idx % C)
        idx //= C
    return tuple(reversed(tup))

@dataclass
class TopParams:
    order: int
    C: int
    startprob: np.ndarray
    transmat: np.ndarray

def init_top_params(C: int, order: int,
                    sticky: float = 0.7) -> TopParams:
    context_size = C ** order
    startprob = np.ones(context_size, dtype=np.float64) / context_size
    transmat = np.ones((context_size, C), dtype=np.float64) / C
    for ctx_idx in range(context_size):
        if C == 1:
            transmat[ctx_idx, 0] = 1.0
            continue
        ctx_tuple = index_to_tuple(ctx_idx, C, order)
        last_cat = ctx_tuple[-1]
        transmat[ctx_idx, :] = (1.0 - sticky) / (C - 1)
        transmat[ctx_idx, last_cat] = sticky
    return TopParams(order, C, startprob, transmat)


# ======================================================================
#  PRISM container
# ======================================================================

@dataclass
class PRISMModel:
    C: int
    K: int
    D: int
    top_order: int
    top: TopParams
    gmm_bottom: List[TorchGMM]             # one GMM per category
    implicit_bridge: Optional[np.ndarray] = None       # [C, C, K, K] P(k2 | c1, c2, k1)
    explicit_bridge: Optional[np.ndarray] = None  # [K, C, C] P(c2 | c1, exit_regime=k)
    joint_bridge: Optional[np.ndarray] = None  # [C, K, C, K] P(c2, k2 | c1, k1)
    converged_: bool = False


def load_prism_model(npz_path: str, device: str = "cpu") -> Tuple["PRISMModel", Dict]:
    """Load PRISM model from NPZ file.
    """
    data = np.load(npz_path, allow_pickle=True)
    C = int(data["C"][0])
    K = int(data["K"][0])
    D = int(data["D"][0])
    top_order = int(data["top_order"][0])

    # Top-level params
    top = TopParams(top_order, C, data["top_start"], data["top_trans"])

    # GMM bottoms
    gmm_bottom = []
    for c in range(C):
        gmm = TorchGMM(n_components=K, covariance_type="diag", device=device)
        gmm.n_features = D
        gmm.fitted_ = True
        gmm.weights_ = torch.from_numpy(data[f"b{c}_weights"].astype(np.float32))
        gmm.means_ = torch.from_numpy(data[f"b{c}_means"].astype(np.float32))
        gmm.covariances_ = torch.from_numpy(data[f"b{c}_vars"].astype(np.float32))
        gmm_bottom.append(gmm)

    # Joint bridge matrix and derived implicit_bridge / explicit_bridge
    joint_bridge = data["joint_bridge"].astype(np.float64) if "joint_bridge" in data else None
    implicit_bridge = data["implicit_bridge"].astype(np.float64) if "implicit_bridge" in data else None
    explicit_bridge = data["explicit_bridge"].astype(np.float64) if "explicit_bridge" in data else None

    model = PRISMModel(C, K, D, top_order, top, gmm_bottom,
                       implicit_bridge=implicit_bridge, explicit_bridge=explicit_bridge,
                       joint_bridge=joint_bridge)

    # Preprocessing params
    num_layers = int(data["num_layers"][0])
    D_in = int(data["D_in"][0])
    layer_mean = np.stack([data[f"prep_L{L}_mean"] for L in range(num_layers)])
    layer_rms = np.array([float(data[f"prep_L{L}_rms"][0]) for L in range(num_layers)])
    per_step_rms = bool(data["meta_per_step_rms"][0]) if "meta_per_step_rms" in data else False

    prep = {
        "num_layers": num_layers,
        "D_in": D_in,
        "layer_mean": layer_mean.astype(np.float64),
        "layer_rms": layer_rms.astype(np.float64),
        "pca_components": data["prep_global_pca_components"].astype(np.float64),
        "pca_mean": data["prep_global_pca_mean"].astype(np.float64),
        "per_step_rms": per_step_rms,
    }
    return model, prep


def preprocess_hidden_states(hs, prep: Dict, skip_embedding: bool = True) -> np.ndarray:
    """Preprocess raw hidden states using saved preprocessing params.

    Args:
        hs: [L+1, D_in] tensor or numpy array (includes embedding layer)
        prep: dict from load_prism_model with layer_mean, layer_rms, etc.
        skip_embedding: if True, skip first row (embedding layer)

    Returns:
        [L, D_pca] numpy array in PCA space
    """
    if isinstance(hs, torch.Tensor):
        hs = hs.cpu().float().numpy()
    if skip_embedding and hs.shape[0] > prep["num_layers"]:
        hs = hs[1:]
    X = hs.astype(np.float64)
    # Per-layer mean subtraction + RMS normalization
    Xc = (X - prep["layer_mean"]) / (prep["layer_rms"][:, None] + EPS)
    # Per-step RMS normalization
    if prep.get("per_step_rms", False):
        D = Xc.shape[1]
        rms_step = np.sqrt(np.mean(np.sum(Xc ** 2, axis=1)) / D)
        Xc = Xc / (rms_step + EPS)
    # PCA
    Xpca = (Xc.astype(np.float32) - prep["pca_mean"]) @ prep["pca_components"].T
    return Xpca.astype(np.float64)


def init_gmm_bottoms(sequences: List[Dict], C: int, K: int,
                     label_key: str = "sentence_labels",
                     seed: int = 0, verbose: bool = True,
                     device: str = "cpu",
                     ) -> List[TorchGMM]:
    """Initialize GMM per category with random data-points.

    - weights: uniform (1/K)
    - means: K random data points from the category
    - covariances: global variance of the category (diag)

    Joint EM in fit_prism handles full training from this init.
    Returns List[TorchGMM] of length C.
    """
    rng = np.random.RandomState(seed)

    cat_features: Dict[int, List[np.ndarray]] = {c: [] for c in range(C)}
    for seq in sequences:
        labels_raw = seq.get(label_key, None)
        if labels_raw is None:
            continue
        y = coerce_labels_to_ids(labels_raw)
        steps = seq["steps"]
        T = min(len(y), len(steps))
        for t in range(T):
            c = int(y[t])
            if c >= C or c == UNKNOWN_ID:
                continue
            cat_features[c].append(steps[t])

    fallback_X = None
    for c in range(C):
        if cat_features[c]:
            fallback_X = np.vstack(cat_features[c])
            break

    gmm_list: List[TorchGMM] = []
    for c in range(C):
        tag = CANON_TAGS[c] if c < len(CANON_TAGS) else f"C{c}"
        gmm = TorchGMM(n_components=K, covariance_type="diag",
                        random_state=seed, device=device)

        if cat_features[c]:
            X = np.vstack(cat_features[c])  # [N*L, D]
        elif fallback_X is not None:
            X = fallback_X
            if verbose:
                print(f"  GMM[{tag}]: no data — using fallback")
        else:
            raise RuntimeError("No data in any category")

        N, D = X.shape
        # Random data-point means
        idx = rng.choice(N, size=min(K, N), replace=(N < K))
        means = torch.from_numpy(X[idx].astype(np.float32))   # [K, D]
        if N < K:
            # Pad with slight jitter for remaining components
            extra = K - N
            base = torch.from_numpy(X[rng.choice(N, size=extra)].astype(np.float32))
            means = torch.cat([means, base + 1e-3 * torch.randn(extra, D)], dim=0)

        # Uniform weights
        weights = torch.ones(K, dtype=torch.float32) / K

        # Global variance of category data (diag)
        Xt = torch.from_numpy(X.astype(np.float32))
        var = Xt.var(dim=0)                                    # [D]
        var = torch.clamp(var, min=1e-6)
        covariances = var.unsqueeze(0).expand(K, -1).clone()   # [K, D]

        gmm.n_features = D
        gmm.weights_ = weights
        gmm.means_ = means
        gmm.covariances_ = covariances

        if verbose:
            print(f"  GMM[{tag}]: {N} vectors, K={K}")

        gmm_list.append(gmm)

    return gmm_list


def init_prism(C: int, K: int, D: int, top_order: int,
                           sequences: List[Dict],
                           label_key: str = "sentence_labels",
                           seed: int = 0, verbose: bool = True,
                           device: str = "cpu",
                           ) -> PRISMModel:
    """Initialize PRISM model with randomly-initialized GMM bottoms.
    """
    top = init_top_params(C, top_order)
    if verbose:
        print(f"\n--- Initializing GMM bottoms (device={device}) ---")
    gmm_bottom = init_gmm_bottoms(sequences, C, K, label_key=label_key,
                                  seed=seed, verbose=verbose, device=device)
    return PRISMModel(C, K, D, top_order, top, gmm_bottom)


def _mstep_gmm_gpu(gmm: TorchGMM, Nk: torch.Tensor, sum_x: torch.Tensor,
                    sum_x2: torch.Tensor):
    """Update TorchGMM params.
    """
    reg_covar = gmm.reg_covar
    nk = Nk + 1e-20
    weights = torch.clamp(nk / nk.sum(), min=1e-20)
    means = sum_x / nk.unsqueeze(1)
    variances = (sum_x2 - sum_x.pow(2) / nk.unsqueeze(1)) / nk.unsqueeze(1)
    variances = torch.clamp(variances, min=0.0) + reg_covar
    gmm.weights_ = weights.float()
    gmm.means_ = means.float()
    gmm.covariances_ = variances.float()


# ======================================================================
#  Training
# ======================================================================

def fit_prism(
    sequences: List[Dict],
    C: int,
    K: int,
    top_order: int = 1,
    label_key: str = "sentence_labels",
    n_iter: int = 10,
    seed: int = 0,
    verbose: bool = True,
    use_bridge: bool = False,
    device: str = "cpu",
    tol: float = 1e-4,
    n_joint_iter: int = 0,
    **kwargs,
) -> Tuple[PRISMModel, List[Dict]]:
    """Train PRISM model.
    """
    # Infer D
    D: Optional[int] = None
    for seq in sequences:
        for x in seq["steps"]:
            D = int(x.shape[1])
            break
        if D is not None:
            break
    assert D is not None, "Empty sequences."

    # Initialize model with random GMM bottoms
    model = init_prism(C, K, D, top_order, sequences,
                                   label_key=label_key, seed=seed,
                                   verbose=verbose, device=device)

    context_size = C ** top_order
    top_start_counts = np.zeros(context_size, dtype=np.float64)
    top_trans_counts = np.zeros((context_size, C), dtype=np.float64)
    for seq in sequences:
        labels_raw = seq.get(label_key, None)
        if labels_raw is None:
            continue
        y_raw = coerce_labels_to_ids(labels_raw)
        # Filter out unknown steps
        y = [c for c in y_raw if c != UNKNOWN_ID and c < C]
        Tsteps = len(y)
        if Tsteps >= top_order:
            init_ctx = tuple(y[:top_order])
            top_start_counts[tuple_to_index(init_ctx, C)] += 1.0
            for t in range(top_order, Tsteps):
                ctx = tuple(y[t - top_order:t])
                top_trans_counts[tuple_to_index(ctx, C), y[t]] += 1.0
        elif Tsteps > 0:
            padded = [y[0]] * (top_order - Tsteps) + y[:Tsteps]
            top_start_counts[tuple_to_index(tuple(padded[:top_order]), C)] += 1.0

    startprob = np.maximum(EPS, top_start_counts)
    startprob /= startprob.sum()
    transmat = np.maximum(EPS, top_trans_counts)
    row_sums = transmat.sum(axis=1, keepdims=True)
    transmat = np.where(row_sums > 0, transmat / row_sums,
                        np.ones_like(transmat) / C)
    model.top = TopParams(top_order, C, startprob, transmat)

    history = []
    device_obj = torch.device(device)

    step_cache = _build_step_cache(sequences, C, label_key, device_obj)

    g_Nk_t = torch.zeros(C, K, dtype=torch.float64, device=device_obj)
    g_sx_t = torch.zeros(C, K, D, dtype=torch.float64, device=device_obj)
    g_sx2_t = torch.zeros(C, K, D, dtype=torch.float64, device=device_obj)
    # Layer-0 Nk for weight correction in joint E-step
    g_Nk_layer0_t = torch.zeros(C, K, dtype=torch.float64, device=device_obj)
    # Joint posterior accumulator for bridge M-step
    g_j_acc = torch.zeros(C, K, C, K, dtype=torch.float64, device=device_obj)

    # Per-step (entry_gamma, exit_gamma) storage for joint_bridge (populated by _estep)
    n_seqs = step_cache["n_seqs"]
    seq_T = step_cache["seq_T"]
    p1_data = None  # p1_data[si][t] = (entry_gamma [K], exit_gamma [K])
    if use_bridge:
        p1_data = [None] * n_seqs
        for si in range(n_seqs):
            if seq_T[si] is not None:
                p1_data[si] = [None] * seq_T[si]

    def _estep(label):
        g_Nk_t.zero_()
        g_sx_t.zero_()
        g_sx2_t.zero_()

        cat_cache = step_cache["cat_cache"]
        seq_T = step_cache["seq_T"]
        n_seqs = step_cache["n_seqs"]

        # Log weights per category
        log_w_list = []
        for c in range(C):
            w = model.gmm_bottom[c].weights_.to(
                dtype=torch.float32, device=device_obj)
            log_w_list.append(torch.log(w + 1e-20))

        total_logZ = 0.0
        n_steps = 0

        # ---- Phase 1: standard posteriors per category ----
        for c in range(C):
            cc = cat_cache[c]
            if cc is None:
                continue

            gmm = model.gmm_bottom[c]
            means_gpu = gmm.means_.to(device_obj)       # [K, D] float32
            vars_gpu = gmm.covariances_.to(device_obj)   # [K, D] float32
            X_gpu = cc["X_gpu"]                          # [N_flat, D] float32
            log_w = log_w_list[c]                        # [K] float32

            # Batch emissions
            precisions = 1.0 / (vars_gpu + 1e-20)
            log_det = torch.sum(torch.log(vars_gpu + 1e-20), dim=1)
            diff = X_gpu.unsqueeze(1) - means_gpu.unsqueeze(0)
            mahal = torch.sum(diff.pow(2) * precisions.unsqueeze(0), dim=2)
            log_2pi = torch.log(
                torch.tensor(2.0 * torch.pi, device=device_obj))
            _, D_gmm = means_gpu.shape
            B_all = -0.5 * (
                D_gmm * log_2pi + log_det.unsqueeze(0) + mahal)  # [N_flat,K]

            # Standard posteriors
            log_resp = log_w.unsqueeze(0) + B_all          # [N_flat, K]
            log_norm = torch.logsumexp(
                log_resp, dim=1, keepdim=True)              # [N_flat, 1]
            gamma = torch.exp(log_resp - log_norm)          # [N_flat, K]

            g_Nk_t[c] += gamma.sum(dim=0)
            g_sx_t[c] += gamma.T @ X_gpu
            g_sx2_t[c] += gamma.T @ (X_gpu.pow(2))

            # Total logZ
            logZ_flat = log_norm.squeeze(1)                 # [N_flat] f32
            total_logZ += logZ_flat.sum().item()
            n_steps += len(cc["refs"])

            # Store (entry_gamma, exit_gamma) per step for joint_bridge counting
            if p1_data is not None:
                offset = 0
                for idx, (si, t) in enumerate(cc["refs"]):
                    L = cc["lengths"][idx]
                    entry_gamma = gamma[offset].clone()          # layer 0, [K]
                    exit_gamma = gamma[offset + L - 1].clone()   # layer L-1, [K]
                    p1_data[si][t] = (entry_gamma, exit_gamma)
                    offset += L

        avg_ll = total_logZ / max(1, n_steps)
        h = {"iter": label, "phase": "gmm", "avg_loglik": float(avg_ll),
             "n_steps": n_steps}

        if verbose:
            print(f"[PRISM-{top_order}order] {label}  "
                  f"avg step loglik = {avg_ll:.4f}", flush=True)

        return h

    def _mstep():
        """Run GMM M-step."""
        for c in range(C):
            if g_Nk_t[c].sum().item() > EPS:
                _mstep_gmm_gpu(model.gmm_bottom[c],
                               g_Nk_t[c], g_sx_t[c], g_sx2_t[c])

    def _count_joint_transitions():
        """Count joint (c2, k2) transitions from p1_data.

        Returns:
            joint_bridge: [C, K, C, K] numpy — P(c2, k2 | c1, k1)
            explicit_bridge: [K, C, C] numpy — P(c2 | c1, k1) (marginalized over k2)
            implicit_bridge: [C, C, K, K] numpy — P(k2 | c1, c2, k1) (conditioned on c2)
        """
        j_acc = torch.zeros(C, K, C, K, dtype=torch.float64, device=device_obj)
        for si, seq in enumerate(sequences):
            if seq_T[si] is None:
                continue
            y = coerce_labels_to_ids(seq.get(label_key))
            T_si = seq_T[si]
            prev_exit_gamma = None
            prev_c = None
            for t in range(T_si):
                c = int(y[t])
                if c == UNKNOWN_ID or c >= C:
                    prev_exit_gamma = None
                    prev_c = None
                    continue
                if prev_exit_gamma is not None and prev_c is not None:
                    entry_gamma = p1_data[si][t][0]  # [K]
                    # outer product: exit_gamma[k1] * entry_gamma[k2]
                    j_acc[prev_c, :, c, :] += prev_exit_gamma.double().unsqueeze(1) * entry_gamma.double().unsqueeze(0)
                prev_exit_gamma = p1_data[si][t][1]  # exit_gamma [K]
                prev_c = c

        # Normalize joint_bridge: each (c1, k1) row sums to 1 over (c2, k2)
        j_flat = j_acc.view(C * K, C * K)
        row_sums = j_flat.sum(dim=1, keepdim=True).clamp(min=EPS)
        j_flat = j_flat / row_sums
        joint_bridge = j_flat.view(C, K, C, K)

        # Derive explicit_bridge: P(c2 | c1, k1) = sum_{k2} joint_bridge[c1, k1, c2, k2]
        explicit_bridge = joint_bridge.sum(dim=3)  # [C, K, C]
        explicit_bridge_out = explicit_bridge.permute(1, 0, 2)  # [K, C, C]

        # Derive implicit_bridge: P(k2 | c1, c2, k1) = joint_bridge[c1, k1, c2, k2] / P(c2 | c1, k1)
        implicit_bridge_out = joint_bridge / explicit_bridge.unsqueeze(3).clamp(min=EPS)  # [C, K, C, K]
        # Reshape to [C, C, K, K]: bridge[c1, c2, k1, k2]
        implicit_bridge_out = implicit_bridge_out.permute(0, 2, 1, 3)  # [C, C, K, K]

        return (joint_bridge.cpu().numpy().astype(np.float64),
                explicit_bridge_out.cpu().numpy().astype(np.float64),
                implicit_bridge_out.cpu().numpy().astype(np.float64))

    # ---- Phase 1: GMM EM ----
    phase1_label = "Phase 1 (GMM)" if use_bridge else "GMM-only"
    if verbose:
        print(f"\n  [{phase1_label}] Training GMMs for {n_iter} iterations")
    h = _estep("init")
    history.append(h)
    prev_lower_bound = h["avg_loglik"]

    for it in range(1, n_iter + 1):
        _mstep()
        h = _estep(f"iter {it:02d}")
        history.append(h)
        avg_ll = h["avg_loglik"]
        rel_change = abs(avg_ll - prev_lower_bound) / (abs(prev_lower_bound) + 1e-20)
        if rel_change < tol:
            _mstep()
            h_final = _estep("final")
            history.append(h_final)
            model.converged_ = True
            if verbose:
                print(f"  Convergence at iteration {it} "
                      f"(rel_change={rel_change:.2e} < tol={tol:.1e}), "
                      f"ran final M+E step; model uses iteration {it+1} params")
            break
        prev_lower_bound = avg_ll

    for gmm in model.gmm_bottom:
        gmm.fitted_ = True

    # ---- Phase 2: Bridge estimation + optional joint EM ----
    if use_bridge:
        if verbose:
            print(f"\n  [Phase 2] Computing joint transition matrix "
                  f"P(c2,k2|c1,k1): [{C},{K},{C},{K}] = {C*K*C*K} entries")
        joint, explicit, implicit = _count_joint_transitions()
        model.joint_bridge = joint
        model.explicit_bridge = explicit
        model.implicit_bridge = implicit

        # -- Build cat_index for sequential access in joint E-step --
        cat_cache = step_cache["cat_cache"]
        cat_index = {}
        for c in range(C):
            cc = cat_cache[c]
            if cc is None:
                continue
            lookup = {}
            offset = 0
            for idx, (si, t) in enumerate(cc["refs"]):
                L = cc["lengths"][idx]
                lookup[(si, t)] = (offset, L)
                offset += L
            cat_index[c] = lookup

        # -- Precompute per-sequence label arrays --
        seq_labels = [None] * n_seqs
        for si, seq in enumerate(sequences):
            if seq_T[si] is not None:
                seq_labels[si] = coerce_labels_to_ids(seq.get(label_key))

        implicit_t = torch.from_numpy(implicit).to(
            dtype=torch.float32, device=device_obj)
        joint_t = torch.from_numpy(joint).to(
            dtype=torch.float32, device=device_obj)  # [C, K, C, K]

        def _estep_joint(label):
            """Joint E-step:

            Layer 0 uses joint-derived prior from previous step's exit gamma.
            Layers 1+ use standard log_w.

            Also accumulates:
            - g_j_acc: joint posterior P(k1,k2|data) for bridge M-step
            - g_Nk_layer0_t: layer-0 Nk for weight correction
            """
            g_Nk_t.zero_()
            g_sx_t.zero_()
            g_sx2_t.zero_()
            g_Nk_layer0_t.zero_()
            g_j_acc.zero_()

            # Step A: Batch-compute emissions per category
            cat_B = {}  # cat_B[c] = [N_flat, K] emission log-probs
            log_w_list = []
            for c in range(C):
                w = model.gmm_bottom[c].weights_.to(
                    dtype=torch.float32, device=device_obj)
                log_w_list.append(torch.log(w + 1e-20))

            for c in range(C):
                cc = cat_cache[c]
                if cc is None:
                    continue
                gmm = model.gmm_bottom[c]
                means_gpu = gmm.means_.to(device_obj)
                vars_gpu = gmm.covariances_.to(device_obj)
                X_gpu = cc["X_gpu"]

                precisions = 1.0 / (vars_gpu + 1e-20)
                log_det = torch.sum(torch.log(vars_gpu + 1e-20), dim=1)
                diff = X_gpu.unsqueeze(1) - means_gpu.unsqueeze(0)
                mahal = torch.sum(diff.pow(2) * precisions.unsqueeze(0), dim=2)
                log_2pi = torch.log(
                    torch.tensor(2.0 * torch.pi, device=device_obj))
                _, D_gmm = means_gpu.shape
                B_all = -0.5 * (
                    D_gmm * log_2pi + log_det.unsqueeze(0) + mahal)
                cat_B[c] = B_all

            # Step B: Sequential posterior with bridge
            total_logZ = 0.0
            n_steps_j = 0

            for si in range(n_seqs):
                if seq_T[si] is None:
                    continue
                y = seq_labels[si]
                T_si = seq_T[si]
                prev_exit_gamma = None
                prev_c = None

                for t in range(T_si):
                    c = int(y[t])
                    if c == UNKNOWN_ID or c >= C:
                        prev_exit_gamma = None
                        prev_c = None
                        continue

                    cc = cat_cache[c]
                    off, L = cat_index[c][(si, t)]
                    B_step = cat_B[c][off:off + L]   # [L, K]
                    X_step = cc["X_gpu"][off:off + L] # [L, D]
                    log_w = log_w_list[c]

                    # Joint-derived startprob for layer 0
                    # P(k2|data) ∝ Σ_k1 γ_exit[k1] * J[c1,k1,c2,k2]
                    has_bridge = (prev_exit_gamma is not None
                                 and prev_c is not None)
                    sp_override = None
                    if has_bridge:
                        raw = prev_exit_gamma @ joint_t[prev_c, :, c, :]
                        raw_sum = raw.sum()
                        if raw_sum.item() > EPS:
                            sp_override = raw / raw_sum
                        else:
                            has_bridge = False

                    log_pi_0 = (torch.log(sp_override + 1e-20)
                                if sp_override is not None else log_w)

                    log_resp = torch.empty_like(B_step)
                    log_resp[0] = log_pi_0 + B_step[0]
                    if L > 1:
                        log_resp[1:] = log_w.unsqueeze(0) + B_step[1:]

                    log_norm = torch.logsumexp(
                        log_resp, dim=1, keepdim=True)
                    gamma = torch.exp(log_resp - log_norm)

                    # Accumulate GMM stats (all layers for means/vars)
                    g_Nk_t[c] += gamma.sum(dim=0)
                    g_sx_t[c] += gamma.T @ X_step
                    g_sx2_t[c] += gamma.T @ (X_step.pow(2))

                    # track layer-0 Nk separately
                    g_Nk_layer0_t[c] += gamma[0].double()

                    total_logZ += log_norm.sum().item()
                    n_steps_j += 1

                    # accumulate joint posterior P(k1,k2|data)
                    # for bridge M-step
                    if has_bridge:
                        # P(k1,k2|data) ∝ γ_exit[k1] * J[c1,k1,c2,k2] * exp(B_entry[k2])
                        log_joint = (
                            torch.log(prev_exit_gamma + 1e-20).unsqueeze(1)
                            + torch.log(joint_t[prev_c, :, c, :] + 1e-20)
                            + B_step[0].unsqueeze(0)
                        )  # [K, K]
                        joint_k1k2 = torch.exp(
                            log_joint - torch.logsumexp(
                                log_joint.reshape(-1), dim=0))
                        g_j_acc[prev_c, :, c, :] += joint_k1k2.double()

                    # Store entry/exit gammas
                    if p1_data is not None:
                        p1_data[si][t] = (gamma[0].clone(),
                                          gamma[-1].clone())

                    prev_exit_gamma = gamma[-1].clone()
                    prev_c = c

            avg_ll = total_logZ / max(1, n_steps_j)
            h = {"iter": label, "phase": "joint",
                 "avg_loglik": float(avg_ll), "n_steps": n_steps_j}
            if verbose:
                print(f"[PRISM-{top_order}order] {label}  "
                      f"avg step loglik = {avg_ll:.4f}", flush=True)
            return h

        def _mstep_bridge():
            """Re-estimate bridge from joint posterior g_j_acc."""
            nonlocal implicit_t, joint_t

            # Normalize: each (c1, k1) row sums to 1 over (c2, k2)
            j_flat = g_j_acc.view(C * K, C * K)
            row_sums = j_flat.sum(dim=1, keepdim=True).clamp(min=EPS)
            j_flat = j_flat / row_sums
            joint_bridge = j_flat.view(C, K, C, K)

            # Derive explicit_bridge: P(c2 | c1, k1)
            explicit_bridge = joint_bridge.sum(dim=3)  # [C, K, C]
            explicit_bridge_out = explicit_bridge.permute(1, 0, 2)  # [K, C, C]

            # Derive bridge: P(k2 | c1, c2, k1)
            implicit_bridge_out = joint_bridge / explicit_bridge.unsqueeze(3).clamp(min=EPS)
            implicit_bridge_out = implicit_bridge_out.permute(0, 2, 1, 3)  # [C, C, K, K]

            joint_np = joint_bridge.cpu().numpy().astype(np.float64)
            explicit_bridge_np = explicit_bridge_out.cpu().numpy().astype(np.float64)
            implicit_bridge_np = implicit_bridge_out.cpu().numpy().astype(np.float64)

            model.joint_bridge = joint_np
            model.explicit_bridge = explicit_bridge_np
            model.implicit_bridge = implicit_bridge_np
            implicit_t = torch.from_numpy(implicit_bridge_np).to(
                dtype=torch.float32, device=device_obj)
            joint_t = torch.from_numpy(joint_np).to(
                dtype=torch.float32, device=device_obj)

            if verbose:
                entropies = []
                for c1 in range(C):
                    for k1 in range(K):
                        row = joint_np[c1, k1].ravel()  # [C*K]
                        h = -np.sum(row * np.log(row + EPS))
                        entropies.append(h)
                print(f"    J entropy: {np.mean(entropies):.2f} / "
                      f"{np.log(C * K):.2f} nats", flush=True)

        def _mstep_weights_fix():
            for c in range(C):
                nk_w = g_Nk_t[c] - g_Nk_layer0_t[c]
                nk_w = nk_w + 1e-20
                model.gmm_bottom[c].weights_ = torch.clamp(
                    nk_w / nk_w.sum(), min=1e-20).float()

        # -- Joint EM iterations --
        if n_joint_iter > 0:
            if verbose:
                print(f"\n  [Phase 2 Joint EM] {n_joint_iter} iterations "
                      f"(GMM + bridge)")
            h = _estep_joint("joint init")
            history.append(h)
            prev_joint_ll = h["avg_loglik"]

            for jit in range(1, n_joint_iter + 1):
                _mstep()
                _mstep_weights_fix()
                _mstep_bridge()
                h = _estep_joint(f"joint iter {jit:02d}")
                history.append(h)
                avg_ll = h["avg_loglik"]
                rel_change = abs(avg_ll - prev_joint_ll) / (
                    abs(prev_joint_ll) + 1e-20)
                if rel_change < tol:
                    _mstep()
                    _mstep_weights_fix()
                    _mstep_bridge()
                    h_final = _estep_joint("joint final")
                    history.append(h_final)
                    if verbose:
                        print(f"  Joint EM convergence at iteration {jit} "
                              f"(rel_change={rel_change:.2e} < tol={tol:.1e})")
                    break
                prev_joint_ll = avg_ll

    return model, history


# ======================================================================
#  Decoding
# ======================================================================

def decode_prism(sequences: List[Dict],
                 model: PRISMModel,
                 label_key: str = "sentence_labels",
                 mode: str = "hard",
                 device: str = "cpu"):
    """Decode PRISM model.

    Args:
        mode: "hard" for argmax per layer,
              "soft" for posterior distributions per layer.
        device: torch device for computation.
    """
    if mode not in ("hard", "soft"):
        raise ValueError(f"mode must be 'hard' or 'soft', got '{mode}'")

    device_obj = torch.device(device)
    joint_tm = model.joint_bridge
    joint_t = (torch.from_numpy(joint_tm).to(dtype=torch.float32, device=device_obj)
               if joint_tm is not None else None)

    out = []
    for seq in sequences:
        steps = seq["steps"]
        labels_raw = seq.get(label_key, None)
        if labels_raw is None:
            raise RuntimeError(f"Missing '{label_key}'")

        y = coerce_labels_to_ids(labels_raw)
        if len(steps) != len(y):
            raise ValueError(
                f"decode_prism: len(steps)={len(steps)} != "
                f"len(labels)={len(y)}")
        T = len(y)
        cats = []
        regimes = []

        prev_exit_gamma = None   # [K] tensor on device
        prev_c = None

        for t in range(T):
            c = int(y[t])
            if c >= model.C or c == UNKNOWN_ID:
                prev_exit_gamma = None
                prev_c = None
                continue

            x_np = steps[t]
            x_t = torch.from_numpy(x_np).to(
                dtype=torch.float32, device=device_obj)  # [L, D]

            gmm = model.gmm_bottom[c]
            B = _emission_log_probs(gmm, x_t)           # [L, K]
            log_w = torch.log(gmm.weights_.to(device_obj) + 1e-20)  # [K]

            # Bridge: use exit posterior of previous step to modify layer 0 prior
            sp_override = None
            if joint_t is not None and prev_exit_gamma is not None and prev_c is not None:
                # joint[c1, k1, c2, k2] = P(c2, k2 | c1, k1)
                # entry_prior[k2] = sum_k1 gamma[k1] * J[c1, k1, c2, k2]
                if mode == "soft":
                    raw = prev_exit_gamma @ joint_t[prev_c, :, c, :]  # [K] @ [K,K] -> [K]
                else:
                    prev_exit_k = int(prev_exit_gamma.argmax())
                    raw = joint_t[prev_c, prev_exit_k, c, :]   # [K]
                raw_sum = raw.sum()
                if raw_sum.item() > EPS:
                    sp_override = raw / raw_sum

            # Compute posteriors via gmm_layer_posterior
            _, gamma = gmm_layer_posterior(B, log_w, startprob_override=sp_override)

            if mode == "hard":
                z = torch.argmax(gamma, dim=1)  # [L]
                regimes.append(z.cpu().tolist())
            else:
                regimes.append(gamma.cpu().tolist())

            prev_exit_gamma = gamma[-1].clone()
            prev_c = c
            cats.append(c)

        out.append({
            "best_categories": cats,
            "best_regimes_per_step": regimes,
        })
    return out


# ======================================================================
#  Analysis helpers
# ======================================================================

def get_top_transition_summary(model: PRISMModel) -> Dict:
    C = model.C
    order = model.top_order
    transmat = model.top.transmat

    summary = {
        "order": order,
        "num_categories": C,
        "context_size": C ** order,
        "most_likely_transitions": [],
    }

    for ctx_idx in range(C ** order):
        ctx_tuple = index_to_tuple(ctx_idx, C, order)
        ctx_str = " -> ".join([CANON_TAGS[c] if c < len(CANON_TAGS) else str(c)
                               for c in ctx_tuple])
        probs = transmat[ctx_idx]
        best_next = np.argmax(probs)
        best_next_str = (CANON_TAGS[best_next] if best_next < len(CANON_TAGS)
                         else str(best_next))
        summary["most_likely_transitions"].append({
            "context": ctx_str,
            "context_tuple": ctx_tuple,
            "next_category": best_next_str,
            "next_category_id": int(best_next),
            "probability": float(probs[best_next]),
            "all_probs": probs.tolist(),
        })

    return summary


def print_top_transition_matrix(model: PRISMModel, top_n: int = 20):
    C = model.C
    order = model.top_order
    transmat = model.top.transmat

    print(f"\n{'='*70}")
    print(f"Top-Level Transition Matrix (Order {order})")
    print(f"{'='*70}")

    # Index of "unknown" category to skip
    unknown_id = CANON_TAG2ID.get("unknown", -1)

    if order == 1:
        visible = [j for j in range(C) if j != unknown_id]
        print(f"\n  {'':30s} | "
              + " ".join(f"{CANON_TAGS[j][:8]:>8s}" for j in visible))
        print("-" * (35 + 9 * len(visible)))
        for i in visible:
            row = transmat[i]
            print(f"  {CANON_TAGS[i]:30s} | "
                  + " ".join(f"{row[j]:8.3f}" for j in visible))
    else:
        print(f"\nContext ({order} previous categories) -> Next Category")
        print("-" * 70)

        transitions = []
        for ctx_idx in range(C ** order):
            ctx_tuple = index_to_tuple(ctx_idx, C, order)
            if unknown_id in ctx_tuple:
                continue
            probs = transmat[ctx_idx]
            entropy = -np.sum(probs * np.log(np.maximum(probs, EPS)))
            best_next = np.argmax(probs)
            transitions.append({
                "ctx_idx": ctx_idx,
                "ctx_tuple": ctx_tuple,
                "probs": probs,
                "entropy": entropy,
                "max_prob": np.max(probs),
                "best_next": best_next,
            })
        transitions.sort(key=lambda x: x["entropy"])

        for i, t in enumerate(transitions[:top_n]):
            ctx_str = " -> ".join([CANON_TAGS[c][:15] if c < len(CANON_TAGS)
                                   else str(c) for c in t["ctx_tuple"]])
            next_str = (CANON_TAGS[t["best_next"]][:15]
                        if t["best_next"] < len(CANON_TAGS)
                        else str(t["best_next"]))
            print(f"  [{ctx_str}] => {next_str} "
                  f"(p={t['max_prob']:.3f}, H={t['entropy']:.3f})")


def count_top_transitions(
    sequences: List[Dict], C: int, top_order: int,
    label_key: str = "sentence_labels",
) -> TopParams:
    context_size = C ** top_order
    top_start_counts = np.zeros(context_size, dtype=np.float64)
    top_trans_counts = np.zeros((context_size, C), dtype=np.float64)

    for seq in sequences:
        labels_raw = seq.get(label_key, None)
        if labels_raw is None:
            continue
        y_raw = coerce_labels_to_ids(labels_raw)
        y = [c for c in y_raw if c != UNKNOWN_ID and c < C]
        Tsteps = len(y)
        if Tsteps >= top_order:
            init_ctx = tuple(y[:top_order])
            top_start_counts[tuple_to_index(init_ctx, C)] += 1.0
            for t in range(top_order, Tsteps):
                ctx = tuple(y[t - top_order:t])
                top_trans_counts[tuple_to_index(ctx, C), y[t]] += 1.0
        elif Tsteps > 0:
            padded = [y[0]] * (top_order - Tsteps) + y[:Tsteps]
            top_start_counts[tuple_to_index(tuple(padded[:top_order]), C)] += 1.0

    startprob = np.maximum(EPS, top_start_counts)
    startprob = (startprob / startprob.sum() if startprob.sum() > 0
                 else np.ones(context_size) / context_size)
    transmat = np.maximum(EPS, top_trans_counts)
    row_sums = transmat.sum(axis=1, keepdims=True)
    transmat = np.where(row_sums > 0, transmat / row_sums,
                        np.ones_like(transmat) / C)
    return TopParams(top_order, C, startprob, transmat)


def create_model_with_new_top_order(
    base_model: PRISMModel,
    sequences: List[Dict],
    new_top_order: int,
    label_key: str = "sentence_labels",
) -> PRISMModel:
    new_top = count_top_transitions(sequences, base_model.C, new_top_order,
                                    label_key)
    return PRISMModel(
        C=base_model.C,
        K=base_model.K,
        D=base_model.D,
        top_order=new_top_order,
        top=new_top,
        gmm_bottom=copy.deepcopy(base_model.gmm_bottom),
        implicit_bridge=copy.deepcopy(base_model.implicit_bridge),
        joint_bridge=copy.deepcopy(base_model.joint_bridge),
        explicit_bridge=copy.deepcopy(base_model.explicit_bridge),
    )

