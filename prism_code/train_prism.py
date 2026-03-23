#!/usr/bin/env python3
"""
PRISM Training
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

import argparse
import random
import json
import numpy as np
from typing import List, Dict

def _make_deterministic(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        pass

from prism_lib import (
    load_pt_records,
    build_top_sequences,
    fit_prism,
    coerce_labels_to_ids,
    gmm_layer_posterior,
    _precompute_emissions,
    CANON_TAGS,
    CANON_TAG2ID,
    UNKNOWN_ID,
    EPS,
    _extract_gmm_diag_vars,
    _to_np,
    print_top_transition_matrix,
    index_to_tuple,
    create_model_with_new_top_order,
)


class _GPUPCAResult:
    """PCA result fitted on GPU via torch.pca_lowrank, sklearn-compatible API."""

    def __init__(self, components_, mean_, singular_values_,
                 explained_variance_, explained_variance_ratio_):
        self.components_ = components_                            # (k, D) float32
        self.mean_ = mean_                                        # (D,)   float32
        self.singular_values_ = singular_values_                  # (k,)   float64
        self.explained_variance_ = explained_variance_            # (k,)   float64
        self.explained_variance_ratio_ = explained_variance_ratio_  # (k,) float64

    def transform(self, X):
        """Transform X: (N, D) numpy array -> (N, k) numpy array."""
        return (X.astype(np.float32) - self.mean_) @ self.components_.T


def compute_prism_metrics(seqs, model, label_key, device="cpu"):
    """Compute full PRISM metrics: LL, BIC, AIC, silhouette.
    """
    C = model.C
    K = model.K
    D = model.gmm_bottom[0].n_features if hasattr(model.gmm_bottom[0], 'n_features') else model.D
    joint_tm = model.joint_bridge

    # Precompute emissions on device
    pre_emissions, pre_log_w, _ = _precompute_emissions(
        seqs, model.gmm_bottom, C, label_key, device)

    # Joint transmat on device (includes P(c2|c1,k1) factor)
    joint_t = None
    if joint_tm is not None:
        joint_t = torch.from_numpy(joint_tm).to(dtype=torch.float32, device=device)

    total_ll = 0.0
    n_layers = 0
    n_steps = 0

    cat_features = {c: [] for c in range(C)}
    cat_argmax = {c: [] for c in range(C)}

    for si, seq in enumerate(seqs):
        if pre_emissions[si] is None:
            continue
        labels_raw = seq.get(label_key)
        y = coerce_labels_to_ids(labels_raw)
        steps = seq["steps"]
        if len(steps) != len(y):
            raise ValueError(
                f"Sequence {si}: len(steps)={len(steps)} != len(labels)={len(y)}")
        T = len(y)

        prev_exit_gamma = None
        prev_c = None

        for t in range(T):
            c = int(y[t])
            if c >= C or c == UNKNOWN_ID:
                prev_exit_gamma = None
                prev_c = None
                continue
            B = pre_emissions[si][t]
            log_w = pre_log_w[c]

            sp_override = None
            if prev_exit_gamma is not None and prev_c is not None:
                if joint_t is not None:
                    raw = prev_exit_gamma @ joint_t[prev_c, :, c, :]
                else:
                    raw = None
                if raw is not None:
                    raw_sum = raw.sum()
                    if raw_sum.item() > EPS:
                        sp_override = raw / raw_sum
                    else:
                        sp_override = torch.ones(K, dtype=torch.float32, device=device) / K

            logZ, gamma_r = gmm_layer_posterior(B, log_w, startprob_override=sp_override)
            total_ll += logZ.item()
            n_layers += B.shape[0]
            n_steps += 1

            # Hard assignment from posterior (argmax) for silhouette
            z_path = torch.argmax(gamma_r, dim=1).cpu().numpy()
            cat_features[c].append(steps[t])
            cat_argmax[c].append(z_path)

            prev_exit_gamma = gamma_r[-1].clone()
            prev_c = c

    # --- Parameter count ---
    # GMM per category: (K-1) weights + K*D means + K*D variances
    n_gmm_params = C * (K - 1 + 2 * K * D)
    # Joint bridge J[c1,k1,c2,k2]: C*K rows, each over C*K entries summing to 1
    n_bridge_params = C * K * (C * K - 1) if (joint_tm is not None) else 0
    n_params = n_gmm_params + n_bridge_params

    # BIC / AIC
    n_obs = n_layers  # each layer observation is one data point
    bic = n_params * np.log(n_obs) - 2 * total_ll if n_obs > 0 else float('inf')
    aic = 2 * n_params - 2 * total_ll if n_obs > 0 else float('inf')

    # --- Silhouette ---
    silhouette = -1.0
    try:
        from tgmm import ClusteringMetrics
        metrics = ClusteringMetrics()
        sil_scores = []
        MAX_SIL = 10_000
        for c in range(C):
            if not cat_features[c]:
                continue
            X = np.vstack(cat_features[c])  # [n_steps * L, D]
            labels = np.concatenate(cat_argmax[c])
            n_unique = len(set(labels.tolist()))
            if n_unique >= 2 and X.shape[0] >= max(K, 2):
                Xt = torch.from_numpy(X).float()
                lt = torch.from_numpy(labels).long() if not isinstance(labels, torch.Tensor) else labels
                if Xt.shape[0] > MAX_SIL:
                    idx = torch.randperm(Xt.shape[0])[:MAX_SIL]
                    sil_scores.append(float(metrics.silhouette_score(Xt[idx], lt[idx], K)))
                else:
                    sil_scores.append(float(metrics.silhouette_score(Xt, lt, K)))
        if sil_scores:
            silhouette = float(np.mean(sil_scores))
    except Exception:
        pass

    return {
        "ll": total_ll,
        "ll_per_layer": total_ll / n_layers if n_layers > 0 else float('-inf'),
        "ll_per_step": total_ll / n_steps if n_steps > 0 else float('-inf'),
        "bic": bic,
        "aic": aic,
        "silhouette": silhouette,
        "n_params": n_params,
        "n_layers": n_layers,
        "n_steps": n_steps,
    }


def _collect_cat_features(seqs, C, label_key):
    """Collect per-category layer vectors from sequences. Returns dict {c: np.ndarray}."""
    cat_features: Dict[int, List[np.ndarray]] = {c: [] for c in range(C)}
    for seq in seqs:
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
    cat_data = {}
    for c in range(C):
        if cat_features[c]:
            cat_data[c] = np.vstack(cat_features[c])
        else:
            cat_data[c] = np.empty((0, 0))
    return cat_data


def compute_gmm_metrics(cat_data, gmm_list, C):
    """Compute per-category metrics (BIC, AIC, silhouette, score).
    """
    import torch
    from tgmm import ClusteringMetrics
    metrics = ClusteringMetrics()

    total_bic = 0.0
    total_aic = 0.0
    total_ll = 0.0
    total_n = 0
    sil_scores = []

    for c in range(C):
        X = cat_data[c]
        if X.shape[0] == 0:
            continue
        gmm = gmm_list[c]
        Xt = torch.from_numpy(X).float()

        ll = float(gmm.score(Xt))
        total_bic += float(metrics.bic_score(ll, Xt, gmm.n_components, gmm.covariance_type))
        total_aic += float(metrics.aic_score(ll, Xt, gmm.n_components, gmm.covariance_type))
        total_ll += ll * X.shape[0]  # score() = mean LL
        total_n += X.shape[0]

        labels = gmm.predict(Xt)
        n_unique = len(set(labels.tolist() if hasattr(labels, 'tolist') else list(labels)))
        if n_unique >= 2 and X.shape[0] >= max(gmm.n_components, 2):
            try:
                # Subsample to avoid O(N²) pairwise distance OOM
                MAX_SIL = 10_000
                if Xt.shape[0] > MAX_SIL:
                    idx = torch.randperm(Xt.shape[0])[:MAX_SIL]
                    sil_scores.append(float(metrics.silhouette_score(
                        Xt[idx], labels[idx], gmm.n_components)))
                else:
                    sil_scores.append(float(metrics.silhouette_score(
                        Xt, labels, gmm.n_components)))
            except Exception:
                pass

    ll_norm = total_ll / total_n if total_n > 0 else float('-inf')
    sil = float(np.mean(sil_scores)) if sil_scores else -1.0

    return {
        "bic": total_bic,
        "aic": total_aic,
        "ll": total_ll,
        "ll_norm": ll_norm,
        "n_obs": total_n,
        "silhouette": sil,
    }


def gmm_sweep(seqs, C, k_values, label_key, seed, device="cpu",
              k_criterion="silhouette", gmm_init="random"):
    """Sweep over K values.
    """
    from prism_lib import train_gmm_bottoms

    results = {
        "k_values": [], "bic": [], "aic": [], "ll": [], "ll_norm": [],
        "sil_score": [],
    }

    use_sil = k_criterion == "silhouette"

    print(f"\n{'='*60}")
    print(f"K Sweep (GMM only): K ∈ {k_values}")
    print(f"Selection criterion: {'silhouette' if use_sil else 'BIC'} (tgmm)")
    print(f"{'='*60}")

    # Collect per-category data once
    cat_data = _collect_cat_features(seqs, C, label_key)

    for i, K in enumerate(k_values):
        print(f"\n[K={K}] Fitting GMMs...", end=" ", flush=True)
        _make_deterministic(seed)

        gmm_list = train_gmm_bottoms(seqs, C, K, label_key=label_key,
                                     seed=seed, verbose=False, device=device,
                                     init_means=gmm_init, n_init=1,
                                     max_iter=50)
        m = compute_gmm_metrics(cat_data, gmm_list, C)

        results["k_values"].append(K)
        results["bic"].append(m["bic"])
        results["aic"].append(m["aic"])
        results["ll"].append(m["ll"])
        results["ll_norm"].append(m["ll_norm"])
        results["sil_score"].append(m["silhouette"])

        extra = f", silhouette={m['silhouette']:.4f}" if use_sil else ""
        if i > 0:
            prev_bic = results["bic"][i-1]
            delta_bic = (prev_bic - m["bic"]) / abs(prev_bic) if prev_bic != 0 else 0
            print(f"LL/layer={m['ll_norm']:.3f}, BIC={m['bic']:.2f}, ΔBIC={delta_bic:.4f}{extra}")
        else:
            print(f"LL/layer={m['ll_norm']:.3f}, BIC={m['bic']:.2f}{extra}")

    # Select best K
    if use_sil:
        best_idx = int(np.nanargmax(results["sil_score"]))
        criterion_str = f"silhouette = {results['sil_score'][best_idx]:.4f}"
    else:
        best_idx = int(np.argmin(results["bic"]))
        criterion_str = f"BIC = {results['bic'][best_idx]:.2f}"

    best_k = results["k_values"][best_idx]

    # Summary table
    width = 76
    print(f"\n{'='*width}")
    print(f"{'K':>5} {'LL/layer':>12} {'Silhouette':>12} {'BIC':>14} {'AIC':>14} {'ΔBIC':>10}")
    print(f"{'-'*width}")
    for i, K in enumerate(results["k_values"]):
        marker = "  <-- BEST" if i == best_idx else ""
        delta_str = (f"{(results['bic'][i-1]-results['bic'][i])/abs(results['bic'][i-1]):>10.4f}"
                     if i > 0 else f"{'---':>10}")
        sil_str = f"{results['sil_score'][i]:>12.4f}"
        print(f"{K:>5} {results['ll_norm'][i]:>12.3f} {sil_str} {results['bic'][i]:>14.2f} "
              f"{results['aic'][i]:>14.2f} {delta_str}{marker}")

    print(f"\n[SELECTED] Best K = {best_k} ({criterion_str})")

    return best_k, results


def main():
    ap = argparse.ArgumentParser(
        description="Train PRISM model (GMM bottom, higher-order top level, optional auto-K)"
    )
    ap.add_argument("--in_pt", required=True, nargs="+")
    ap.add_argument("--C", type=int, default=None)

    k_group = ap.add_mutually_exclusive_group()
    k_group.add_argument("--K", type=int, default=None)
    k_group.add_argument("--auto_k", action="store_true")

    ap.add_argument("--k_min", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=15)
    ap.add_argument("--k_step", type=int, default=2)
    ap.add_argument("--k_values", type=int, nargs="+", default=None)
    ap.add_argument("--no_skip_embedding", action="store_true",
                    help="Don't skip embedding layer (layer 0) from hidden states")

    ap.add_argument("--top_orders", type=int, nargs="+", default=[1])
    ap.add_argument("--top_order", type=int, default=None,
                    help="(Deprecated) Use --top_orders instead")

    ap.add_argument("--iters", type=int, default=10,
                    help="Max EM iterations for K sweep")
    ap.add_argument("--warmup_iter", type=int, default=25,
                    help="Max GMM-only EM iterations (Phase 1 warmup when using --joint_iter)")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="Convergence tolerance: rel_change = |ll - prev_ll| / (|prev_ll| + 1e-20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_npz", default="prism_model.npz")
    ap.add_argument("--label_key", default="sentence_labels")
    ap.add_argument("--pca_dim", type=int, default=64)
    ap.add_argument("--per_step_rms", action="store_true")
    ap.add_argument("--save_sweep_results", action="store_true")
    ap.add_argument("--print_transitions", action="store_true")
    ap.add_argument("--k_criterion", choices=["silhouette", "bic"],
                    default="silhouette")
    ap.add_argument("--gmm_init", type=str, default="random",
                    help="GMM initialization method for K sweep (tgmm init_means)")
    ap.add_argument("--bridge", action="store_true",
                    help="Learn cross-step bridge matrices")
    ap.add_argument("--joint_iter", type=int, default=0,
                    help="Number of joint GMM+bridge EM iterations after Phase 1 (implies --bridge)")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device for tgmm GMM fitting: 'cpu', 'cuda', or 'mps'")
    args = ap.parse_args()

    if args.joint_iter > 0:
        args.bridge = True

    if not args.auto_k and args.K is None:
        args.K = 7

    if args.C is None:
        args.C = len(CANON_TAGS) - 1   # 4 known categories (exclude unknown)
        print(f"[INFO] C={args.C} (4 known categories, unknown steps skipped)")

    if args.top_order is not None:
        print(f"[WARN] --top_order is deprecated, use --top_orders instead")
        args.top_orders = [args.top_order]

    for order in args.top_orders:
        if order < 1:
            raise ValueError(f"--top_orders must be >= 1, got {order}")
    max_order = max(args.top_orders)
    if max_order > 3:
        print(f"[WARNING] top_order={max_order} will create {args.C ** max_order} context states.")

    print(f"[INFO] Will train top_orders: {args.top_orders}")
    print(f"[INFO] Bottom type: GMM (tgmm, device={args.device})")

    _make_deterministic(args.seed)

    # Load data
    recs = []
    for pt_path in args.in_pt:
        r = load_pt_records(pt_path)
        recs.extend(r)
        print(f"  {pt_path}: {len(r)} records")
    print(f"Loaded {len(recs)} records total from {len(args.in_pt)} file(s)")

    skip_embedding = not args.no_skip_embedding
    seqs = build_top_sequences(recs, skip_embedding_layer=skip_embedding)
    del recs  # free raw PT records — no longer needed
    if skip_embedding:
        print(f"[INFO] Skipping embedding layer (layer 0) from hidden states")
    before = len(seqs)
    seqs = [s for s in seqs if args.label_key in s]
    after = len(seqs)
    if after == 0:
        raise RuntimeError(f"No sequences with '{args.label_key}'")
    print(f"Kept {after}/{before} sequences with labels")

    # Analyze labels
    print(f"\n[INFO] Label distribution:")
    label_counts = {i: 0 for i in range(len(CANON_TAGS))}
    total_steps = 0
    for i, s in enumerate(seqs):
        labels_raw = s.get(args.label_key)
        if labels_raw is None:
            raise ValueError(f"Sequence {i} has '{args.label_key}' key but value is None.")
        labels = coerce_labels_to_ids(labels_raw)
        for lbl in labels:
            if lbl < len(CANON_TAGS):
                label_counts[lbl] += 1
            total_steps += 1

    n_unknown = label_counts.get(UNKNOWN_ID, 0)
    n_known = total_steps - n_unknown
    for i in range(args.C):
        tag = CANON_TAGS[i] if i < len(CANON_TAGS) else f"cat_{i}"
        pct = 100.0 * label_counts[i] / max(total_steps, 1)
        print(f"  {i} ({tag:30s}): {label_counts[i]:6d} ({pct:5.2f}%)")
    print(f"  * ({CANON_TAGS[UNKNOWN_ID]:30s}): {n_unknown:6d} ({100.0*n_unknown/max(total_steps,1):5.2f}%) [SKIPPED]")
    print(f"  Known steps used for training: {n_known}/{total_steps}")

    # Feature preprocessing
    import time as _time
    _t0 = _time.time()

    num_layers = seqs[0]["steps"][0].shape[0]
    D_in = seqs[0]["steps"][0].shape[1]

    print(f"\n[INFO] Preprocessing: per-layer mean + RMS + global PCA")
    print(f"  num_layers={num_layers}, D_in={D_in}")

    # 1. Stack all steps into one array: (total_steps, num_layers, D_in)
    all_steps = [step for seq in seqs for step in seq["steps"]]
    n_steps_per_seq = [len(seq["steps"]) for seq in seqs]
    all_steps_arr = np.stack(all_steps, axis=0)  # float32
    n_total_steps = all_steps_arr.shape[0]
    del all_steps

    for seq in seqs:
        seq["steps"] = []
    print(f"  total steps: {n_total_steps}, "
          f"mem: {all_steps_arr.nbytes / 1e9:.2f} GB")

    PREP_EPS = 1e-8
    k_pca = min(args.pca_dim, D_in)

    print(f"  [GPU] Moving data to {args.device} ...", end=" ", flush=True)
    Xt = torch.from_numpy(all_steps_arr).to(args.device)  # (N, L, D) float32
    del all_steps_arr
    print(f"done ({Xt.element_size() * Xt.nelement() / 1e9:.2f} GB on GPU)")

    # 2. Per-layer mean and RMS
    layer_mean_t = Xt.mean(dim=0)                        # (L, D)
    Xt -= layer_mean_t.unsqueeze(0)                       # center in-place
    num_layers = Xt.shape[1]
    layer_rms_t = torch.empty(num_layers, dtype=Xt.dtype, device=Xt.device)
    for li in range(num_layers):
        layer_rms_t[li] = torch.sqrt(Xt[:, li, :].pow(2).mean())
    # layer_rms_t shape: (L,)
    layer_rms_t = torch.where(
        torch.isfinite(layer_rms_t) & (layer_rms_t >= PREP_EPS),
        layer_rms_t, torch.ones_like(layer_rms_t)
    )
    layer_mean = layer_mean_t.cpu().numpy().astype(np.float64)
    layer_rms = layer_rms_t.cpu().numpy().astype(np.float64)
    print(f"  Layer RMS range: [{layer_rms.min():.4f}, {layer_rms.max():.4f}]")

    # 3. Normalize
    Xt /= (layer_rms_t.unsqueeze(0).unsqueeze(2) + PREP_EPS)
    if args.per_step_rms:
        step_sq_sum = torch.zeros(Xt.shape[0], dtype=Xt.dtype, device=Xt.device)
        for li in range(num_layers):
            step_sq_sum += Xt[:, li, :].pow(2).sum(dim=1)
        step_rms_t = torch.sqrt(step_sq_sum / (num_layers * D_in))  # (N,)
        del step_sq_sum
        Xt /= (step_rms_t.unsqueeze(1).unsqueeze(2) + PREP_EPS)
        del step_rms_t
    del layer_mean_t, layer_rms_t

    # 4. Reshape for PCA: (N*L, D)
    Xf = Xt.reshape(-1, D_in)
    del Xt
    N_pca = Xf.shape[0]

    # 5. PCA fit
    print(f"  Fitting PCA ({D_in} -> {k_pca}) on {N_pca} vectors ...",
          end=" ", flush=True)
    pca_mean = Xf.mean(dim=0)
    Xf -= pca_mean                                        # center in-place
    torch.manual_seed(args.seed)
    _U, S, V = torch.pca_lowrank(Xf, q=k_pca, niter=2, center=False)

    n_fit = Xf.shape[0]
    exp_var = (S ** 2) / (n_fit - 1)
    total_var = Xf.var(dim=0).sum()
    exp_var_ratio = exp_var / total_var

    components_np = V.T.cpu().numpy().astype(np.float32)  # (k, D)
    mean_np = pca_mean.cpu().numpy().astype(np.float32)   # (D,)
    pca = _GPUPCAResult(
        components_=components_np,
        mean_=mean_np,
        singular_values_=S.cpu().numpy().astype(np.float64),
        explained_variance_=exp_var.cpu().numpy().astype(np.float64),
        explained_variance_ratio_=exp_var_ratio.cpu().numpy().astype(np.float64),
    )
    del _U, S, exp_var, exp_var_ratio
    print(f"done")

    # 6. PCA transform
    X_transformed = (Xf @ V).cpu().numpy()
    del Xf, V, pca_mean
    if args.device == "cuda":
        torch.cuda.empty_cache()

    total_var_explained = pca.explained_variance_ratio_.sum()
    print(f"    explained_var_ratio (first 5): {pca.explained_variance_ratio_[:5]}")
    print(f"    total explained variance: {total_var_explained:.4f} ({total_var_explained*100:.2f}%)")

    # Reshape back: (N, L, D_pca), then assign to sequences
    X_transformed = X_transformed.reshape(n_total_steps, num_layers, k_pca)
    idx = 0
    for seq, n in zip(seqs, n_steps_per_seq):
        seq["steps"] = [X_transformed[idx + i].copy() for i in range(n)]
        idx += n
    del X_transformed, n_steps_per_seq

    print(f"  Preprocessing done in {_time.time() - _t0:.1f}s")

    # ============================================================
    # Train
    # ============================================================
    base_order = args.top_orders[0]
    sweep_results = None

    print(f"\n[INFO] Training GMM bottoms with top_order={base_order}")
    print(f"       Context states: {args.C ** base_order}")

    if args.bridge:
        print(f"\n[INFO] Bridge matrices ENABLED")

    train_history = []

    if args.auto_k:
        k_values = args.k_values or list(range(args.k_min, args.k_max + 1, args.k_step))

        best_k, sweep_results = gmm_sweep(
            seqs, args.C, k_values, args.label_key, args.seed,
            device=args.device, k_criterion=args.k_criterion,
            gmm_init=args.gmm_init,
        )
        final_K = best_k

        # Train full PRISM with best K (GMM + bridge)
        train_iters = args.warmup_iter if args.warmup_iter > 0 else args.iters
        print(f"\n[INFO] Training full PRISM with K={best_k}, iters={train_iters}")
        _make_deterministic(args.seed)
        base_model, train_history = fit_prism(
            seqs, args.C, best_k, base_order, args.label_key,
            train_iters, seed=args.seed, verbose=True,
            use_bridge=args.bridge, device=args.device,
            tol=args.tol, n_joint_iter=args.joint_iter,
        )

        # Report metrics
        pm = compute_prism_metrics(seqs, base_model, args.label_key, device=args.device)
        print(f"[INFO] PRISM metrics: LL/layer={pm['ll_per_layer']:.3f}, "
              f"LL/step={pm['ll_per_step']:.3f}, BIC={pm['bic']:.2f}, "
              f"AIC={pm['aic']:.2f}, silhouette={pm['silhouette']:.4f}")
    else:
        manual_iters = args.warmup_iter if args.warmup_iter > 0 else args.iters
        print(f"\n[INFO] Training with K={args.K} (manual), iters={manual_iters}")
        base_model, train_history = fit_prism(
            seqs, args.C, args.K, base_order, args.label_key,
            manual_iters, seed=args.seed, verbose=True,
            use_bridge=args.bridge, device=args.device,
            tol=args.tol, n_joint_iter=args.joint_iter,
        )

        # Report metrics
        pm = compute_prism_metrics(seqs, base_model, args.label_key, device=args.device)
        print(f"\n[INFO] PRISM metrics: LL/layer={pm['ll_per_layer']:.3f}, "
              f"LL/step={pm['ll_per_step']:.3f}, BIC={pm['bic']:.2f}, "
              f"AIC={pm['aic']:.2f}, silhouette={pm['silhouette']:.4f}")
        final_K = args.K

    # Build models for all orders
    models = {base_order: base_model}
    for order in args.top_orders:
        if order == base_order:
            continue
        print(f"\n[INFO] Creating model for top_order={order} (reusing GMM bottoms)")
        models[order] = create_model_with_new_top_order(base_model, seqs, order, args.label_key)

    def get_output_path(order):
        if len(args.top_orders) == 1:
            return args.out_npz
        elif "{order}" in args.out_npz:
            return args.out_npz.replace("{order}", str(order))
        else:
            base, ext = args.out_npz.rsplit(".", 1) if "." in args.out_npz else (args.out_npz, "npz")
            return f"{base}_order{order}.{ext}"

    # Save each model
    for order in args.top_orders:
        model = models[order]
        out_path = get_output_path(order)

        if args.print_transitions:
            print(f"\n[ORDER={order}] Transition analysis:")
            print_top_transition_matrix(model, top_n=25)

        context_size = args.C ** order
        out = {
            "C": np.array([model.C], dtype=np.int32),
            "K": np.array([model.K], dtype=np.int32),
            "D": np.array([model.D], dtype=np.int32),
            "top_order": np.array([model.top_order], dtype=np.int32),
            "top_start": model.top.startprob,
            "top_trans": model.top.transmat,
            # GMM bottom params: weights, means, diagonal variances
            **{f"b{c}_weights": _to_np(model.gmm_bottom[c].weights_).astype(np.float64) for c in range(model.C)},
            **{f"b{c}_means": _to_np(model.gmm_bottom[c].means_).astype(np.float64) for c in range(model.C)},
            **{f"b{c}_vars": _extract_gmm_diag_vars(model.gmm_bottom[c]).astype(np.float64) for c in range(model.C)},
            # preprocessing
            "num_layers": np.array([num_layers], dtype=np.int32),
            "D_in": np.array([D_in], dtype=np.int32),
            "prep_pca_k": np.array([k_pca], dtype=np.int32),
            **{f"prep_L{L}_mean": layer_mean[L].astype(np.float64) for L in range(num_layers)},
            **{f"prep_L{L}_rms": np.array([layer_rms[L]], dtype=np.float64) for L in range(num_layers)},
            "prep_global_pca_components": pca.components_.astype(np.float64),
            "prep_global_pca_mean": pca.mean_.astype(np.float64),
            "prep_global_pca_explained_variance": pca.explained_variance_.astype(np.float64),
            "prep_global_pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float64),
            "prep_global_pca_singular_values": pca.singular_values_.astype(np.float64),
            # metadata
            "meta_preproc": np.array("per_layer_mean__per_layer_rms__global_pca", dtype=object),
            "meta_per_step_rms": np.array([args.per_step_rms], dtype=bool),
            "meta_canon_tags": np.array(CANON_TAGS, dtype=object),
            "meta_auto_k": np.array([args.auto_k], dtype=bool),
            "meta_selected_k": np.array([final_K], dtype=np.int32),
            "meta_bottom_type": np.array("gmm", dtype=object),
        }

        # Joint bridge matrix and derived quantities
        if model.joint_bridge is not None:
            out["joint_bridge"] = model.joint_bridge.astype(np.float64)
            out["meta_use_bridge"] = np.array([True], dtype=bool)
        if model.implicit_bridge is not None:
            out["implicit_bridge"] = model.implicit_bridge.astype(np.float64)
        if model.explicit_bridge is not None:
            out["explicit_bridge"] = model.explicit_bridge.astype(np.float64)

        # Sweep results
        if order == base_order and sweep_results is not None:
            out["sweep_k_values"] = np.array(sweep_results["k_values"], dtype=np.int32)
            out["sweep_bic"] = np.array(sweep_results["bic"], dtype=np.float64)
            out["sweep_aic"] = np.array(sweep_results["aic"], dtype=np.float64)
            out["sweep_ll"] = np.array(sweep_results["ll"], dtype=np.float64)

        np.savez(out_path, **out)

        if order == base_order and train_history:
            history_path = out_path.replace(".npz", "_history.json")
            with open(history_path, "w") as f:
                json.dump(train_history, f, indent=2)
            print(f"[INFO] Saved training history to {history_path}")

        print(f"\n{'='*60}")
        print(f"[SAVED] {out_path}")
        print(f"  C = {model.C} categories")
        print(f"  K = {model.K} GMM components {'(auto)' if args.auto_k else '(manual)'}")
        print(f"  D = {model.D} dimensions")
        print(f"  top_order = {model.top_order} ({context_size} context states)")
        print(f"  bottom = GMM (device={args.device})")
        print(f"  bridge = {'YES' if model.joint_bridge is not None else 'NO'}")
        print(f"{'='*60}")

        # Summary
        unknown_id = CANON_TAG2ID.get("unknown", -1)
        print(f"\n[RESULTS] Top-level start probabilities:")
        top_starts = sorted(range(context_size), key=lambda i: -model.top.startprob[i])
        shown = 0
        for idx in top_starts:
            if shown >= 5:
                break
            ctx = index_to_tuple(idx, model.C, model.top_order)
            if unknown_id in ctx:
                continue
            prob = model.top.startprob[idx]
            ctx_str = " -> ".join([CANON_TAGS[c][:20] if c < len(CANON_TAGS) else str(c) for c in ctx])
            print(f"  [{ctx_str}]: {prob:.4f}")
            shown += 1

    print(f"\n[DONE] Trained {len(args.top_orders)} PRISM models")


if __name__ == "__main__":
    main()
