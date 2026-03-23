#!/usr/bin/env python3
"""Aggregate bottom-level analysis.

Usage:
    python aggregate_prism_bottom.py \
        --bottom_dir /scratch/$USER/reasoning_newstart/analysis_bottom_joint \
        --output_dir aggregate_bottom_results
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregate_lib import (
    TAG_SHORT as CAT_TAGS, CAT_PAIRS, CAT_COLORS,
    CORR_COLOR, INCORR_COLOR, SAVE_KW,
    setup_style, discover_jsons as _discover, load_json,
    aggregate, _bar_annotate,
)


def discover_jsons(base_dir, models=None, datasets=None):
    return _discover(base_dir, "bottom_analysis.json", models=models, datasets=datasets)


# ── Metric extraction ─────────────────────────────────────────────────

def extract_metrics(d: Dict) -> Dict[str, float]:
    m = {}

    # 1. Transition direction consistency + magnitude
    td = d.get("transition_directions", {})
    tvecs = td.get("transition_vectors", {})
    for key, info in tvecs.items():
        if info.get("n_transitions", 0) >= 5:
            m[f"td_cons_{key}"] = info.get("consistency")
            m[f"td_mag_{key}"] = info.get("magnitude")

    # Cross-transition cosine similarity
    for pair_key, cos_val in td.get("cross_transition_cosine", {}).items():
        # pair_key looks like "FA->SR vs AC->UV"
        m[f"td_cross_cos_{pair_key}"] = cos_val

    # 5. Avg inter-regime distance + spread per category
    rchars = d.get("regime_characteristics", {})
    for i, cat in enumerate(CAT_TAGS):
        rc_cat = rchars.get(str(i), {})
        m[f"inter_regime_dist_{cat}"] = rc_cat.get("avg_inter_regime_distance")
        m[f"regime_spread_{cat}"] = rc_cat.get("regime_spread")
        m[f"stickiness_{cat}"] = rc_cat.get("stickiness")

    model_info = d.get("model_info", {})
    has_bridge = model_info.get("has_bridge", False)
    m["has_bridge"] = 1.0 if has_bridge else 0.0

    _normalize_pca_dependent(m)

    return m


def _normalize_pca_dependent(m: Dict):
    # 1. Distance-based metrics: normalize by global mean inter-regime distance
    raw_dists = [m[f"inter_regime_dist_{c}"] for c in CAT_TAGS
                 if m.get(f"inter_regime_dist_{c}") is not None]
    if raw_dists:
        dist_scale = float(np.mean(raw_dists))
        if dist_scale > 1e-8:
            for cat in CAT_TAGS:
                for prefix in ("inter_regime_dist_", "regime_spread_"):
                    key = f"{prefix}{cat}"
                    if m.get(key) is not None:
                        m[key] = m[key] / dist_scale
    # 2. Direction magnitude: normalize by mean across all transitions
    mag_keys = [k for k in m if k.startswith("td_mag_") and m[k] is not None]
    if mag_keys:
        mag_scale = float(np.mean([m[k] for k in mag_keys]))
        if mag_scale > 1e-8:
            for k in mag_keys:
                m[k] = m[k] / mag_scale



# ── Plotting ──────────────────────────────────────────────────────────


def plot_cosine_heatmap(agg: Dict, out: str):
    """4×4 cosine similarity heatmap from averaged values."""
    n = 4
    mat = np.eye(n)
    for pair in CAT_PAIRS:
        val = agg.get(f"cos_{pair}", {}).get("mean", 0)
        parts = pair.split("-")
        i = CAT_TAGS.index(parts[0])
        j = CAT_TAGS.index(parts[1])
        mat[i, j] = val
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            std_key = f"cos_{CAT_TAGS[min(i,j)]}-{CAT_TAGS[max(i,j)]}"
            std_val = agg.get(std_key, {}).get("std", 0) if i != j else 0
            txt = f"{mat[i,j]:.3f}"
            if std_val > 0:
                txt += f"\n±{std_val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="white" if abs(mat[i, j]) > 0.6 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CAT_TAGS)
    ax.set_yticklabels(CAT_TAGS)
    ax.set_title("Category Cosine Similarity (avg ± std)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(out, "category_cosine_avg.png"), **SAVE_KW)
    plt.close(fig)


def plot_boundary_js(agg: Dict, out: str):
    """Bar chart: boundary JS divergence per category."""
    vals = [agg.get(f"boundary_js_{c}", {}).get("mean", 0) for c in CAT_TAGS]
    errs = [agg.get(f"boundary_js_{c}", {}).get("std", 0) for c in CAT_TAGS]
    colors = [CAT_COLORS[c] for c in CAT_TAGS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(CAT_TAGS, vals, yerr=errs, capsize=5,
                   color=colors, edgecolor="black", linewidth=0.5)
    _bar_annotate(ax1, bars, ".4f")
    ax1.set_ylabel("JS Divergence")
    ax1.set_title("Boundary vs Internal (JS)")

    # LL diff
    ll_vals = [agg.get(f"boundary_ll_diff_{c}", {}).get("mean", 0) for c in CAT_TAGS]
    ll_errs = [agg.get(f"boundary_ll_diff_{c}", {}).get("std", 0) for c in CAT_TAGS]
    bars = ax2.bar(CAT_TAGS, ll_vals, yerr=ll_errs, capsize=5,
                   color=colors, edgecolor="black", linewidth=0.5)
    _bar_annotate(ax2, bars, ".4f")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("LL Difference (boundary − internal)")
    ax2.set_title("Boundary LL Effect")

    fig.suptitle("Boundary Analysis (averaged)", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "boundary_avg.png"), **SAVE_KW)
    plt.close(fig)


def plot_transition_directions(agg: Dict, out: str):
    """Consistency + magnitude for each transition direction."""
    # Collect all td_cons_* keys
    td_keys = sorted(set(
        k.replace("td_cons_", "") for k in agg if k.startswith("td_cons_")
    ))
    if not td_keys:
        print("  [SKIP] no transition direction data")
        return

    cons_vals = [agg.get(f"td_cons_{k}", {}).get("mean", 0) for k in td_keys]
    cons_errs = [agg.get(f"td_cons_{k}", {}).get("std", 0) for k in td_keys]
    mag_vals = [agg.get(f"td_mag_{k}", {}).get("mean", 0) for k in td_keys]
    mag_errs = [agg.get(f"td_mag_{k}", {}).get("std", 0) for k in td_keys]
    colors = [CAT_COLORS.get(k.split("->")[0], "#999") for k in td_keys]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    bars = ax1.bar(range(len(td_keys)), cons_vals, yerr=cons_errs, capsize=3,
                   color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Consistency (mean cosine)")
    ax1.set_ylim(0, 1)
    ax1.set_title("Transition Direction Consistency")

    bars = ax2.bar(range(len(td_keys)), mag_vals, yerr=mag_errs, capsize=3,
                   color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Magnitude (L2)")
    ax2.set_title("Transition Direction Magnitude")
    ax2.set_xticks(range(len(td_keys)))
    ax2.set_xticklabels(td_keys, rotation=45, ha="right", fontsize=10)

    fig.suptitle("Transition Directions (averaged)", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "transition_dirs_avg.png"), **SAVE_KW)
    plt.close(fig)


def plot_inter_regime_distance(agg: Dict, out: str):
    """3-panel: inter-regime distance, spread, stickiness per category."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = [CAT_COLORS[c] for c in CAT_TAGS]

    # (a) inter-regime distance
    vals = [agg.get(f"inter_regime_dist_{c}", {}).get("mean", 0) for c in CAT_TAGS]
    errs = [agg.get(f"inter_regime_dist_{c}", {}).get("std", 0) for c in CAT_TAGS]
    bars = axes[0].bar(CAT_TAGS, vals, yerr=errs, capsize=5,
                       color=colors, edgecolor="black", linewidth=0.5)
    _bar_annotate(axes[0], bars)
    axes[0].set_title("Avg Inter-Regime Distance")

    # (b) spread
    vals = [agg.get(f"regime_spread_{c}", {}).get("mean", 0) for c in CAT_TAGS]
    errs = [agg.get(f"regime_spread_{c}", {}).get("std", 0) for c in CAT_TAGS]
    bars = axes[1].bar(CAT_TAGS, vals, yerr=errs, capsize=5,
                       color=colors, edgecolor="black", linewidth=0.5)
    _bar_annotate(axes[1], bars)
    axes[1].set_title("Regime Spread (from centroid)")

    # (c) stickiness
    vals = [agg.get(f"stickiness_{c}", {}).get("mean", 0) for c in CAT_TAGS]
    errs = [agg.get(f"stickiness_{c}", {}).get("std", 0) for c in CAT_TAGS]
    bars = axes[2].bar(CAT_TAGS, vals, yerr=errs, capsize=5,
                       color=colors, edgecolor="black", linewidth=0.5)
    _bar_annotate(axes[2], bars)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Bottom-Level Stickiness")

    fig.suptitle("Regime Geometry (averaged)", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "regime_geometry_avg.png"), **SAVE_KW)
    plt.close(fig)


def plot_min_regime_distance(agg: Dict, out: str):
    """Bar chart: min regime distance between category pairs."""
    vals = [agg.get(f"min_regime_dist_{p}", {}).get("mean", 0) for p in CAT_PAIRS]
    errs = [agg.get(f"min_regime_dist_{p}", {}).get("std", 0) for p in CAT_PAIRS]
    if all(v == 0 for v in vals):
        return

    pair_colors = plt.cm.Set2(np.linspace(0, 1, len(CAT_PAIRS)))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CAT_PAIRS, vals, yerr=errs, capsize=5,
                  color=pair_colors, edgecolor="black", linewidth=0.5)
    _bar_annotate(ax, bars)
    ax.set_ylabel("Min Distance")
    ax.set_title("Min Regime Distance Between Category Pairs (averaged)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "min_regime_dist_avg.png"), **SAVE_KW)
    plt.close(fig)



# ── Report ────────────────────────────────────────────────────────────

def write_report(agg: Dict, n_runs: int, out: str):
    """Write concise markdown summary."""
    L = []
    L.append("# Bottom-Level Aggregate Report\n")
    L.append(f"**Averaged across {n_runs} runs**\n")

    # 1. Category cosine
    L.append("## 2. Category Cosine Similarity\n")
    L.append("| Pair | Mean | Std |")
    L.append("|------|------|-----|")
    for pair in CAT_PAIRS:
        d = agg.get(f"cos_{pair}", {})
        L.append(f"| {pair} | {d.get('mean', 0):.4f} | {d.get('std', 0):.4f} |")
    L.append("")

    # 3. Boundary JS
    L.append("## 3. Boundary JS Divergence\n")
    L.append("| Category | JS Mean | JS Std | LL Diff Mean | LL Diff Std |")
    L.append("|----------|---------|--------|--------------|-------------|")
    for cat in CAT_TAGS:
        js = agg.get(f"boundary_js_{cat}", {})
        ll = agg.get(f"boundary_ll_diff_{cat}", {})
        L.append(f"| {cat} | {js.get('mean', 0):.4f} | {js.get('std', 0):.4f} "
                 f"| {ll.get('mean', 0):.4f} | {ll.get('std', 0):.4f} |")
    L.append("")

    # 4. Transition directions
    td_keys = sorted(set(k.replace("td_cons_", "") for k in agg if k.startswith("td_cons_")))
    if td_keys:
        L.append("## 4. Transition Directions\n")
        L.append("| Direction | Consistency | Magnitude |")
        L.append("|-----------|-------------|-----------|")
        for k in td_keys:
            cons = agg.get(f"td_cons_{k}", {})
            mag = agg.get(f"td_mag_{k}", {})
            L.append(f"| {k} | {cons.get('mean', 0):.3f}±{cons.get('std', 0):.3f} "
                     f"| {mag.get('mean', 0):.3f}±{mag.get('std', 0):.3f} |")
        L.append("")

    # 5. Inter-regime distance
    L.append("## 5. Regime Geometry\n")
    L.append("| Category | Inter-Regime Dist | Spread | Stickiness |")
    L.append("|----------|-------------------|--------|------------|")
    for cat in CAT_TAGS:
        ird = agg.get(f"inter_regime_dist_{cat}", {})
        sp = agg.get(f"regime_spread_{cat}", {})
        st = agg.get(f"stickiness_{cat}", {})
        L.append(f"| {cat} | {ird.get('mean', 0):.3f}±{ird.get('std', 0):.3f} "
                 f"| {sp.get('mean', 0):.3f}±{sp.get('std', 0):.3f} "
                 f"| {st.get('mean', 0):.3f}±{st.get('std', 0):.3f} |")
    L.append("")

    # Min regime distance
    L.append("### Cross-Category Min Regime Distance\n")
    L.append("| Pair | Mean | Std |")
    L.append("|------|------|-----|")
    for pair in CAT_PAIRS:
        d = agg.get(f"min_regime_dist_{pair}", {})
        if d:
            L.append(f"| {pair} | {d.get('mean', 0):.3f} | {d.get('std', 0):.3f} |")
    L.append("")

    path = os.path.join(out, "aggregate_bottom_report.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"  [OK] aggregate_bottom_report.md")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate bottom-level analysis across runs")
    parser.add_argument("--bottom_dir", type=str, required=True,
                        help="Dir containing bottom_analysis.json files")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter: only include these models (e.g. stratos)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Filter: only include these datasets (e.g. aime24)")
    parser.add_argument("--output_dir", type=str, default="aggregate_bottom_results")
    args = parser.parse_args()

    setup_style()
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # Load all
    found = discover_jsons(args.bottom_dir, models=args.models, datasets=args.datasets)
    print(f"Found {len(found)} bottom_analysis.json files")

    all_metrics = []
    all_step_details = [] 
    all_cat_dists = []    
    all_regime_data = []  
    all_dir_correctness = []  
    all_explicit_bridge = []  
    all_step_trajectories = []    
    all_soft_profiles = []        
    per_seed_data = []    
    per_md_data = defaultdict(lambda: {
        "seeds": [],
        "regime_characteristics": None,
        "step_trajectories": None, "soft_profiles": None,
        "explicit_bridge": None, "rep_details": None,
        "category_distributions": None,
    })
    # Pooled run (seed_name=="bottom"): one entry per md — used as mean
    pooled_metrics = {}             # md_key → metrics_dict
    pooled_dir_correctness = {}     # md_key → dcc_dict
    # Per-seed runs (seed_name!="bottom"): multiple per md — used for std
    per_seed_metrics_by_md = defaultdict(list)    # md_key → [metrics_dict, ...]
    per_seed_cat_dists_by_md = defaultdict(list)  # md_key → [(label, cd), ...]
    per_seed_soft_profiles_by_md = defaultdict(list)  # md_key → [soft_profiles_dict, ...]
    per_md_dir_correctness = defaultdict(list)    # md_key → [dcc_dict, ...] (pooled + per-seed)
    seen_models = set()
    seen_datasets = set()

    def _parse_label(label):
        """Extract (model, dataset, seed) from label like 'model/dataset/seed'."""
        parts = label.split("/")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return parts[0], parts[1], "unknown"
        return label, "unknown", "unknown"

    for label, path in found:
        d = load_json(path)
        if d:
            model_name, dataset_name, seed_name = _parse_label(label)
            md_key = f"{model_name}/{dataset_name}"
            seen_models.add(model_name)
            seen_datasets.add(dataset_name)

            is_pooled = (seed_name == "bottom")

            m = extract_metrics(d)
            if is_pooled:
                # Pooled: contributes to global aggregation (cross-md mean/std)
                all_metrics.append(m)
                pooled_metrics[md_key] = m
            else:
                # Per-seed: contributes to within-md std only
                per_seed_metrics_by_md[md_key].append(m)

            # Collect sampled step details if present
            rsd = d.get("sampled_step_details")
            if is_pooled and rsd:
                all_step_details.append(rsd)
            # Collect category distributions (label kept for source tracking)
            cd = d.get("category_distributions")
            if is_pooled and cd:
                all_cat_dists.append((label, cd))
            if not is_pooled and cd:
                per_seed_cat_dists_by_md[md_key].append((label, cd))
            # Collect regime characteristics for website visualization
            rc_chars = d.get("regime_characteristics")
            if is_pooled and rc_chars:
                all_regime_data.append((label, rc_chars))
            # Collect direction correctness comparison for aggregation
            dcc = d.get("direction_correctness_comparison")
            if is_pooled and dcc:
                all_dir_correctness.append(dcc)
            if dcc:
                per_md_dir_correctness[md_key].append(dcc)
                if is_pooled:
                    pooled_dir_correctness[md_key] = dcc
            # Collect regime top transitions and step trajectories (pooled only)
            rtt = d.get("explicit_bridge")
            if is_pooled and rtt:
                all_explicit_bridge.append((label, rtt))
            st = d.get("step_trajectories")
            if is_pooled and st:
                all_step_trajectories.append((label, st))
            sp = d.get("soft_profiles")
            if is_pooled and sp:
                all_soft_profiles.append((label, sp))
            if not is_pooled and sp:
                per_seed_soft_profiles_by_md[md_key].append(sp)
            # Store per-seed data for Seeds comparison tab (only actual seeds, not pooled)
            if not is_pooled:
                per_seed_data.append({
                    "label": label,
                    "model": model_name,
                    "dataset": dataset_name,
                    "seed": seed_name,
                })

            # Per model/dataset: store first available for each data type
            # Prefer pooled run; only use per-seed if pooled not yet available
            md = per_md_data[md_key]
            if not is_pooled:
                md["seeds"].append(seed_name)
            # Prefer pooled run for qualitative/visualization data; only use per-seed as fallback
            if rc_chars and (is_pooled or md["regime_characteristics"] is None):
                md["regime_characteristics"] = rc_chars
                md["regime_source_run"] = label
                md["cat_dist_source_run"] = label
            if st and (is_pooled or md["step_trajectories"] is None):
                md["step_trajectories"] = st
                md["step_trajectories_source_run"] = label
            if sp and (is_pooled or md["soft_profiles"] is None):
                md["soft_profiles"] = sp
                md["soft_profiles_source_run"] = label
            if rtt and (is_pooled or md["explicit_bridge"] is None):
                md["explicit_bridge"] = rtt
                md["explicit_bridge_source_run"] = label
            if rsd and (is_pooled or md["rep_details"] is None):
                md["rep_details"] = rsd
            if cd and (is_pooled or md["category_distributions"] is None):
                md["category_distributions"] = cd

            print(f"  [{label}] loaded" + (" (+step_details)" if rsd else ""))

    if not all_metrics:
        print("No data. Exiting.")
        return

    # Aggregate
    agg = aggregate(all_metrics)
    n_runs = len(all_metrics)
    print(f"\nAggregated {n_runs} runs")

    # Print key results
    print("\n" + "=" * 60)
    print("AVERAGED BOTTOM-LEVEL RESULTS")
    print("=" * 60)

    print("\n--- Category Cosine ---")
    for pair in CAT_PAIRS:
        d = agg.get(f"cos_{pair}", {})
        print(f"  {pair}: {d.get('mean', 0):+.4f} ± {d.get('std', 0):.4f}")

    print("\n--- Boundary JS ---")
    for cat in CAT_TAGS:
        d = agg.get(f"boundary_js_{cat}", {})
        print(f"  {cat}: {d.get('mean', 0):.4f} ± {d.get('std', 0):.4f}")

    print("\n--- Inter-Regime Distance ---")
    for cat in CAT_TAGS:
        d = agg.get(f"inter_regime_dist_{cat}", {})
        print(f"  {cat}: {d.get('mean', 0):.3f} ± {d.get('std', 0):.3f}")

    print("\n--- Transition Direction Consistency ---")
    td_keys = sorted(set(k.replace("td_cons_", "") for k in agg if k.startswith("td_cons_")))
    for k in td_keys:
        d = agg.get(f"td_cons_{k}", {})
        print(f"  {k}: {d.get('mean', 0):.3f} ± {d.get('std', 0):.3f}")

    # Plots
    print("\n--- Generating figures ---")
    plot_cosine_heatmap(agg, out)
    print("  [OK] category_cosine_avg.png")
    plot_boundary_js(agg, out)
    print("  [OK] boundary_avg.png")
    plot_transition_directions(agg, out)
    plot_inter_regime_distance(agg, out)
    print("  [OK] regime_geometry_avg.png")
    plot_min_regime_distance(agg, out)
    print("  [OK] min_regime_dist_avg.png")
    # Report
    write_report(agg, n_runs, out)

    # JSON
    merged_step_details = None
    if all_step_details:
        merged_step_details = {}
        for group in ["correct", "long_fail", "short_fail"]:
            pool = []
            seen_labels = set()
            for rd in all_step_details:
                for seq in rd.get(group, []):
                    # Deduplicate by label tuple
                    key = tuple(seq.get("labels", []))
                    if key and key not in seen_labels:
                        seen_labels.add(key)
                        pool.append(seq)
            merged_step_details[group] = pool
        # For contrast_pair, pick the first available
        for rd in all_step_details:
            cp = rd.get("contrast_pair")
            if cp:
                merged_step_details["contrast_pair"] = cp
                break
        n_det = sum(len(merged_step_details.get(g, [])) for g in ["correct", "long_fail", "short_fail"])
        print(f"\n--- Representative Step Details ---")
        print(f"  {n_det} sequences with per-step hidden state data (merged from {len(all_step_details)} runs)")

    # Merge category distributions — use first available run.
    merged_cat_dists = None
    cat_dist_source_run = None
    if all_cat_dists:
        best_lbl_cd, best_cd = all_cat_dists[0]
        merged_cat_dists = {tag: dict(vals) for tag, vals in best_cd.items()}
        cat_dist_source_run = best_lbl_cd
        total_pts = sum(v.get("n_steps", 0) for v in best_cd.values())
        print(f"\n--- Category Distributions ---")
        print(f"  Source run: {best_lbl_cd} ({total_pts} total steps)")
        # Global centroid: mean and std across model/dataset (pooled runs only)
        for tag in merged_cat_dists:
            seed_m0 = [cd[tag]["mean_0"] for _, cd in all_cat_dists
                       if tag in cd and "mean_0" in cd[tag]]
            seed_m1 = [cd[tag]["mean_1"] for _, cd in all_cat_dists
                       if tag in cd and "mean_1" in cd[tag]]
            if seed_m0:
                merged_cat_dists[tag]["mean_0"] = round(float(np.mean(seed_m0)), 4)
                merged_cat_dists[tag]["mean_1"] = round(float(np.mean(seed_m1)), 4)
            if len(seed_m0) >= 2:
                merged_cat_dists[tag]["min_0"] = round(float(np.min(seed_m0)), 4)
                merged_cat_dists[tag]["max_0"] = round(float(np.max(seed_m0)), 4)
                merged_cat_dists[tag]["min_1"] = round(float(np.min(seed_m1)), 4)
                merged_cat_dists[tag]["max_1"] = round(float(np.max(seed_m1)), 4)

    # Aggregate direction correctness comparison across runs (weighted by n observations)
    merged_dir_correctness = None
    if all_dir_correctness:
        sums = defaultdict(lambda: {"sum_cos": 0.0, "n_correct": 0, "n_incorrect": 0,
                                    "sum_mag_c": 0.0, "sum_mag_i": 0.0, "weight": 0})
        for dcc in all_dir_correctness:
            for key, val in dcc.items():
                if key.startswith("_") or not isinstance(val, dict):
                    continue
                w = (val.get("n_correct") or 0) + (val.get("n_incorrect") or 0)
                if w == 0 or val.get("dir_cosine") is None:
                    continue
                sums[key]["sum_cos"] += val["dir_cosine"] * w
                sums[key]["n_correct"] += val.get("n_correct") or 0
                sums[key]["n_incorrect"] += val.get("n_incorrect") or 0
                if val.get("magnitude_correct") is not None:
                    sums[key]["sum_mag_c"] += val["magnitude_correct"] * w
                if val.get("magnitude_incorrect") is not None:
                    sums[key]["sum_mag_i"] += val["magnitude_incorrect"] * w
                sums[key]["weight"] += w
        merged_dir_correctness = {}
        for key, s in sums.items():
            w = s["weight"]
            merged_dir_correctness[key] = {
                "dir_cosine": round(s["sum_cos"] / w, 6),
                "n_correct": s["n_correct"],
                "n_incorrect": s["n_incorrect"],
                "magnitude_correct": round(s["sum_mag_c"] / w, 6) if s["sum_mag_c"] else None,
                "magnitude_incorrect": round(s["sum_mag_i"] / w, 6) if s["sum_mag_i"] else None,
            }
        print(f"\n--- Direction Correctness Comparison ---")
        print(f"  Aggregated {len(merged_dir_correctness)} transitions from {len(all_dir_correctness)} runs")

    # Select regime_characteristics — use first available run.
    merged_regime_chars = None
    best_regime_label = None
    if all_regime_data:
        best_lbl, best_rc = all_regime_data[0]
        merged_regime_chars = best_rc
        best_regime_label = best_lbl
        n_cats_with_pca = 0
        if merged_regime_chars:
            n_cats_with_pca = sum(
                1 for v in merged_regime_chars.values()
                if isinstance(v, dict) and "pca2d_x" in v
            )
        print(f"\n--- Regime Characteristics ---")
        print(f"  Source run: {best_lbl}"
              f" ({n_cats_with_pca} cats with PCA2D coords)")

    json_result = {
        "n_runs": n_runs,
        "models": sorted(seen_models),
        "datasets": sorted(seen_datasets),
        "metrics": {k: v for k, v in agg.items()},
    }
    if merged_step_details:
        json_result["sampled_step_details"] = merged_step_details
    if merged_cat_dists:
        json_result["category_distributions"] = merged_cat_dists
        json_result["cat_dist_source_run"] = cat_dist_source_run
    if merged_dir_correctness:
        json_result["direction_correctness_comparison"] = merged_dir_correctness
    if merged_regime_chars:
        json_result["regime_source_run"] = best_regime_label
        json_result["regime_characteristics"] = merged_regime_chars
    if per_seed_data:
        json_result["per_seed_data"] = per_seed_data
    if all_explicit_bridge:
        rtt_lbl, rtt_data = all_explicit_bridge[0]
        json_result["explicit_bridge"] = rtt_data
        json_result["explicit_bridge_source_run"] = rtt_lbl
        print(f"\n--- Explicit Bridge Transitions ---")
        print(f"  Source run: {rtt_lbl} (K={rtt_data.get('K', '?')})")
    if all_step_trajectories:
        st_lbl, st_data = all_step_trajectories[0]
        json_result["step_trajectories"] = st_data
        json_result["step_trajectories_source_run"] = st_lbl
        print(f"\n--- Step Trajectories ---")
        print(f"  Source run: {st_lbl}")
    if all_soft_profiles:
        sp_lbl, sp_data = all_soft_profiles[0]
        json_result["soft_profiles"] = sp_data
        json_result["soft_profiles_source_run"] = sp_lbl
        print(f"\n--- Soft Profiles ---")
        print(f"  Source run: {sp_lbl} ({len(sp_data)} categories)")

    # Per model/dataset data for website selector
    if per_md_data:
        md_out = {}
        for md_key, md in per_md_data.items():
            entry = {"seeds": md["seeds"]}
            model_name, dataset_name = md_key.split("/", 1)
            entry["model"] = model_name
            entry["dataset"] = dataset_name
            for field in ["regime_characteristics",
                          "step_trajectories", "soft_profiles",
                          "explicit_bridge",
                          "category_distributions",
                          "rep_details"]:
                if md.get(field) is not None:
                    if field == "regime_characteristics":
                        entry[field] = md[field]
                    elif field == "rep_details":
                        entry["sampled_step_details"] = md[field]
                    else:
                        entry[field] = md[field]
            for src_field in ["regime_source_run", "cat_dist_source_run",
                              "step_trajectories_source_run",
                              "soft_profiles_source_run",
                              "explicit_bridge_source_run"]:
                if md.get(src_field):
                    entry[src_field] = md[src_field]
            # Per-md metrics: mean from pooled run, std from per-seed runs
            pooled_m = pooled_metrics.get(md_key)
            seed_ms = per_seed_metrics_by_md.get(md_key, [])
            if pooled_m:
                md_metrics_agg = {}
                for k, v in pooled_m.items():
                    if v is None:
                        continue
                    seed_vals = [float(sm[k]) for sm in seed_ms
                                 if k in sm and sm[k] is not None]
                    md_metrics_agg[k] = {
                        "mean": float(v),
                        "std": float(np.std(seed_vals, ddof=1)) if len(seed_vals) >= 2 else 0.0,
                        "n": len(seed_vals),
                    }
                entry["metrics"] = md_metrics_agg
            elif seed_ms:
                # Fallback: no pooled run, use aggregate of per-seed
                entry["metrics"] = {k: v for k, v in aggregate(seed_ms).items()}

            # Per-md centroid std: mean from pooled category_distributions, std from per-seed
            md_cd = md.get("category_distributions")
            seed_cds = per_seed_cat_dists_by_md.get(md_key, [])
            if md_cd and seed_cds:
                merged_md_cd = {tag: dict(vals) for tag, vals in md_cd.items()}
                for tag in merged_md_cd:
                    s_m0 = [cd[tag]["mean_0"] for _, cd in seed_cds
                            if tag in cd and "mean_0" in cd[tag]]
                    s_m1 = [cd[tag]["mean_1"] for _, cd in seed_cds
                            if tag in cd and "mean_1" in cd[tag]]
                    if len(s_m0) >= 2:
                        merged_md_cd[tag]["min_0"] = round(float(np.min(s_m0)), 4)
                        merged_md_cd[tag]["max_0"] = round(float(np.max(s_m0)), 4)
                        merged_md_cd[tag]["min_1"] = round(float(np.min(s_m1)), 4)
                        merged_md_cd[tag]["max_1"] = round(float(np.max(s_m1)), 4)
                entry["category_distributions"] = merged_md_cd

            # Per-md soft_profiles std: pooled run as mean, per-seed for std
            md_sp = md.get("soft_profiles")
            seed_sps = per_seed_soft_profiles_by_md.get(md_key, [])
            if md_sp and len(seed_sps) >= 2:
                merged_sp = {}
                for tag, pooled_tag_data in md_sp.items():
                    merged_tag = dict(pooled_tag_data)
                    for group in ["correct", "incorrect"]:
                        if group not in merged_tag:
                            continue
                        # Collect per-seed mean_profile matrices for this tag+group
                        seed_profiles = []
                        for ssp in seed_sps:
                            if tag in ssp and group in ssp[tag] and "mean_profile" in ssp[tag][group]:
                                seed_profiles.append(np.array(ssp[tag][group]["mean_profile"]))
                        if len(seed_profiles) >= 2:
                            stacked = np.stack(seed_profiles)  # (n_seeds, L, K)
                            std_profile = np.std(stacked, axis=0, ddof=1)  # (L, K)
                            merged_tag[group] = dict(merged_tag[group])
                            merged_tag[group]["std_profile"] = std_profile.round(4).tolist()
                    # std for diff
                    if "diff_correct_minus_incorrect" in merged_tag:
                        seed_diffs = []
                        for ssp in seed_sps:
                            if tag in ssp and "diff_correct_minus_incorrect" in ssp[tag]:
                                seed_diffs.append(np.array(ssp[tag]["diff_correct_minus_incorrect"]))
                        if len(seed_diffs) >= 2:
                            stacked = np.stack(seed_diffs)
                            merged_tag["std_diff"] = np.std(stacked, axis=0, ddof=1).round(4).tolist()
                            merged_tag["n_seeds_std"] = len(seed_diffs)
                    merged_sp[tag] = merged_tag
                entry["soft_profiles"] = merged_sp

            # Per-md direction correctness comparison (weighted merge)
            md_dccs = per_md_dir_correctness.get(md_key, [])
            if md_dccs:
                dcc_sums = defaultdict(lambda: {"sum_cos": 0.0, "n_correct": 0,
                                                "n_incorrect": 0, "sum_mag_c": 0.0,
                                                "sum_mag_i": 0.0, "weight": 0})
                for dcc_item in md_dccs:
                    for dk, dv in dcc_item.items():
                        if dk.startswith("_") or not isinstance(dv, dict):
                            continue
                        w = (dv.get("n_correct") or 0) + (dv.get("n_incorrect") or 0)
                        if w == 0 or dv.get("dir_cosine") is None:
                            continue
                        dcc_sums[dk]["sum_cos"] += dv["dir_cosine"] * w
                        dcc_sums[dk]["n_correct"] += dv.get("n_correct") or 0
                        dcc_sums[dk]["n_incorrect"] += dv.get("n_incorrect") or 0
                        if dv.get("magnitude_correct") is not None:
                            dcc_sums[dk]["sum_mag_c"] += dv["magnitude_correct"] * w
                        if dv.get("magnitude_incorrect") is not None:
                            dcc_sums[dk]["sum_mag_i"] += dv["magnitude_incorrect"] * w
                        dcc_sums[dk]["weight"] += w
                entry["direction_correctness_comparison"] = {
                    dk: {
                        "dir_cosine": round(ds["sum_cos"] / ds["weight"], 6),
                        "n_correct": ds["n_correct"],
                        "n_incorrect": ds["n_incorrect"],
                        "magnitude_correct": round(ds["sum_mag_c"] / ds["weight"], 6) if ds["sum_mag_c"] else None,
                        "magnitude_incorrect": round(ds["sum_mag_i"] / ds["weight"], 6) if ds["sum_mag_i"] else None,
                    }
                    for dk, ds in dcc_sums.items()
                }

            md_out[md_key] = entry
        json_result["per_model_dataset"] = md_out
        print(f"\n--- Per Model/Dataset ---")
        for md_key in sorted(md_out):
            fields = [f for f in ["step_trajectories",
                                  "soft_profiles", "explicit_bridge",
                                  "metrics"]
                      if f in md_out[md_key]]
            print(f"  {md_key}: seeds={md_out[md_key]['seeds']}, "
                  f"data=[{', '.join(fields)}]")

    with open(os.path.join(out, "aggregate_bottom.json"), "w") as f:
        json.dump(json_result, f, indent=2)
    print("  [OK] aggregate_bottom.json")

    print(f"\n{'=' * 60}")
    print(f"DONE — {out}/")
    print(f"{'=' * 60}")
    for fn in sorted(os.listdir(out)):
        print(f"  {fn}")


if __name__ == "__main__":
    main()
