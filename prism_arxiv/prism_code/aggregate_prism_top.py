#!/usr/bin/env python3
"""Aggregate top analysis.

Usage:
    python aggregate_prism_top.py \
        --top_dir /scratch/$USER/reasoning_newstart/analysis_top_joint \
        --output_dir aggregate_results
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
    C, TAG_SHORT, CAT_COLORS, SAVE_KW,
    setup_style, discover_jsons as _discover, load_json,
    plot_matrix, plot_diff_matrix, renormalize_4x4,
)


def discover_jsons(base_dir, models=None, datasets=None):
    return _discover(base_dir, "analysis.json", models=models, datasets=datasets)


def extract_from_json(data: dict) -> dict:
    """Extract transition matrices, population stats, and representatives."""
    result = {
        "matrices": {}, "weights": {},
        "start_all": None,
        "population": {},
        "sampled_sequences": None,
    }

    ci = data.get("correct_vs_incorrect", {})
    n_corr = ci.get("num_correct", 0)
    n_incorr = ci.get("num_incorrect", 0)

    # ── Transition matrices ──────────────────────────────────────────
    if "correct_transition_matrix" in ci and n_corr > 0:
        result["matrices"]["correct"] = renormalize_4x4(ci["correct_transition_matrix"])
        result["weights"]["correct"] = n_corr

    if "incorrect_transition_matrix" in ci and n_incorr > 0:
        result["matrices"]["incorrect"] = renormalize_4x4(ci["incorrect_transition_matrix"])
        result["weights"]["incorrect"] = n_incorr

    n_long = ci.get("num_long_fail", 0)
    n_short = ci.get("num_short_fail", 0)
    if "long_fail_transition_matrix" in ci and n_long > 0:
        result["matrices"]["long_fail"] = renormalize_4x4(ci["long_fail_transition_matrix"])
        result["weights"]["long_fail"] = n_long
    if "short_fail_transition_matrix" in ci and n_short > 0:
        result["matrices"]["short_fail"] = renormalize_4x4(ci["short_fail_transition_matrix"])
        result["weights"]["short_fail"] = n_short

    if "correct" in result["matrices"] and "incorrect" in result["matrices"]:
        total = n_corr + n_incorr
        if total > 0:
            result["matrices"]["all"] = (
                n_corr * result["matrices"]["correct"]
                + n_incorr * result["matrices"]["incorrect"]
            ) / total
            result["weights"]["all"] = total

    # ── Start distribution ───────────────────────────────────────────
    bs = data.get("basic_stats_order1", {})
    sd = bs.get("start_distribution", None)
    if sd is not None and len(sd) >= C:
        s4 = np.array(sd[:C], dtype=np.float64)
        s = s4.sum()
        result["start_all"] = (s4 / s) if s > 0 else s4

    # ── Population-level stats ───────────────────────────────────────
    pop = result["population"]

    # Path lengths
    pl = data.get("path_lengths", {})
    pop["path_length_mean"] = pl.get("mean")
    pop["path_length_median"] = pl.get("median")
    pop["path_length_std"] = pl.get("std")
    pop["path_length_min"] = pl.get("min")
    pop["path_length_max"] = pl.get("max")
    pop["path_length_percentiles"] = pl.get("percentiles", {})

    # Correct vs incorrect path lengths
    pop["correct_path_length_mean"] = ci.get("correct_path_length_mean")
    pop["incorrect_path_length_mean"] = ci.get("incorrect_path_length_mean")

    # Start/end patterns
    sep = data.get("start_end_patterns", {})
    pop["start_distribution_raw"] = sep.get("start_distribution")
    pop["end_distribution_raw"] = sep.get("end_distribution")

    # Correct/incorrect counts
    pop["num_correct"] = n_corr
    pop["num_incorrect"] = n_incorr
    pop["long_failures"] = ci.get("num_long_fail", 0)
    pop["short_failures"] = ci.get("num_short_fail", 0)

    # Markov chain properties
    mc = data.get("markov_chain", {})
    pop["stationary_distribution"] = mc.get("stationary_distribution")
    pop["expected_steps_to_final_answer"] = mc.get("expected_steps_to_final_answer")

    # N-grams
    pop["top_3grams"] = data.get("top_3grams", [])[:10]

    # ── Nth-order transition matrices (N = 2, 3, 4, 5) ───────────────
    # analyze_prism_top.py stores matrices as (5^N, 5) including the unknown category.
    # We extract the (4^N, 4) submatrix for the 4 known categories only.
    C5 = 5  # number of categories including unknown in source matrices

    def _extract_higher_order_matrix(ci_n: dict, order: int, key: str):
        """Extract and renormalize the 4^order × 4 submatrix from a 5^order × 5 matrix."""
        mat_key = f"{key}_transition_matrix"
        if mat_key not in ci_n:
            return None
        mat = np.array(ci_n[mat_key], dtype=np.float64)
        n_ctx_5 = C5 ** order
        n_ctx_4 = C ** order
        # Accept both possible shapes
        if mat.shape == (n_ctx_5, C5):
            # Build 4^order × 4 submatrix by selecting only contexts with all categories < C
            mat4 = np.zeros((n_ctx_4, C), dtype=np.float64)
            # Enumerate all valid 4-category context tuples
            import itertools
            for ctx4 in itertools.product(range(C), repeat=order):
                # Map 4-based index to 5-based index
                idx5 = 0
                idx4 = 0
                for c in ctx4:
                    idx5 = idx5 * C5 + c
                    idx4 = idx4 * C + c
                mat4[idx4, :] = mat[idx5, :C]
            row_sums = mat4.sum(axis=1, keepdims=True)
            safe = np.where(row_sums > 0, row_sums, 1.0)
            return np.where(row_sums > 0, mat4 / safe, 0.0)
        elif mat.shape == (n_ctx_4, C):
            return mat
        return None

    result["matrices_2nd"] = {}
    ci2 = data.get("correct_vs_incorrect_2nd", {})
    for key in ("correct", "incorrect"):
        m = _extract_higher_order_matrix(ci2, order=2, key=key)
        if m is not None:
            result["matrices_2nd"][key] = m
    if "correct" in result["matrices_2nd"] and "incorrect" in result["matrices_2nd"]:
        nc2 = ci2.get("num_correct", n_corr)
        ni2 = ci2.get("num_incorrect", n_incorr)
        total2 = nc2 + ni2
        if total2 > 0:
            result["matrices_2nd"]["all"] = (
                nc2 * result["matrices_2nd"]["correct"]
                + ni2 * result["matrices_2nd"]["incorrect"]
            ) / total2

    for _ord, _suffix in [(3, "3rd"), (4, "4th"), (5, "5th")]:
        ci_n = data.get(f"correct_vs_incorrect_{_suffix}", {})
        key_name = f"matrices_{_suffix}"
        result[key_name] = {}
        for key in ("correct", "incorrect"):
            m = _extract_higher_order_matrix(ci_n, order=_ord, key=key)
            if m is not None:
                result[key_name][key] = m
        if "correct" in result[key_name] and "incorrect" in result[key_name]:
            nc_n = ci_n.get("num_correct", n_corr)
            ni_n = ci_n.get("num_incorrect", n_incorr)
            total_n = nc_n + ni_n
            if total_n > 0:
                result[key_name]["all"] = (
                    nc_n * result[key_name]["correct"]
                    + ni_n * result[key_name]["incorrect"]
                ) / total_n

    # ── Representative sequences ─────────────────────────────────────
    result["sampled_sequences"] = data.get("sampled_sequences")

    # ── Per-seed matrices (for cross-seed std in website) ────────────
    result["per_seed_matrices"] = data.get("per_seed_matrices", {})

    return result


# ── Visualization ────────────────────────────────────────────────────

def plot_all_1st_order(matrices: Dict[str, np.ndarray], out_dir: str):
    """Plot all 1st-order matrices + differences."""
    names = ["all", "correct", "incorrect", "long_fail", "short_fail"]
    titles = ["All", "Correct", "Incorrect", "Long Failures", "Short Failures"]

    for name, title in zip(names, titles):
        mat = matrices.get(name)
        if mat is not None:
            plot_matrix(mat, f"1st-Order: {title}",
                        os.path.join(out_dir, f"1st_order_{name}.png"))

    available = [(n, t) for n, t in zip(names, titles) if n in matrices]
    if len(available) >= 2:
        fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]
        for ax, (name, title) in zip(axes, available):
            mat = matrices[name]
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
            for i in range(C):
                for j in range(C):
                    color = "white" if mat[i, j] > 0.5 else "black"
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                            fontsize=10, color=color)
            ax.set_xticks(range(C))
            ax.set_yticks(range(C))
            ax.set_xticklabels(TAG_SHORT)
            ax.set_yticklabels(TAG_SHORT)
            ax.set_title(title)
        fig.suptitle("1st-Order Transition Matrices (Averaged)", fontsize=18, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "1st_order_panel.png"), **SAVE_KW)
        plt.close(fig)

    if "correct" in matrices and "incorrect" in matrices:
        diff = matrices["correct"] - matrices["incorrect"]
        plot_diff_matrix(diff, "1st-Order Diff: Correct − Incorrect",
                         os.path.join(out_dir, "1st_order_diff_corr_incorr.png"))

    if "long_fail" in matrices and "short_fail" in matrices:
        diff = matrices["long_fail"] - matrices["short_fail"]
        plot_diff_matrix(diff, "1st-Order Diff: Long − Short Failures",
                         os.path.join(out_dir, "1st_order_diff_long_short.png"))


def plot_start_probs(start_probs: Dict[str, np.ndarray], out_dir: str):
    """Bar chart of start probabilities."""
    if "all" not in start_probs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(C)
    probs = start_probs["all"]
    colors = [CAT_COLORS[t] for t in TAG_SHORT]
    bars = ax.bar(x, probs, color=colors, edgecolor="black", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(TAG_SHORT)
    ax.set_ylabel("Start Probability")
    ax.set_title("Start Category Distribution (All Sequences, Averaged)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "start_prob.png"), **SAVE_KW)
    plt.close(fig)


# ── Aggregation helpers ──────────────────────────────────────────────

def aggregate_scalar(all_extracts: list, key: str) -> dict:
    """Compute mean/std for a scalar population metric across runs."""
    vals = []
    for _, ext in all_extracts:
        v = ext["population"].get(key)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def merge_sampled_sequences(all_extracts: list, n_per_group: int = 3) -> dict:
    """Merge sampled sequences across runs.
    """
    groups = defaultdict(list)

    for label, ext in all_extracts:
        reps = ext.get("sampled_sequences")
        if not reps:
            continue
        for group_name in ["correct", "long_fail", "short_fail"]:
            for seq in reps.get(group_name, []):
                seq_copy = dict(seq)
                seq_copy["source_run"] = label
                groups[group_name].append(seq_copy)

    # Keep n_per_group per group (random order from analysis)
    result = {}
    for group_name, seqs in groups.items():
        result[group_name] = seqs[:n_per_group]

    return result



# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate transition matrices across runs (reads analysis.json)")
    parser.add_argument("--top_dir", type=str, required=True,
                        help="Base dir containing analysis.json files")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter: only include these models (e.g. stratos)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Filter: only include these datasets (e.g. aime24)")
    parser.add_argument("--output_dir", type=str, default="aggregate_results")
    parser.add_argument("--bottom_analysis", type=str, default=None,
                        help="Path to bottom_analysis.json to extract regime-level danger thresholds")
    args = parser.parse_args()

    setup_style()
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── Discover analysis.json files ─────────────────────────────────
    runs = discover_jsons(args.top_dir, models=args.models, datasets=args.datasets)
    if not runs:
        print(f"No analysis.json found under {args.top_dir}")
        return
    print(f"Found {len(runs)} analysis.json files\n")

    # ── Load and extract per-run data ────────────────────────────────
    all_extracts = []
    seen_models = set()
    seen_datasets = set()
    # Per model/dataset grouping: label → (model, dataset)
    label_to_md = {}
    for label, path in runs:
        parts = label.split("/")
        if len(parts) >= 2:
            seen_models.add(parts[0])
            seen_datasets.add(parts[1])
            label_to_md[label] = (parts[0], parts[1])
        with open(path) as f:
            data = json.load(f)
        ext = extract_from_json(data)
        all_extracts.append((label, ext))
        n = data.get("num_sequences", "?")
        nc = data.get("correct_vs_incorrect", {}).get("num_correct", "?")
        ni = data.get("correct_vs_incorrect", {}).get("num_incorrect", "?")
        print(f"  [{label}] {n} seqs (correct={nc}, incorrect={ni})")

    # ── Aggregate: weighted average of 4×4 matrices ──────────────────
    group_names = ["all", "correct", "incorrect", "long_fail", "short_fail"]
    group_titles = ["All", "Correct", "Incorrect", "Long Failures", "Short Failures"]
    matrices_avg = {}
    group_total_seqs = {}

    print(f"\n{'=' * 60}")
    print("1st-ORDER TRANSITION MATRICES (weighted avg, excluding unknown)")
    print("=" * 60)

    for group in group_names:
        mats = []
        weights = []
        for _, ext in all_extracts:
            if group in ext["matrices"]:
                mats.append(ext["matrices"][group])
                weights.append(ext["weights"][group])
        if not mats:
            continue
        weights_arr = np.array(weights, dtype=np.float64)
        total_w = weights_arr.sum()
        avg = sum(w * m for w, m in zip(weights, mats)) / total_w
        matrices_avg[group] = avg
        group_total_seqs[group] = int(total_w)

        title = group_titles[group_names.index(group)]
        print(f"\n--- {title} ({len(mats)} runs, {int(total_w)} total seqs) ---")
        header = "     " + "  ".join(f"{t:>7s}" for t in TAG_SHORT)
        print(header)
        for i in range(C):
            row = f"{TAG_SHORT[i]:>4s} " + "  ".join(f"{avg[i, j]:7.4f}" for j in range(C))
            print(row)

    plot_all_1st_order(matrices_avg, out)
    print(f"\n[OK] 1st-order figures saved to {out}/")

    # ── Aggregate: 2nd–5th-order matrices ──────────────────────────
    matrices_2nd_avg = {}
    for group in ["all", "correct", "incorrect"]:
        mats = []
        for _, ext in all_extracts:
            m2 = ext.get("matrices_2nd", {})
            if group in m2:
                mats.append(m2[group])
        if mats:
            matrices_2nd_avg[group] = np.mean(mats, axis=0)
            print(f"\n  2nd-order {group}: {len(mats)} runs, shape={mats[0].shape}")

    matrices_higher_avg = {}  # key: suffix ("3rd", "4th", "5th") → {group → matrix}
    for _ord, _suffix in [(3, "3rd"), (4, "4th"), (5, "5th")]:
        key_name = f"matrices_{_suffix}"
        avg_n = {}
        for group in ["all", "correct", "incorrect"]:
            mats = []
            for _, ext in all_extracts:
                mn = ext.get(key_name, {})
                if group in mn:
                    mats.append(mn[group])
            if mats:
                avg_n[group] = np.mean(mats, axis=0)
                print(f"  {_ord}th-order {group}: {len(mats)} runs, shape={mats[0].shape}")
        if avg_n:
            matrices_higher_avg[_suffix] = avg_n

    # ── Start probabilities ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("START PROBABILITIES (excluding unknown)")
    print("=" * 60)

    starts = []
    for _, ext in all_extracts:
        if ext["start_all"] is not None:
            starts.append(ext["start_all"])

    start_probs = {}
    if starts:
        start_avg = np.mean(starts, axis=0)
        start_probs["all"] = start_avg
        print(f"\n  all ({len(starts)} runs): "
              + "  ".join(f"{TAG_SHORT[i]}={start_avg[i]:.4f}" for i in range(C)))
        plot_start_probs(start_probs, out)
        print(f"  [OK] start_prob.png saved")
    else:
        print("  [SKIP] no start distributions found")

    # ── Aggregate population stats ───────────────────────────────────
    print(f"\n{'=' * 60}")
    print("POPULATION STATISTICS")
    print("=" * 60)

    pop_agg = {}
    scalar_keys = [
        "path_length_mean", "path_length_median", "path_length_std",
        "correct_path_length_mean", "incorrect_path_length_mean",
        "num_correct", "num_incorrect",
        "long_failures", "short_failures",
    ]
    for key in scalar_keys:
        agg = aggregate_scalar(all_extracts, key)
        if agg:
            pop_agg[key] = agg
            print(f"  {key}: {agg['mean']:.2f} ± {agg['std']:.2f} (n={agg['n']})")

    # Stationary distribution (average across runs)
    stat_dists = []
    for _, ext in all_extracts:
        sd = ext["population"].get("stationary_distribution")
        if sd and len(sd) >= C:
            s4 = np.array(list(sd.values()) if isinstance(sd, dict) else sd[:C], dtype=np.float64)
            stat_dists.append(s4)
    stationary_avg = None
    if stat_dists:
        stationary_avg = {
            TAG_SHORT[i]: float(np.mean([s[i] for s in stat_dists]))
            for i in range(min(C, len(stat_dists[0])))
        }
        print(f"  stationary: {stationary_avg}")

    # Expected steps to FA (average across runs)
    hitting_times = []
    for _, ext in all_extracts:
        ht = ext["population"].get("expected_steps_to_final_answer")
        if ht:
            hitting_times.append(ht)
    hitting_avg = None
    if hitting_times:
        if isinstance(hitting_times[0], dict):
            all_keys = set()
            for h in hitting_times:
                all_keys.update(h.keys())
            all_keys.discard("unknown")
            hitting_avg = {}
            for k in all_keys:
                vals = [h[k] for h in hitting_times if k in h and h[k] is not None]
                if vals:
                    hitting_avg[k] = float(np.mean(vals))
        print(f"  hitting_times_to_FA: {hitting_avg}")

    # ── Aggregate representative sequences ───────────────────────────
    print(f"\n{'=' * 60}")
    print("REPRESENTATIVE SEQUENCES")
    print("=" * 60)

    merged_reps = merge_sampled_sequences(all_extracts, n_per_group=3)
    for group_name in ["correct", "long_fail", "short_fail"]:
        seqs = merged_reps.get(group_name, [])
        print(f"  {group_name}: {len(seqs)} examples")
        for s in seqs:
            print(f"    len={s.get('path_length', '?')}"
                  f"  q={s.get('question', '')[:60]}...")

    # ── Per model/dataset aggregation ────────────────────────────────
    per_md_extracts = defaultdict(list)  # md_key → [(label, ext), ...]
    for label, ext in all_extracts:
        if label in label_to_md:
            model, dataset = label_to_md[label]
            md_key = f"{model}/{dataset}"
            per_md_extracts[md_key].append((label, ext))

    per_model_dataset = {}
    for md_key, md_exts in sorted(per_md_extracts.items()):
        model, dataset = md_key.split("/", 1)
        md_entry = {"model": model, "dataset": dataset,
                    "seeds": [lbl.split("/")[-1] for lbl, _ in md_exts]}

        # Per-md transition matrices (weighted average)
        md_matrices = {}
        for group in group_names:
            mats, weights = [], []
            for _, ext in md_exts:
                if group in ext["matrices"]:
                    mats.append(ext["matrices"][group])
                    weights.append(ext["weights"][group])
            if mats:
                w_arr = np.array(weights, dtype=np.float64)
                md_matrices[group] = (sum(w * m for w, m in zip(weights, mats))
                                      / w_arr.sum()).tolist()
        if md_matrices:
            md_entry["1st_order_matrices"] = md_matrices

        # Per-md higher-order matrices (2nd–5th)
        for _ord, _key, _out_key in [
            (2, "matrices_2nd", "2nd_order_matrices"),
            (3, "matrices_3rd", "3rd_order_matrices"),
            (4, "matrices_4th", "4th_order_matrices"),
            (5, "matrices_5th", "5th_order_matrices"),
        ]:
            md_higher = {}
            for group in group_names:
                mats = [ext[_key][group] for _, ext in md_exts
                        if group in ext.get(_key, {})]
                if mats:
                    md_higher[group] = np.mean(mats, axis=0).tolist()
            if md_higher:
                md_entry[_out_key] = md_higher

        # Per-seed matrices for cross-seed std (collected from per_seed_matrices in each ext)
        md_seed_matrices = {}
        for group in group_names:
            all_seed_mats = []
            for _, ext in md_exts:
                psm = ext.get("per_seed_matrices", {})
                if group in psm:
                    all_seed_mats.extend(psm[group])
            if all_seed_mats:
                md_seed_matrices[group] = all_seed_mats
        if md_seed_matrices:
            md_entry["seed_matrices"] = md_seed_matrices

        # Per-md start probs
        md_starts = [ext["start_all"] for _, ext in md_exts
                     if ext["start_all"] is not None]
        if md_starts:
            s_avg = np.mean(md_starts, axis=0)
            sp_dict = {}
            for i in range(C):
                vals = [float(s[i]) for s in md_starts]
                sp_dict[TAG_SHORT[i]] = {
                    "mean": float(s_avg[i]),
                    "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0,
                }
            md_entry["start_probs"] = sp_dict

        # Per-md representative sequences
        md_reps = merge_sampled_sequences(md_exts, n_per_group=3)
        if md_reps:
            md_entry["sampled_sequences"] = md_reps

        # Per-md sequence counts
        md_seq_counts = {}
        for group in group_names:
            w_total = sum(ext["weights"].get(group, 0) for _, ext in md_exts)
            if w_total > 0:
                md_seq_counts[group] = int(w_total)
        if md_seq_counts:
            md_entry["group_sequence_counts"] = md_seq_counts

        # Per-md population stats
        md_pop = {}
        for key in scalar_keys:
            agg = aggregate_scalar(md_exts, key)
            if agg:
                md_pop[key] = agg
        if md_pop:
            md_entry["population_stats"] = md_pop

        # Per-md stationary distribution
        md_stat = []
        for _, ext in md_exts:
            sd = ext["population"].get("stationary_distribution")
            if sd and len(sd) >= C:
                s4 = np.array(list(sd.values()) if isinstance(sd, dict)
                              else sd[:C], dtype=np.float64)
                md_stat.append(s4)
        if md_stat:
            sd_dict = {}
            for i in range(min(C, len(md_stat[0]))):
                vals = [float(s[i]) for s in md_stat]
                sd_dict[TAG_SHORT[i]] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0,
                }
            md_entry["stationary_distribution"] = sd_dict

        # Per-md hitting times to FA
        md_ht = []
        for _, ext in md_exts:
            ht = ext["population"].get("expected_steps_to_final_answer")
            if ht:
                md_ht.append(ht)
        if md_ht and isinstance(md_ht[0], dict):
            ht_keys = set()
            for h in md_ht:
                ht_keys.update(h.keys())
            ht_keys.discard("unknown")
            ht_dict = {}
            for k in ht_keys:
                vals = [float(h[k]) for h in md_ht if k in h and h[k] is not None]
                if vals:
                    ht_dict[k] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0,
                    }
            md_entry["hitting_times_to_FA"] = ht_dict

        per_model_dataset[md_key] = md_entry

    if per_model_dataset:
        print(f"\n--- Per Model/Dataset ---")
        for md_key in sorted(per_model_dataset):
            e = per_model_dataset[md_key]
            print(f"  {md_key}: seeds={e['seeds']}, "
                  f"matrices={list(e.get('1st_order_matrices', {}).keys())}")

    # ══════════════════════════════════════════════════════════════════
    #  Save JSON
    # ══════════════════════════════════════════════════════════════════
    result = {
        "config": {
            "top_dir": args.top_dir,
            "num_runs": len(runs),
            "run_labels": [label for label, _ in runs],
        },
        "models": sorted(seen_models),
        "datasets": sorted(seen_datasets),
        "group_sequence_counts": group_total_seqs,
        "start_probs": {
            name: {TAG_SHORT[i]: float(prob[i]) for i in range(C)}
            for name, prob in start_probs.items()
        },
        "1st_order_matrices": {
            name: mat.tolist() for name, mat in matrices_avg.items()
        },
        "2nd_order_matrices": {
            name: mat.tolist() for name, mat in matrices_2nd_avg.items()
        },
        **{
            f"{_suffix}_order_matrices": {
                name: mat.tolist() for name, mat in matrices_higher_avg.get(_suffix, {}).items()
            }
            for _, _suffix in [(3, "3rd"), (4, "4th"), (5, "5th")]
            if matrices_higher_avg.get(_suffix)
        },
        "population_stats": pop_agg,
        "stationary_distribution": stationary_avg,
        "hitting_times_to_FA": hitting_avg,
        "sampled_sequences": merged_reps,
        "per_model_dataset": per_model_dataset,
    }

    json_path = os.path.join(out, "aggregate_transitions.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE — results saved to {out}/")
    print("=" * 60)
    for fn in sorted(os.listdir(out)):
        print(f"  {fn}")


if __name__ == "__main__":
    main()
