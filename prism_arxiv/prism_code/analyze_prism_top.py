#!/usr/bin/env python3
"""
PRISM Top-Level Transition Analysis

Usage:
    python analyze_prism_top.py --prism_dir /path/to/prism_models --pt_dir /path/to/pt_files --output_dir /path/to/output
"""

import os
import argparse
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import json

from prism_lib import (
    CANON_TAGS, coerce_labels_to_ids, load_pt_records,
    tuple_to_index,
)

# ---------- Constants ----------
SHORT_TAGS = ["FA", "SR", "AC", "UV", "UN"]  # Short names for display
C = len(CANON_TAGS)
EPS = 1e-12


# ---------- Utility functions ----------
def extract_label_sequences(records: List[Dict], label_key: str = "sentence_labels") -> List[Dict]:
    """Extract label sequences from records, including correctness info."""
    sequences = []
    sample_idx = 0
    for rec in records:
        if "error" in rec:
            sample_idx += 1
            continue
        labels = rec.get(label_key, [])
        if not labels:
            sample_idx += 1
            continue

        seq = {
            "labels": coerce_labels_to_ids(labels),
            "is_correct": rec.get("is_correct", None),
            "num_sentences": len(labels),
            "question": rec.get("question", ""),
            "sample_idx": sample_idx,
        }

        # Get generation length (token count) - use gen_token_count from preprocessing
        gen_len = rec.get("gen_token_count") or rec.get("gen_len") or rec.get("output_len")
        seq["gen_len"] = gen_len

        # Check for long traces (based on step_hidden_states if available)
        step_hidden = rec.get("step_hidden_states", [])
        if step_hidden:
            seq["approx_tokens"] = len(step_hidden)

        sequences.append(seq)
        sample_idx += 1
    return sequences


# ============================================================
# 1. Basic Statistics
# ============================================================

def compute_stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Compute stationary distribution of a Markov chain (for 1st order)."""
    # Solve pi * P = pi, with sum(pi) = 1
    # Equivalent to (P^T - I) * pi = 0
    n = transmat.shape[0]
    A = transmat.T - np.eye(n)
    A[-1, :] = 1  # Replace last row with constraint sum(pi) = 1
    b = np.zeros(n)
    b[-1] = 1
    try:
        pi = np.linalg.solve(A, b)
        pi = np.maximum(pi, 0)
        pi = pi / pi.sum()
    except np.linalg.LinAlgError:
        pi = np.ones(n) / n
    return pi


def analyze_basic_stats(startprob: np.ndarray) -> Dict:
    """Analyze basic transition statistics."""
    return {
        "start_distribution": startprob,
    }


def analyze_start_end_patterns(sequences: List[Dict]) -> Dict:
    """Analyze common starting and ending patterns."""
    start_counts = Counter()
    end_counts = Counter()
    start_2gram = Counter()
    end_2gram = Counter()

    for seq in sequences:
        labels = seq["labels"]
        if len(labels) >= 1:
            start_counts[labels[0]] += 1
            end_counts[labels[-1]] += 1
        if len(labels) >= 2:
            start_2gram[(labels[0], labels[1])] += 1
            end_2gram[(labels[-2], labels[-1])] += 1

    return {
        "start_distribution": dict(start_counts),
        "end_distribution": dict(end_counts),
        "start_2gram": dict(start_2gram),
        "end_2gram": dict(end_2gram),
    }


# ============================================================
# 2. Path/Sequence Analysis
# ============================================================

def count_ngrams(sequences: List[Dict], n: int = 3) -> Counter:
    """Count n-gram occurrences in sequences."""
    ngram_counts = Counter()
    for seq in sequences:
        labels = seq["labels"]
        for i in range(len(labels) - n + 1):
            ngram = tuple(labels[i:i+n])
            ngram_counts[ngram] += 1
    return ngram_counts


def analyze_path_lengths(sequences: List[Dict]) -> Dict:
    """Analyze path lengths and their distribution."""
    lengths = [len(seq["labels"]) for seq in sequences]

    return {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "mean": np.mean(lengths) if lengths else 0,
        "median": np.median(lengths) if lengths else 0,
        "std": np.std(lengths) if lengths else 0,
        "percentiles": {
            "10": np.percentile(lengths, 10) if lengths else 0,
            "25": np.percentile(lengths, 25) if lengths else 0,
            "50": np.percentile(lengths, 50) if lengths else 0,
            "75": np.percentile(lengths, 75) if lengths else 0,
            "90": np.percentile(lengths, 90) if lengths else 0,
        },
        "length_distribution": dict(Counter(lengths)),
    }


# ============================================================
# 3. Comparison Analysis (Correct vs Incorrect)
# ============================================================

def split_by_correctness(sequences: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split sequences into correct and incorrect groups."""
    correct = [s for s in sequences if s.get("is_correct") == True]
    incorrect = [s for s in sequences if s.get("is_correct") == False]
    return correct, incorrect


def split_incorrect_by_length(sequences: List[Dict], threshold: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """Split incorrect sequences into long and short failures."""
    long_fails = [s for s in sequences if len(s["labels"]) >= threshold]
    short_fails = [s for s in sequences if len(s["labels"]) < threshold]
    return long_fails, short_fails


def count_transitions(sequences: List[Dict], order: int = 1) -> np.ndarray:
    """Count transition frequencies from sequences."""
    context_size = C ** order
    trans_counts = np.zeros((context_size, C), dtype=np.float64)

    for seq in sequences:
        labels = seq["labels"]
        for t in range(order, len(labels)):
            context = tuple(labels[t-order:t])
            ctx_idx = tuple_to_index(context, C)
            next_cat = labels[t]
            trans_counts[ctx_idx, next_cat] += 1

    return trans_counts


def normalize_transition_matrix(counts: np.ndarray) -> np.ndarray:
    """Normalize count matrix to probability matrix."""
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid divide by zero warning
    safe_row_sums = np.where(row_sums > 0, row_sums, 1.0)
    uniform = np.ones_like(counts) / max(counts.shape[1], 1)
    return np.where(row_sums > 0, counts / safe_row_sums, uniform)


def analyze_correct_vs_incorrect(sequences: List[Dict], order: int = 1) -> Dict:
    """Compare correct and incorrect paths."""
    correct, incorrect = split_by_correctness(sequences)

    if not correct or not incorrect:
        return {"error": "Need both correct and incorrect sequences for comparison"}

    # Count and normalize transitions
    correct_trans = normalize_transition_matrix(count_transitions(correct, order))
    incorrect_trans = normalize_transition_matrix(count_transitions(incorrect, order))

    # Split incorrect into long/short failures
    long_fails, short_fails = split_incorrect_by_length(incorrect, threshold=100)

    # Path length comparison
    correct_lengths = [len(s["labels"]) for s in correct]
    incorrect_lengths = [len(s["labels"]) for s in incorrect]

    result = {
        "num_correct": len(correct),
        "num_incorrect": len(incorrect),
        "correct_transition_matrix": correct_trans,
        "incorrect_transition_matrix": incorrect_trans,
        "correct_path_length_mean": np.mean(correct_lengths) if correct_lengths else 0,
        "incorrect_path_length_mean": np.mean(incorrect_lengths) if incorrect_lengths else 0,
    }

    if long_fails:
        result["long_fail_transition_matrix"] = normalize_transition_matrix(
            count_transitions(long_fails, order))
        result["num_long_fail"] = len(long_fails)

    if short_fails:
        result["short_fail_transition_matrix"] = normalize_transition_matrix(
            count_transitions(short_fails, order))
        result["num_short_fail"] = len(short_fails)

    return result


def sample_sequences_per_group(sequences: List[Dict],
                                     n_per_group: int = 3,
                                     long_threshold: int = 100) -> Dict:
    """Randomly sample sequences per group (correct/long_fail/short_fail)."""
    import random
    rng = random.Random(42)

    correct, incorrect = split_by_correctness(sequences)
    long_fails, short_fails = split_incorrect_by_length(incorrect, threshold=long_threshold)

    groups = {
        "correct": correct,
        "long_fail": long_fails,
        "short_fail": short_fails,
    }

    def _seq_entry(seq):
        return {
            "question": seq.get("question", ""),
            "labels": seq["labels"],
            "path_length": len(seq["labels"]),
            "is_correct": seq.get("is_correct"),
            "gen_len": seq.get("gen_len", 0),
            "sample_idx": seq.get("sample_idx"),
        }

    result = {}
    for group_name, group_seqs in groups.items():
        eligible = [s for s in group_seqs if len(s["labels"]) >= 3]
        if not eligible:
            continue
        sampled = rng.sample(eligible, min(n_per_group, len(eligible)))
        result[group_name] = [_seq_entry(s) for s in sampled]

    return result


# ============================================================
# 4. Markov Chain Analysis
# ============================================================

def compute_expected_hitting_times(transmat: np.ndarray, target: int) -> np.ndarray:
    """
    Compute expected hitting times to target state from all other states.

    Solves: h_i = 1 + sum_j P_ij * h_j  for i != target
            h_target = 0
    """
    n = transmat.shape[0]
    # Set up linear system
    A = np.eye(n) - transmat.copy()
    A[target, :] = 0
    A[target, target] = 1
    b = np.ones(n)
    b[target] = 0

    try:
        h = np.linalg.solve(A, b)
        h[target] = 0
        h = np.maximum(h, 0)  # Numerical stability
    except np.linalg.LinAlgError:
        h = np.full(n, np.inf)
        h[target] = 0

    return h


def analyze_markov_chain(transmat: np.ndarray) -> Dict:
    """Analyze Markov chain properties (for 1st order only)."""
    results = {}

    # Stationary distribution
    pi = compute_stationary_distribution(transmat)
    results["stationary_distribution"] = {CANON_TAGS[i]: float(pi[i]) for i in range(C)}

    # Expected hitting times to final_answer (category 0)
    h_to_final = compute_expected_hitting_times(transmat, target=0)
    results["expected_steps_to_final_answer"] = {CANON_TAGS[i]: float(h_to_final[i]) for i in range(C)}

    return results


# ============================================================
# Main Analysis Runner
# ============================================================

def filter_unknown_labels(sequences: List[Dict], unknown_id: int = 4) -> List[Dict]:
    """Filter out 'unknown' category from label sequences.

    Args:
        sequences: List of sequence dicts with 'labels' field
        unknown_id: The ID of the unknown category (default 4)

    Returns:
        New list of sequences with unknown labels removed
    """
    filtered = []
    for seq in sequences:
        new_labels = [lbl for lbl in seq["labels"] if lbl != unknown_id]
        if new_labels:  # Only keep if there are remaining labels
            new_seq = seq.copy()
            new_seq["labels"] = new_labels
            new_seq["num_sentences"] = len(new_labels)
            filtered.append(new_seq)
    return filtered


def run_full_analysis(prism_dir: str, pt_dir: str, output_dir: str,
                      models: List[str], datasets: List[str], seeds: List[int],
                      ignore_unknown: bool = False):
    """Run full analysis on all models/datasets/seeds."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for model in models:
        for dataset in datasets:
            for seed in seeds:
                key = f"{model}/{dataset}/{seed}"
                print(f"\n{'='*60}")
                print(f"Analyzing: {key}")
                print(f"{'='*60}")

                # Find PT file
                pt_dir_path = os.path.join(pt_dir, model, dataset, str(seed), "runs")
                pt_files = []
                if os.path.isdir(pt_dir_path):
                    for f in os.listdir(pt_dir_path):
                        if f.endswith(".pt") and "_first" not in f:
                            pt_files.append(os.path.join(pt_dir_path, f))

                if not pt_files:
                    print(f"[SKIP] No PT files found in {pt_dir_path}")
                    continue

                # Load data
                records = []
                for pt_file in pt_files:
                    records.extend(load_pt_records(pt_file))

                sequences = extract_label_sequences(records)
                print(f"[INFO] Loaded {len(sequences)} sequences")

                # Filter out unknown labels if requested
                if ignore_unknown:
                    orig_count = len(sequences)
                    sequences = filter_unknown_labels(sequences, unknown_id=4)
                    print(f"[INFO] Filtered unknown: {orig_count} -> {len(sequences)} sequences")

                result = {"key": key, "num_sequences": len(sequences)}

                # Create output subdir
                out_subdir = os.path.join(output_dir, model, dataset, str(seed))
                os.makedirs(out_subdir, exist_ok=True)

                # 1. Basic stats (from empirical counts)
                trans1_counts = count_transitions(sequences, order=1)
                trans1 = normalize_transition_matrix(trans1_counts)
                start1 = np.zeros(C)
                for seq in sequences:
                    if seq["labels"]:
                        start1[seq["labels"][0]] += 1
                start1 = start1 / (start1.sum() + EPS)

                result["basic_stats_order1"] = analyze_basic_stats(start1)
                result["start_end_patterns"] = analyze_start_end_patterns(sequences)

                # 2. Path analysis
                result["path_lengths"] = analyze_path_lengths(sequences)
                result["top_3grams"] = count_ngrams(sequences, n=3).most_common(20)

                # 3. Correct vs Incorrect comparison (1st–5th order)
                result["correct_vs_incorrect"] = analyze_correct_vs_incorrect(sequences, order=1)
                cvi2 = analyze_correct_vs_incorrect(sequences, order=2)
                result["correct_vs_incorrect_2nd"] = {
                    "correct_transition_matrix": cvi2.get("correct_transition_matrix"),
                    "incorrect_transition_matrix": cvi2.get("incorrect_transition_matrix"),
                    "num_correct": cvi2.get("num_correct", 0),
                    "num_incorrect": cvi2.get("num_incorrect", 0),
                }
                # Higher orders (3rd, 4th, 5th): matrices grow as C^N × C (C=5 incl. unknown).
                # These are stored for aggregation and higher-order danger detection.
                for _ord, _suffix in [(3, "3rd"), (4, "4th"), (5, "5th")]:
                    _cvi = analyze_correct_vs_incorrect(sequences, order=_ord)
                    result[f"correct_vs_incorrect_{_suffix}"] = {
                        "correct_transition_matrix": _cvi.get("correct_transition_matrix"),
                        "incorrect_transition_matrix": _cvi.get("incorrect_transition_matrix"),
                        "num_correct": _cvi.get("num_correct", 0),
                        "num_incorrect": _cvi.get("num_incorrect", 0),
                        "order": _ord,
                    }

                # 6. Sample sequences per group
                print("[6] Sampling sequences per group...")
                result["sampled_sequences"] = sample_sequences_per_group(
                    sequences, n_per_group=3, long_threshold=100)

                # 7. Markov chain analysis
                result["markov_chain"] = analyze_markov_chain(trans1)

                all_results[key] = result

                # Save individual results
                result_path = os.path.join(out_subdir, "analysis.json")
                with open(result_path, "w") as f:
                    json.dump(convert_for_json(result), f, indent=2)
                print(f"[SAVED] {result_path}")

    # Save aggregated results
    agg_path = os.path.join(output_dir, "all_results.json")
    with open(agg_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"\n[SAVED] Aggregated results: {agg_path}")

    return all_results


def convert_for_json(obj):
    """Convert numpy arrays and other non-JSON types for serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        # Convert tuple keys to strings
        return {(str(k) if isinstance(k, tuple) else k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_for_json(v) for v in obj]
    else:
        return obj


def print_summary(results: Dict):
    """Print a human-readable summary of analysis results."""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    for key, result in results.items():
        print(f"\n--- {key} ---")
        print(f"Sequences: {result['num_sequences']}")

        if "path_lengths" in result:
            pl = result["path_lengths"]
            print(f"Path lengths: mean={pl['mean']:.1f}, median={pl['median']:.1f}, max={pl['max']}")

        if "correct_vs_incorrect" in result:
            cvi = result["correct_vs_incorrect"]
            if "num_correct" in cvi:
                print(f"Correct: {cvi['num_correct']}, Incorrect: {cvi['num_incorrect']}")
                print(f"Path length - Correct: {cvi['correct_path_length_mean']:.1f}, Incorrect: {cvi['incorrect_path_length_mean']:.1f}")

        if "markov_chain" in result:
            mc = result["markov_chain"]
            print(f"Stationary distribution:")
            for cat, prob in mc["stationary_distribution"].items():
                print(f"  {cat}: {prob:.3f}")



def run_joint_analysis(model_npz: str, pt_files: List[str], output_dir: str,
                       label_key: str = "sentence_labels",
                       ignore_unknown: bool = False) -> Dict:
    """Run analysis on pooled PT files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load and pool records — also track per-seed for std computation
    records = []
    per_seed_records = []
    for p in pt_files:
        r = load_pt_records(p)
        per_seed_records.append(r)
        records.extend(r)
        print(f"  {p}: {len(r)} records")
    print(f"Loaded {len(records)} records total from {len(pt_files)} file(s)")

    sequences = extract_label_sequences(records)
    print(f"[INFO] Loaded {len(sequences)} sequences")

    if ignore_unknown:
        orig_count = len(sequences)
        sequences = filter_unknown_labels(sequences, unknown_id=4)
        print(f"[INFO] Filtered unknown: {orig_count} -> {len(sequences)} sequences")

    key = "joint"
    result = {"key": key, "num_sequences": len(sequences)}

    # Compute per-seed transition matrices
    per_seed_matrices = {}
    _group_names = ["all", "correct", "incorrect", "long_fail", "short_fail"]
    for _group in _group_names:
        _seed_mats = []
        for _sr in per_seed_records:
            _seqs = extract_label_sequences(_sr, label_key=label_key)
            if ignore_unknown:
                _seqs = filter_unknown_labels(_seqs, unknown_id=4)
            _pop = _seqs
            if _group == "all":
                _grp = _pop
            elif _group == "correct":
                _grp, _ = split_by_correctness(_pop)
            elif _group == "incorrect":
                _, _grp = split_by_correctness(_pop)
            elif _group == "long_fail":
                _, _inc = split_by_correctness(_pop)
                _grp, _ = split_incorrect_by_length(_inc, threshold=100)
            else:  # short_fail
                _, _inc = split_by_correctness(_pop)
                _, _grp = split_incorrect_by_length(_inc, threshold=100)
            if _grp:
                _cnt = count_transitions(_grp, order=1)
                _mat = normalize_transition_matrix(_cnt)
                _seed_mats.append(_mat.tolist())
        if _seed_mats:
            per_seed_matrices[_group] = _seed_mats
    result["per_seed_matrices"] = per_seed_matrices

    # 1. Basic stats
    trans1_counts = count_transitions(sequences, order=1)
    trans1 = normalize_transition_matrix(trans1_counts)
    start1 = np.zeros(C)
    for seq in sequences:
        if seq["labels"]:
            start1[seq["labels"][0]] += 1
    start1 = start1 / (start1.sum() + EPS)

    result["basic_stats_order1"] = analyze_basic_stats(start1)
    result["start_end_patterns"] = analyze_start_end_patterns(sequences)

    # 2. Path analysis
    result["path_lengths"] = analyze_path_lengths(sequences)
    result["top_3grams"] = count_ngrams(sequences, n=3).most_common(20)

    # 3. Correct vs Incorrect comparison (1st–5th order)
    result["correct_vs_incorrect"] = analyze_correct_vs_incorrect(sequences, order=1)
    cvi2 = analyze_correct_vs_incorrect(sequences, order=2)
    result["correct_vs_incorrect_2nd"] = {
        "correct_transition_matrix": cvi2.get("correct_transition_matrix"),
        "incorrect_transition_matrix": cvi2.get("incorrect_transition_matrix"),
        "num_correct": cvi2.get("num_correct", 0),
        "num_incorrect": cvi2.get("num_incorrect", 0),
    }
    for _ord, _suffix in [(3, "3rd"), (4, "4th"), (5, "5th")]:
        _cvi = analyze_correct_vs_incorrect(sequences, order=_ord)
        result[f"correct_vs_incorrect_{_suffix}"] = {
            "correct_transition_matrix": _cvi.get("correct_transition_matrix"),
            "incorrect_transition_matrix": _cvi.get("incorrect_transition_matrix"),
            "num_correct": _cvi.get("num_correct", 0),
            "num_incorrect": _cvi.get("num_incorrect", 0),
            "order": _ord,
        }

    # 6. Sample sequences per group
    print("[6] Sampling sequences per group...")
    result["sampled_sequences"] = sample_sequences_per_group(
        sequences, n_per_group=3, long_threshold=100)

    # 7. Markov chain analysis
    result["markov_chain"] = analyze_markov_chain(trans1)

    result_path = os.path.join(output_dir, "analysis.json")
    with open(result_path, "w") as f:
        json.dump(convert_for_json(result), f, indent=2)
    print(f"[SAVED] {result_path}")

    return {key: result}


def main():
    parser = argparse.ArgumentParser(description="Analyze PRISM top-level transitions")

    parser.add_argument("--model_npz", type=str, default=None,
                        help="[Joint mode] Path to a single jointly-trained .npz model")
    parser.add_argument("--pt_files", type=str, nargs="+", default=None,
                        help="[Joint mode] One or more .pt files whose records are pooled")
    parser.add_argument("--prism_dir", type=str, default=None,
                        help="[Legacy] Directory containing PRISM .npz models")
    parser.add_argument("--pt_dir", type=str, default=None,
                        help="[Legacy] Directory containing .pt files with labels")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for analysis results")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["stratos", "openthinker", "qwen", "nemotron"],
                        help="Reasoning models to analyze")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["aime24"],
                        help="Datasets to analyze")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1, 42, 123],
                        help="Seeds to analyze")
    parser.add_argument("--ignore_unknown", action="store_true",
                        help="Ignore 'unknown' category (id=4) in transition analysis")
    args = parser.parse_args()

    if args.model_npz and args.pt_files:
        results = run_joint_analysis(
            args.model_npz, args.pt_files, args.output_dir,
            ignore_unknown=args.ignore_unknown,
        )
        print_summary(results)
        return

    if not args.prism_dir or not args.pt_dir:
        parser.error("Either (--model_npz + --pt_files) or (--prism_dir + --pt_dir) must be provided")

    results = run_full_analysis(
        args.prism_dir, args.pt_dir, args.output_dir,
        args.models, args.datasets, args.seeds,
        ignore_unknown=args.ignore_unknown,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
