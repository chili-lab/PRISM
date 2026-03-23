#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, math, difflib, random, argparse, datetime, signal, traceback, faulthandler
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT_DIR = "runs"
RUN_NAME = "unified_run"

def _log_path():
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(os.path.abspath(OUT_DIR), f"{RUN_NAME}.log")

def _log_exc(where, extra=""):
    try:
        with open(_log_path(), "a") as f:
            f.write(f"\n[{datetime.datetime.now()}] EXCEPTION @ {where}\n")
            if extra: f.write(extra + "\n")
            f.write(traceback.format_exc() + "\n")
            f.flush()
    except Exception:
        pass

def _install_signal_dumps():
    logf = open(_log_path(), "a", buffering=1)
    faulthandler.enable(file=logf)
    def _dump(signum, frame):
        logf.write(f"\n[{datetime.datetime.now()}] Got signal {signum}, dumping stacks...\n")
        faulthandler.dump_traceback(file=logf, all_threads=True)
        logf.flush()
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, _dump)
    signal.signal(signal.SIGINT, _dump)

# ==========================
#     DEFAULT MODELS
# ==========================
DEFAULT_MODEL_MAP = {
    "nemotron": "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
    "qwen": "Qwen/Qwen3-1.7B",
    "r1": "bespokelabs/Bespoke-Stratos-7B",
}

# ==========================
#   PROMPTS (dataset-specific)
# ==========================

PROMPT_GPQA = (
    "You are answering a multiple-choice question.\n"
    "Options are labeled A, B, C, and D.\n"
    "Think step-by-step and show your reasoning.\n"
    "At the very end, output ONE line exactly in this format:\n"
    "Final Answer: \\boxed{A}\n"
    "where the letter is A, B, C, or D.\n"
)

# UNIFIED PROMPT for TIGER, AIME, MATH-500
PROMPT_STANDARD = (
    "Answer the following question step-by-step. "
    "At the very end, output exactly one line formatted as:\n"
    "Final Answer: \\boxed{...}\n"
)

LETTERS = ["A", "B", "C", "D"]

# ==========================
#   DEVICE SETUP
# ==========================
def setup_device():
    """
    Setup single GPU or CPU device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    return device

# ==========================
#   DETERMINISM HELPERS
# ==========================
def set_global_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reseed_for_sample(base_seed: int, sample_idx: int):
    """
    Reseed before each sample generation for full reproducibility.
    Uses sample_idx to ensure different samples get different seeds.
    """
    s = (base_seed if base_seed is not None else 0)
    s = (s * 1_000_003 + sample_idx) % (2**31 - 1)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    return s

# ==========================
#  TOKEN-LEVEL SEGMENTATION
# ==========================
def segment_by_newlines_2plus(token_ids, tokenizer):
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
    n = len(ids)
    if n == 0:
        return []

    segments = []
    start_tok = 0
    consec_nl = 0

    for i in range(n):
        try:
            piece = tokenizer.decode([ids[i]], skip_special_tokens=True)
        except Exception:
            piece = ""

        piece = piece.replace("\r\n", "\n")

        for ch in piece:
            if ch == "\n":
                consec_nl += 1
            else:
                if consec_nl >= 2:
                    if i > start_tok:
                        segments.append((start_tok, i))
                    start_tok = i
                consec_nl = 0

    if start_tok < n:
        segments.append((start_tok, n))

    return segments if segments else [(0, n)]

def decode_token_span(token_ids: torch.Tensor, start: int, end: int, tokenizer) -> str:
    """Safely decode a span of tokens."""
    if start >= end or start < 0:
        return ""
    total_len = len(token_ids) if isinstance(token_ids, torch.Tensor) else len(list(token_ids))
    if end > total_len:
        end = total_len
    if start >= end:
        return ""
    span_ids = token_ids[start:end]
    if isinstance(span_ids, torch.Tensor):
        span_ids = span_ids.tolist()
    try:
        text = tokenizer.decode(span_ids, skip_special_tokens=True)
        return text
    except:
        return ""

# ==========================
#   CHAT TEMPLATE HELPERS
# ==========================
def build_prompt_nemotron(tokenizer, user_text: str) -> str:
    """
    Build chat prompt using Nemotron's chat template.
    Uses system_text="detailed thinking on" for generation.
    """
    messages = []
    messages.append({"role": "system", "content": "detailed thinking on"})
    messages.append({"role": "user", "content": user_text})

    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise RuntimeError(
            f"Tokenizer {tokenizer.name_or_path} does not have a chat_template! "
            f"This is required for Nemotron models."
        )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_prompt_qwen(tokenizer, user_text: str) -> str:
    """
    Build chat prompt using Qwen chat template.
    Uses enable_thinking=True for generation tasks.
    """
    messages = [{"role": "user", "content": user_text}]

    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise RuntimeError(
            f"Tokenizer {tokenizer.name_or_path} does not have a chat_template! "
            f"This is required for Qwen models."
        )

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Qwen-specific for generation
        )
        return text
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply chat template for {tokenizer.name_or_path}: {e}"
        )

def build_prompt_r1(tokenizer, user_text: str) -> str:
    """
    Build chat prompt for R1/Stratos models.
    Uses generic system prompt.
    """
    messages = []
    system_prompt = "You are a reasoning assistant."
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise RuntimeError(
            f"Tokenizer {tokenizer.name_or_path} does not have a chat_template! "
            f"This is required for proper formatting."
        )

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply chat template for {tokenizer.name_or_path}: {e}"
        )

def build_prompt(tokenizer, user_text: str, model_type: str) -> str:
    """
    Unified prompt builder that dispatches to appropriate model-specific function.
    """
    if model_type == "nemotron":
        return build_prompt_nemotron(tokenizer, user_text)
    elif model_type == "qwen":
        return build_prompt_qwen(tokenizer, user_text)
    elif model_type == "r1":
        return build_prompt_r1(tokenizer, user_text)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ==========================
#   CHECKPOINT HELPERS
# ==========================
def load_checkpoint(output_path: str, expected_seed: int) -> Tuple[List[Dict], Set[int]]:
    """
    Load existing checkpoint and return records + set of completed sample indices.
    Validates that the seed matches to ensure reproducibility.
    """
    if not os.path.exists(output_path):
        return [], set()

    try:
        checkpoint = torch.load(output_path, map_location="cpu")

        # Validate seed matches
        saved_seed = checkpoint.get("seed_info", {}).get("seed")
        if saved_seed is not None and saved_seed != expected_seed:
            print(f"[WARNING] Checkpoint seed ({saved_seed}) differs from current seed ({expected_seed})!")
            print(f"[WARNING] This may cause inconsistent results. Consider using --seed {saved_seed}")
            raise ValueError(f"Seed mismatch: checkpoint has {saved_seed}, but running with {expected_seed}")

        records = checkpoint.get("records", [])

        # Extract completed sample indices (excluding error records)
        completed_indices = set()
        for rec in records:
            if "sample_idx" in rec and "error" not in rec:
                completed_indices.add(rec["sample_idx"])

        print(f"[RESUME] Loaded checkpoint with {len(records)} records")
        print(f"[RESUME] Found {len(completed_indices)} successfully completed samples")

        return records, completed_indices

    except Exception as e:
        print(f"[WARNING] Failed to load checkpoint: {e}")
        print(f"[WARNING] Starting from scratch")
        return [], set()

def save_checkpoint(output_path: str, records: List[Dict], args, is_final: bool = False):
    """
    Save current progress to checkpoint file.
    """
    save_data = {
        "records": records,
        "config": vars(args),
        "seed_info": {
            "seed": args.seed,
        },
        "is_complete": is_final,
        "saved_at": datetime.datetime.now().isoformat(),
    }

    # Save to temp file first, then rename (atomic operation)
    temp_path = output_path + ".tmp"
    torch.save(save_data, temp_path)
    os.replace(temp_path, output_path)

    if not is_final:
        print(f"[CHECKPOINT] Saved {len(records)} records to {output_path}")

# ==========================
#          MAIN
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
        choices=["webinstruct", "aime24", "math500", "gpqa_diamond"],
        required=True)
    parser.add_argument("--model_type",
        choices=["nemotron", "qwen", "r1"],
        default="r1",
        help="Model architecture type: nemotron, qwen, or r1")
    parser.add_argument("--gen_model", default=None,
        help="Generation model path (overrides default for model_type)")
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None,
        help="Top-k sampling (Qwen-specific, optional)")
    parser.add_argument("--min_p", type=float, default=None,
        help="Min-p sampling (Qwen-specific, optional)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_dataset", action="store_true",
                    help="Shuffle dataset order deterministically with --seed.")
    parser.add_argument("--tiger_answer_types", nargs="*",
                       default=["Float","Multiple Choice","Integer","Percentage"])
    parser.add_argument("--tiger_difficulties", nargs="*",
                       default=["Senior High School","Junior High School","Primary School"])
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Enable trust_remote_code for model loading (required for Qwen)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode to print first sample result")
    parser.add_argument("--no_compile", action="store_true",
                       help="Disable torch.compile() optimization")
    # New resume arguments
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing checkpoint if available")
    parser.add_argument("--save_every", type=int, default=100,
                       help="Save checkpoint every N samples (default: 100)")
    args = parser.parse_args()

    global OUT_DIR, RUN_NAME
    OUT_DIR = args.out_dir
    RUN_NAME = args.run_name or f"{args.task}_{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ===== Setup device and logging =====
    device = setup_device()
    _install_signal_dumps()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== Determine output path =====
    output_path = os.path.join(OUT_DIR, f"{RUN_NAME}.pt")

    # ===== Load checkpoint if resuming =====
    completed_indices: Set[int] = set()
    records: List[Dict] = []

    if args.resume:
        records, completed_indices = load_checkpoint(output_path, args.seed)
        if completed_indices:
            print(f"[RESUME] Will skip {len(completed_indices)} already-completed samples")

    # ===== Seeds / determinism (GUARANTEED REPRODUCIBILITY) =====
    set_global_seed(args.seed)
    gpqa_seed = args.seed

    print(f"[INFO] Full reproducibility enabled with seed={args.seed}")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Resume mode: {args.resume}")
    print(f"[INFO] Save every: {args.save_every} samples")

    # ===== Perf toggles =====
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ===== Determine model path =====
    gen_model_path = args.gen_model or DEFAULT_MODEL_MAP[args.model_type]

    # ===== Load model =====
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    DTYPE = dtype_map[args.dtype]

    print(f"[INFO] Loading generation model: {gen_model_path} (type: {args.model_type})")

    # Load tokenizer with optional trust_remote_code
    gen_tok = AutoTokenizer.from_pretrained(
        gen_model_path,
        trust_remote_code=args.trust_remote_code
    )
    if gen_tok.pad_token is None and gen_tok.eos_token is not None:
        gen_tok.pad_token = gen_tok.eos_token

    # Load model with optional trust_remote_code and quantization
    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto"
    }
    load_kwargs["torch_dtype"] = DTYPE

    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_path,
        **load_kwargs
    ).eval()

    if not hasattr(gen_tok, 'chat_template') or gen_tok.chat_template is None:
        raise RuntimeError(
            f"Generation model {gen_model_path} does not have a chat_template! "
            f"This is required for proper formatting."
        )

    print(f"[INFO] Generation model loaded successfully with chat template")

    if not args.no_compile:
        try:
            print("[INFO] Compiling model with torch.compile() for faster inference...")
            gen_model = torch.compile(gen_model, mode="reduce-overhead")
            print("[INFO] Model compilation successful")
        except Exception as e:
            print(f"[WARNING] torch.compile() failed: {e}")
            print("[WARNING] Continuing without compilation (slower but still works)")
    else:
        print("[INFO] Skipping torch.compile() as requested")

    GEN_PAD_ID = gen_tok.pad_token_id if gen_tok.pad_token_id is not None else gen_tok.eos_token_id

    first_sample_processed = False
    samples_since_save = 0

    # ===== Single-sample generate+collect =====
    def generate_and_collect(prompt: str, question: str, gold_answer: str,
                            label_note: str, is_mc: bool = False,
                            extra_mc=None, sample_idx: int = 0):
        nonlocal first_sample_processed, samples_since_save

        def build_last_token_hidden_3d(hidden_states_out):
            per_step = []
            for step_idx, per_layer in enumerate(hidden_states_out):
                step_last = []
                for h in per_layer:
                    if h.dim() == 3:
                        v = h[:, -1, :].squeeze(0)  # [D]
                    elif h.dim() == 2:
                        v = h[-1, :]                 # [D]
                    else:
                        raise RuntimeError(f"Unexpected hidden state shape {tuple(h.shape)} at step {step_idx}")
                    step_last.append(v)
                step_tensor = torch.stack(step_last, dim=0)  # [L+1, D]
                per_step.append(step_tensor)
            # [T_gen, L+1, D] -> [L+1, T_gen, D]
            hs_3d = torch.stack(per_step, dim=0).transpose(0, 1).contiguous()
            return hs_3d

        reseed_for_sample(args.seed, sample_idx)

        # Build prompt using model-specific template
        formatted_prompt = build_prompt(gen_tok, prompt, args.model_type)

        gen_inputs = gen_tok(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
        assert gen_inputs["input_ids"].shape[0] == 1, f"Batch size must be 1, got {gen_inputs['input_ids'].shape[0]}"

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "eos_token_id": gen_tok.eos_token_id,
            "pad_token_id": GEN_PAD_ID,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }

        # Add optional parameters for Qwen
        if args.top_k is not None:
            gen_kwargs["top_k"] = args.top_k
        if args.min_p is not None:
            gen_kwargs["min_p"] = args.min_p

        with torch.no_grad():
            out = gen_model.generate(**gen_inputs, **gen_kwargs)

        seq = out.sequences[0]                      
        inp_len = gen_inputs["input_ids"].shape[1]
        gen_only_ids = seq[inp_len:]                
        assert gen_only_ids.dim() == 1, f"gen_only_ids should be 1D, got shape {gen_only_ids.shape}"

        generated_text = gen_tok.decode(gen_only_ids, skip_special_tokens=True)

        hs_out = out.hidden_states
        hidden_states = build_last_token_hidden_3d(hs_out).cpu()  

        assert hidden_states.shape[1] == len(gen_only_ids), \
            f"hidden_states tokens {hidden_states.shape[1]} != gen_only_ids {len(gen_only_ids)}"
        if isinstance(hs_out, (list, tuple)):
            assert len(hs_out) == len(gen_only_ids), f"steps {len(hs_out)} != gen_tokens {len(gen_only_ids)}"

        token_segments = segment_by_newlines_2plus(gen_only_ids, gen_tok)
        if not token_segments:
            token_segments = [(0, len(gen_only_ids))]

        step_hidden_states = []      # List[Tensor [L+1, D]]
        sentences = []               # List[str]

        for (t0, t1) in token_segments:
            if t1 <= t0 or t0 < 0 or t1 > hidden_states.shape[1]:
                continue

            segment_text = decode_token_span(gen_only_ids, t0, t1, gen_tok)
            if not segment_text or not segment_text.strip():
                continue

            step_hidden = hidden_states[:, t0, :].clone()  # [L+1, D]
            step_hidden_states.append(step_hidden.cpu())
            sentences.append(segment_text)

        del hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(step_hidden_states) != len(sentences):
            raise ValueError(
                f"Step/sentence mismatch: len(steps)={len(step_hidden_states)} "
                f"len(sentences)={len(sentences)} "
                f"sample_idx={sample_idx}"
            )

        rec = {
            "sample_idx": sample_idx,
            "label_note": label_note,
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "generated_text": generated_text,
            "question": question,
            "ground_truth_answer": gold_answer,
            "sentences": sentences,
            "step_hidden_states": step_hidden_states,
            "gen_token_count": int(gen_only_ids.numel()),
            "num_steps": len(step_hidden_states),
            "model_type": args.model_type,
        }

        if is_mc and extra_mc:
            if "options_shuffled" in extra_mc:
                rec["options_shuffled"] = extra_mc["options_shuffled"]

        records.append(rec)
        samples_since_save += 1

        # Incremental checkpoint save
        if samples_since_save >= args.save_every:
            save_checkpoint(output_path, records, args, is_final=False)
            samples_since_save = 0

        # Debug: print first sample result
        if args.debug and not first_sample_processed:
            print("\n" + "="*80)
            print("DEBUG: First Sample Result")
            print("="*80)
            print(f"Sample Index: {sample_idx}")
            print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")
            print(f"\nGenerated Text ({len(generated_text)} chars):")
            print("-" * 80)
            print(generated_text[:1000] + "..." if len(generated_text) > 1000 else generated_text)
            print("-" * 80)
            print(f"\nNumber of sentences/segments: {len(sentences)}")
            print(f"Number of step hidden states: {len(step_hidden_states)}")
            if step_hidden_states:
                print(f"Hidden state shape: {step_hidden_states[0].shape}")
            print(f"Total generated tokens: {gen_only_ids.numel()}")
            print("\nFirst 3 sentences:")
            for i, sent in enumerate(sentences[:3]):
                print(f"  [{i}] {sent[:100]}..." if len(sent) > 100 else f"  [{i}] {sent}")
            print("="*80 + "\n")
            first_sample_processed = True

    # ===== Load + run per task =====
    try:
        task = args.task

        if task == "webinstruct":
            ds_full = load_dataset("TIGER-Lab/WebInstruct-verified", split="test")
            ats = set(args.tiger_answer_types)
            diffs = set(args.tiger_difficulties)
            ANSWER_TYPE_KEY = "answer_type"

            if ANSWER_TYPE_KEY in ds_full.column_names and "difficulty" in ds_full.column_names:
                ds_full = ds_full.filter(lambda x: x[ANSWER_TYPE_KEY] in ats and x["difficulty"] in diffs)
            elif ANSWER_TYPE_KEY in ds_full.column_names:
                ds_full = ds_full.filter(lambda x: x[ANSWER_TYPE_KEY] in ats)
            elif "difficulty" in ds_full.column_names:
                ds_full = ds_full.filter(lambda x: x["difficulty"] in diffs)

            total = len(ds_full)
            take = min(args.num_samples if args.num_samples is not None else total, total)

            print(f"[INFO] Tiger filtered dataset: {total} samples, taking {take}")

            ds = ds_full.select(range(take))

            if args.shuffle_dataset:
                ds = ds.shuffle(seed=args.seed)

            # Count how many to process
            to_process = sum(1 for idx in range(len(ds)) if idx not in completed_indices)
            print(f"[INFO] Total samples: {len(ds)}, Already completed: {len(completed_indices)}, To process: {to_process}")

            for idx, sample in enumerate(tqdm(ds, desc="Processing")):
                # Skip already completed samples
                if idx in completed_indices:
                    continue

                try:
                    question = str(sample["question"]).strip()
                    gold_answer = str(sample["answer"]).strip()
                    prompt = f"{PROMPT_STANDARD}\n{question}\n"

                    generate_and_collect(
                        prompt, question, gold_answer,
                        label_note="webinstruct_filtered",
                        is_mc=False,
                        sample_idx=idx
                    )
                except Exception as e:
                    _log_exc("webinstruct-per-sample", f"idx={idx}")
                    records.append({
                        "error": f"{type(e).__name__}: {e}",
                        "sample_idx": idx
                    })

        elif task == "aime24":
            ds_full = load_dataset("HuggingFaceH4/aime_2024", split="train")
            total = len(ds_full)
            take = min(args.num_samples if args.num_samples is not None else total, total)

            print(f"[INFO] AIME24 dataset: {total} samples, taking {take}")

            ds = ds_full.select(range(take))

            if args.shuffle_dataset:
                ds = ds.shuffle(seed=args.seed)

            # Count how many to process
            to_process = sum(1 for idx in range(len(ds)) if idx not in completed_indices)
            print(f"[INFO] Total samples: {len(ds)}, Already completed: {len(completed_indices)}, To process: {to_process}")

            for idx, sample in enumerate(tqdm(ds, desc="Processing")):
                # Skip already completed samples
                if idx in completed_indices:
                    continue

                try:
                    question = str(sample["problem"]).strip()
                    gold_answer = str(sample["answer"]).strip()
                    prompt = f"{PROMPT_STANDARD}\n{question}\n"

                    generate_and_collect(
                        prompt, question, gold_answer,
                        label_note="aime24",
                        is_mc=False,
                        sample_idx=idx
                    )
                except Exception as e:
                    _log_exc("aime24-per-sample", f"idx={idx}")
                    records.append({
                        "error": f"{type(e).__name__}: {e}",
                        "sample_idx": idx
                    })

        elif task == "math500":
            ds_full = load_dataset("HuggingFaceH4/MATH-500", split="test")
            total = len(ds_full)
            take = min(args.num_samples if args.num_samples is not None else total, total)

            print(f"[INFO] MATH-500 dataset: {total} samples, taking {take}")

            ds = ds_full.select(range(take))

            if args.shuffle_dataset:
                ds = ds.shuffle(seed=args.seed)

            # Count how many to process
            to_process = sum(1 for idx in range(len(ds)) if idx not in completed_indices)
            print(f"[INFO] Total samples: {len(ds)}, Already completed: {len(completed_indices)}, To process: {to_process}")

            for idx, sample in enumerate(tqdm(ds, desc="Processing")):
                # Skip already completed samples
                if idx in completed_indices:
                    continue

                try:
                    question = str(sample["problem"]).strip()
                    gold_answer = str(sample["answer"]).strip()
                    prompt = f"{PROMPT_STANDARD}\n{question}\n"

                    generate_and_collect(
                        prompt, question, gold_answer,
                        label_note="math500",
                        is_mc=False,
                        sample_idx=idx
                    )
                except Exception as e:
                    _log_exc("math500-per-sample", f"idx={idx}")
                    records.append({
                        "error": f"{type(e).__name__}: {e}",
                        "sample_idx": idx
                    })

        elif task == "gpqa_diamond":
            ds_full = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            total = len(ds_full)
            take = min(args.num_samples if args.num_samples is not None else total, total)

            print(f"[INFO] GPQA Diamond dataset: {total} samples, taking {take}")

            ds = ds_full.select(range(take))

            if args.shuffle_dataset:
                ds = ds.shuffle(seed=args.seed)

            # Count how many to process
            to_process = sum(1 for idx in range(len(ds)) if idx not in completed_indices)
            print(f"[INFO] Total samples: {len(ds)}, Already completed: {len(completed_indices)}, To process: {to_process}")

            for idx, sample in enumerate(tqdm(ds, desc="Processing")):
                # Skip already completed samples
                if idx in completed_indices:
                    continue

                try:
                    rng = random.Random((gpqa_seed if gpqa_seed is not None else 0) + idx)

                    needed = ["Question","Correct Answer","Incorrect Answer 1",
                             "Incorrect Answer 2","Incorrect Answer 3"]
                    for k in needed:
                        if k not in sample:
                            raise ValueError(f"Missing column '{k}'")

                    q = str(sample["Question"]).strip()
                    opts = [
                        str(sample["Correct Answer"]).strip(),
                        str(sample["Incorrect Answer 1"]).strip(),
                        str(sample["Incorrect Answer 2"]).strip(),
                        str(sample["Incorrect Answer 3"]).strip(),
                    ]

                    idxs = [0, 1, 2, 3]
                    rng.shuffle(idxs)
                    shuf = [opts[i] for i in idxs]
                    correct_idx = idxs.index(0)
                    correct_letter = LETTERS[correct_idx]

                    options_block = "\n".join(f"{LETTERS[i]}. {shuf[i]}" for i in range(4))
                    prompt = f"{PROMPT_GPQA}\n{q}\n\n{options_block}\n"

                    extra = {
                        "options_shuffled": {LETTERS[i]: shuf[i] for i in range(4)},
                        "correct_letter": correct_letter
                    }

                    generate_and_collect(
                        prompt, q, correct_letter,
                        label_note="gpqa_diamond",
                        is_mc=True,
                        extra_mc=extra,
                        sample_idx=idx
                    )
                except Exception as e:
                    _log_exc("gpqa-per-sample", f"idx={idx}")
                    records.append({
                        "error": f"{type(e).__name__}: {e}",
                        "sample_idx": idx
                    })

        # ===== Final save =====
        save_checkpoint(output_path, records, args, is_final=True)

        # Count successful samples
        successful = sum(1 for r in records if "error" not in r)
        print(f"\n{'='*60}")
        print(f"[INFO] Completed {successful} samples successfully ({len(records)} total records)")
        print(f"[INFO] Results saved to: {output_path}")
        print(f"{'='*60}\n")

    except Exception as e:
        # Emergency save on crash
        print(f"\n[ERROR] Fatal error: {e}")
        print(f"[ERROR] Attempting emergency checkpoint save...")
        try:
            save_checkpoint(output_path, records, args, is_final=False)
            print(f"[ERROR] Emergency save successful: {len(records)} records saved")
        except Exception as save_err:
            print(f"[ERROR] Emergency save failed: {save_err}")

        _log_exc("main-fatal", f"Fatal error: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    main()
