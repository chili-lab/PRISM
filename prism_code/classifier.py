#!/usr/bin/env python3
"""
Add sentence labels to existing .pt files from preprocessing_unified.
Uses Qwen2.5-32B-Instruct for classification.
"""

import os
import re
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Default classification model
DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# Tags (same as preprocessing)
TAG_LIST = [
    "final_answer",
    "setup_and_retrieval",
    "analysis_and_computation",
    "uncertainty_and_verification",
]

CLASSIFY_PROMPT = """Classify the sentence into ONE of these 4 tags:

- setup_and_retrieval: restates the problem or recalls known facts from the question
  Example: "The question is to find the value of x such that 2x + 3 = 7."
  Example: "Recall that the sum of angles in a triangle is 180 degrees."

- analysis_and_computation: performs math, logic, or derivation
  Example: "Subtracting 3 from both sides gives 2x = 4."
  Example: "Since a + b = 10 and a = 3, we have b = 7."

- uncertainty_and_verification: expresses doubt or checks results
  Example: "Let me verify this by substituting back into the original equation."
  Example: "Wait, I think I made an error in the previous step."

- final_answer: states the final conclusion
  Example: "Therefore, the answer is 42."
  Example: "The final answer is x = 2."

Output: \\boxed{tag_name}
"""


def build_prompt(tokenizer, user_text: str) -> str:
    """Build classification prompt using standard chat template."""
    messages = [
        {"role": "system", "content": "You are a classification assistant. Answer concisely with only the boxed tag."},
        {"role": "user", "content": user_text}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...}."""
    matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip().lower()
    return ""


def classify_batch(tokenizer, model, items: list, device, max_new_tokens: int = 128) -> list:
    """
    Classify a batch of sentences with their original questions as context.
    Args:
        items: list of (sentence_text, question_text) tuples
    Returns: list of (tag, confidence, gen_txt) tuples
    """
    if not items:
        return []

    # Build prompts for all items
    prompts = []
    for sent_text, question_text in items:
        if question_text:
            user_prompt = (
                f"{CLASSIFY_PROMPT}\n\n"
                f"Original Problem:\n\"{question_text}\"\n\n"
                f"Sentence to classify:\n\"{sent_text}\"\n\n"
                f"Your classification:"
            )
        else:
            user_prompt = (
                f"{CLASSIFY_PROMPT}\n\n"
                f"Sentence to classify:\n\"{sent_text}\"\n\n"
                f"Your classification:"
            )
        prompts.append(build_prompt(tokenizer, user_prompt))
        print(user_prompt)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tokenizer.padding_side = original_padding_side

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode each output
    results = []
    for i in range(len(items)):
        gen_ids = generated_ids[i, input_len:]
        gen_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)

        boxed_content = extract_boxed(gen_txt)

        if boxed_content in TAG_LIST:
            results.append((boxed_content, 1.0, gen_txt))
        elif boxed_content == "unknown":
            results.append(("unknown", 0.5, gen_txt))
        else:
            results.append(("unknown", 0.0, gen_txt))

    return results


def process_file(pt_path: str, tokenizer, model, device,
                 dry_run: bool = False, force_reclassify: bool = False,
                 debug: bool = False, debug_n: int = 5, batch_size: int = 1,
                 test_limit: int = 0):
    """Process a single .pt file and add labels."""
    print(f"\n{'='*60}")
    print(f"Processing: {pt_path}")
    print(f"{'='*60}")

    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        return 0, 0, {}

    records = data.get('records', [])

    total_sentences = 0
    labeled_sentences = 0
    tag_counts = {tag: 0 for tag in TAG_LIST + ["unknown"]}
    confidence_sum = 0.0
    all_debug_candidates = []

    to_classify = []

    for rec_idx, rec in enumerate(records):
        if 'error' in rec:
            continue

        sentences = rec.get('sentences', rec.get('sentences_with_labels', []))
        if not sentences:
            continue

        existing_labels = rec.get('sentence_labels', [])
        question = rec.get('question', '')

        for sent_idx, sent_item in enumerate(sentences):
            if isinstance(sent_item, tuple):
                sent_text = sent_item[0]
                existing_label = sent_item[1] if len(sent_item) > 1 else None
            elif isinstance(sent_item, str):
                sent_text = sent_item
                existing_label = existing_labels[sent_idx] if sent_idx < len(existing_labels) else None
            else:
                continue

            total_sentences += 1

            if not force_reclassify and existing_label and existing_label != "unknown" and existing_label in TAG_LIST:
                continue

            to_classify.append((rec_idx, sent_idx, sent_text, question))

    # Apply test limit
    if test_limit > 0 and len(to_classify) > test_limit:
        print(f"[TEST MODE] Limiting to first {test_limit} sentences")
        to_classify = to_classify[:test_limit]

    print(f"[INFO] Total sentences: {total_sentences}, to classify: {len(to_classify)}")

    # Classify in batches
    classification_results = {}

    for batch_start in tqdm(range(0, len(to_classify), batch_size), desc="Batches"):
        batch = to_classify[batch_start:batch_start + batch_size]
        batch_items = [(item[2], item[3]) for item in batch]

        results = classify_batch(tokenizer, model, batch_items, device)

        for (rec_idx, sent_idx, sent_text, question), result in zip(batch, results):
            classification_results[(rec_idx, sent_idx)] = result

            if debug:
                label, confidence, gen_txt = result
                all_debug_candidates.append({
                    'rec_idx': rec_idx,
                    'sent_idx': sent_idx,
                    'sentence': sent_text[:200] + ('...' if len(sent_text) > 200 else ''),
                    'question': question[:100] + ('...' if len(question) > 100 else ''),
                    'gen_txt': gen_txt,
                    'boxed': extract_boxed(gen_txt),
                    'label': label,
                    'confidence': confidence
                })

    for rec_idx, rec in enumerate(records):
        if 'error' in rec:
            continue

        sentences = rec.get('sentences', rec.get('sentences_with_labels', []))
        if not sentences:
            continue

        existing_labels = rec.get('sentence_labels', [])
        existing_confidences = rec.get('sentence_confidences', [])

        new_sentences = []
        new_labels = []
        new_confidences = []

        for sent_idx, sent_item in enumerate(sentences):
            if isinstance(sent_item, tuple):
                sent_text = sent_item[0]
                existing_label = sent_item[1] if len(sent_item) > 1 else None
            elif isinstance(sent_item, str):
                sent_text = sent_item
                existing_label = existing_labels[sent_idx] if sent_idx < len(existing_labels) else None
            else:
                continue

            existing_conf = existing_confidences[sent_idx] if sent_idx < len(existing_confidences) else None

            new_sentences.append(sent_text)

            if (rec_idx, sent_idx) in classification_results:
                label, confidence, _ = classification_results[(rec_idx, sent_idx)]
                new_labels.append(label)
                new_confidences.append(confidence)
                tag_counts[label] += 1
                labeled_sentences += 1
                confidence_sum += confidence
            else:
                new_labels.append(existing_label)
                new_confidences.append(existing_conf if existing_conf is not None else 1.0)
                if existing_label:
                    tag_counts[existing_label] += 1
                    labeled_sentences += 1
                    confidence_sum += existing_conf if existing_conf is not None else 1.0

        rec['sentences'] = new_sentences
        rec['sentence_labels'] = new_labels
        rec['sentence_confidences'] = new_confidences

    # Print debug samples
    if debug and all_debug_candidates:
        if len(all_debug_candidates) <= debug_n:
            debug_samples = all_debug_candidates
        else:
            step = len(all_debug_candidates) / debug_n
            indices = [int(i * step) for i in range(debug_n)]
            debug_samples = [all_debug_candidates[i] for i in indices]

        print(f"\n[DEBUG] Showing {len(debug_samples)} classification samples (sampled from {len(all_debug_candidates)} total):")
        for i, s in enumerate(debug_samples):
            print(f"\n--- Sample {i+1} (record {s['rec_idx']}, sent {s['sent_idx']}) ---")
            print(f"  Question: {s['question']}")
            print(f"  Sentence: {s['sentence']}")
            print(f"  Model output: {s['gen_txt']}")
            print(f"  Extracted boxed: {s['boxed']}")
            print(f"  Final label: {s['label']} (conf={s['confidence']:.1f})")

    # Print stats
    avg_confidence = confidence_sum / max(1, labeled_sentences)
    print(f"\n[STATS] Total sentences: {total_sentences}")
    print(f"[STATS] Labeled: {labeled_sentences}")
    print(f"[STATS] Avg confidence: {avg_confidence:.2%}")
    print(f"[STATS] Tag distribution:")
    for tag, count in tag_counts.items():
        if count > 0:
            pct = count / max(1, labeled_sentences) * 100
            print(f"  {tag}: {count} ({pct:.1f}%)")

    if dry_run:
        print(f"\n[DRY-RUN] Would save to: {pt_path}")
    else:
        torch.save(data, pt_path)
        print(f"\n[SAVED] {pt_path}")

    return total_sentences, labeled_sentences, tag_counts


def main():
    parser = argparse.ArgumentParser(description="Add labels to existing .pt files using Qwen2.5-32B-Instruct")
    parser.add_argument("pt_files", nargs='+', help="Path(s) to .pt file(s)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't save, just show what would happen")
    parser.add_argument("--force-reclassify", action="store_true",
                        help="Reclassify all sentences, even those with existing labels")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info showing how model classifies sentences")
    parser.add_argument("--debug-n", type=int, default=5,
                        help="Number of samples to show in debug mode")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for classification (default: 4)")
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: only classify first N sentences (0 = disabled)")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model_path = args.model
    print(f"[INFO] Loading model: {model_path}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval()

    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Batch size: {args.batch_size}")
    if args.test > 0:
        print(f"[INFO] TEST MODE: only classifying first {args.test} sentences per file")

    # Process files
    total_all = 0
    labeled_all = 0

    for pt_path in args.pt_files:
        if not os.path.exists(pt_path):
            print(f"[WARN] File not found: {pt_path}")
            continue

        try:
            total, labeled, _ = process_file(
                pt_path, tokenizer, model, device,
                args.dry_run, args.force_reclassify,
                args.debug, args.debug_n, args.batch_size, args.test
            )
            total_all += total
            labeled_all += labeled
        except Exception as e:
            print(f"[ERROR] Failed to process {pt_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(args.pt_files)}")
    print(f"Total sentences: {total_all}")
    print(f"Labeled sentences: {labeled_all}")


if __name__ == "__main__":
    main()
