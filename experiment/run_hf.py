#!/usr/bin/env python3
"""
EuraGovExam HuggingFace Evaluation Script
==========================================

Evaluate VLMs on the EuraGovExam benchmark using locally downloaded
HuggingFace models (Qwen2-VL, LLaVA, LLaVA-NeXT, and more).

Usage:
    python run_hf.py --model Qwen/Qwen2-VL-7B-Instruct
    python run_hf.py --model llava-hf/llava-1.5-7b-hf --dtype float16
    python run_hf.py --model Qwen/Qwen2-VL-7B-Instruct --nation japan --sample-size 100
"""

import re
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────

DATASET_NAME = "EuraGovExam/EuraGovExam"
RANDOM_BASELINE = 23.7  # Weighted average of 4-choice / 5-choice random guessing

PROMPT = (
    "You are solving a multiple-choice exam question shown in the image.\n"
    "Carefully read the question and all answer options.\n"
    "At the very end, provide the final answer in exactly this format:\n"
    "The answer is X. (For example: The answer is B.)"
)

NATION_MAP = {
    "korea": "South Korea",
    "japan": "Japan",
    "taiwan": "Taiwan",
    "india": "India",
    "eu": "EU",
}

# ─── Shared helpers ───────────────────────────────────────────────────────────


def extract_answer(response_text: str) -> str:
    """Extract answer (A-E) from model response using a 4-level regex cascade."""
    if not response_text:
        return "INVALID"

    # Pattern 1: "The answer is X."
    match = re.search(r"[Tt]he answer is\s*[:\s]*([A-Ea-e])[\.\s]?", response_text)
    if match:
        return match.group(1).upper()

    # Pattern 2: "Answer: X"
    match = re.search(r"[Aa]nswer\s*[:\s]+([A-Ea-e])[\.\s]?", response_text)
    if match:
        return match.group(1).upper()

    # Pattern 3: Standalone letter at end of line
    match = re.search(r"(?:^|\s)([A-E])\.?\s*$", response_text.strip(), re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 4: Last capital letter (fallback)
    letters = re.findall(r"\b([A-E])\b", response_text.upper())
    if letters:
        return letters[-1]

    return "INVALID"


def load_and_filter_dataset(
    nation: Optional[str] = None,
    domain: Optional[str] = None,
    split: str = "train",
) -> List[Tuple[int, dict]]:
    """Load HuggingFace dataset and optionally filter by nation / domain."""
    from datasets import load_dataset

    print(f"Loading dataset: {DATASET_NAME} (split={split})...")
    dataset = load_dataset(DATASET_NAME, split=split)
    print(f"  Dataset loaded: {len(dataset)} items")

    filtered = []
    for idx, item in enumerate(dataset):
        if nation is not None and item["nation"] != NATION_MAP[nation]:
            continue
        if domain is not None and item["task"] != domain:
            continue
        filtered.append((idx, item))

    if not filtered:
        print(f"  WARNING: No samples match filters (nation={nation}, domain={domain})")
    else:
        print(f"  Filtered to {len(filtered)} items")

    return filtered


def calculate_metrics(results: List[dict]) -> dict:
    """Compute overall / by-nation / by-domain accuracy."""
    if not results:
        return {"overall_accuracy": 0.0, "random_baseline": RANDOM_BASELINE,
                "correct_count": 0, "total_count": 0, "by_nation": {}, "by_domain": {}}

    correct_count = sum(1 for r in results if r["is_correct"])
    overall_accuracy = (correct_count / len(results)) * 100

    by_nation = defaultdict(lambda: {"correct": 0, "total": 0})
    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        by_nation[r["nation"]]["total"] += 1
        by_domain[r["task"]]["total"] += 1
        if r["is_correct"]:
            by_nation[r["nation"]]["correct"] += 1
            by_domain[r["task"]]["correct"] += 1

    def _pct(d):
        return {k: {"accuracy": round(v["correct"] / v["total"] * 100, 2), **v}
                for k, v in d.items()}

    return {
        "overall_accuracy": round(overall_accuracy, 2),
        "random_baseline": RANDOM_BASELINE,
        "correct_count": correct_count,
        "total_count": len(results),
        "by_nation": _pct(by_nation),
        "by_domain": _pct(by_domain),
    }


def save_results(results: List[dict], metrics: dict, args, output_dir: Path) -> Path:
    """Save evaluation results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.replace("/", "_")
    filename = f"eval_hf_{model_short}_{ts}.json"
    output_file = output_dir / filename

    data = {
        "metadata": {
            "model": args.model,
            "provider": "huggingface",
            "split": args.split,
            "filters": {"nation": args.nation, "domain": args.domain},
            "sample_size": metrics["total_count"],
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "timestamp": ts,
        },
        "metrics": metrics,
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_file


def print_summary(metrics: dict, args):
    """Pretty-print evaluation summary."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model:   {args.model}")
    print(f"Device:  {args.device}")
    print(f"Dtype:   {args.dtype}")
    print(f"Filters: nation={args.nation or 'all'}, domain={args.domain or 'all'}")
    print(f"Samples: {metrics['total_count']}")
    print("-" * 70)
    print(f"Overall Accuracy: {metrics['overall_accuracy']}%")
    print(f"Random Baseline:  {metrics['random_baseline']}%")
    print(f"Correct: {metrics['correct_count']}/{metrics['total_count']}")

    if metrics["by_nation"]:
        print("\nBy Nation:")
        for nation, s in sorted(metrics["by_nation"].items()):
            print(f"  {nation:15s}: {s['accuracy']:5.1f}% ({s['correct']}/{s['total']})")

    if metrics["by_domain"]:
        print("\nBy Domain:")
        for domain, s in sorted(metrics["by_domain"].items()):
            print(f"  {domain:20s}: {s['accuracy']:5.1f}% ({s['correct']}/{s['total']})")

    print("=" * 70)


# ─── Model loading ────────────────────────────────────────────────────────────


def detect_model_type(model_id: str) -> str:
    """Detect VLM architecture from model ID string."""
    lower = model_id.lower()
    if "qwen2-vl" in lower or "qwen2.5-vl" in lower:
        return "qwen2_vl"
    if "llava-next" in lower or "llava-v1.6" in lower:
        return "llava_next"
    if "llava" in lower:
        return "llava"
    return "auto"


def resolve_dtype(dtype_str: str):
    """Map CLI dtype string to a torch dtype."""
    import torch

    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def load_model(model_id: str, device: str, dtype: str):
    """
    Load a VLM model + processor from HuggingFace.

    Explicitly supports Qwen2-VL, LLaVA, LLaVA-NeXT.
    Falls back to AutoModelForVision2Seq + AutoProcessor for others.

    Returns (model, processor, model_type).
    """
    from transformers import AutoProcessor

    torch_dtype = resolve_dtype(dtype)
    model_type = detect_model_type(model_id)

    if model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    elif model_type == "llava":
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    elif model_type == "llava_next":
        from transformers import LlavaNextForConditionalGeneration

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_id)

    else:  # auto — works for InternVL, Ovis, Phi-3-vision, Llama-Vision, etc.
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    return model, processor, model_type


# ─── Inference ────────────────────────────────────────────────────────────────


def generate_response(model, processor, model_type, image, prompt, max_new_tokens) -> str:
    """Generate text from a HuggingFace VLM given an image and prompt."""
    if model_type == "qwen2_vl":
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_type == "llava":
        conversation = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(text=conversation, images=image, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        return response

    if model_type == "llava_next":
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        conversation = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=conversation, images=image, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        return response

    # auto fallback
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output_ids[0], skip_special_tokens=True)


# ─── Evaluation loop ─────────────────────────────────────────────────────────


def evaluate(model, processor, model_type, dataset_items, sample_size, seed,
             verbose, max_new_tokens):
    """Run image-only evaluation over *dataset_items*."""
    random.seed(seed)
    if sample_size is not None and sample_size < len(dataset_items):
        dataset_items = random.sample(dataset_items, sample_size)

    results = []
    total = len(dataset_items)

    for i, (idx, item) in enumerate(dataset_items):
        image = item["img"]
        correct = item["correct answer"].strip().upper()
        nation = item["nation"]
        task = item["task"]

        if verbose or (i + 1) % 10 == 0:
            print(f"[{i + 1}/{total}] Index={idx}, Nation={nation}, Task={task}")

        try:
            response = generate_response(
                model, processor, model_type, image, PROMPT, max_new_tokens,
            )
        except Exception as e:
            response = f"ERROR: {str(e)}"
            if verbose:
                print(f"  Error: {str(e)[:80]}")

        predicted = extract_answer(response)
        is_correct = predicted == correct

        if verbose:
            print(f"  Correct={correct}, Predicted={predicted} "
                  f"{'CORRECT' if is_correct else 'WRONG'}")

        results.append({
            "index": idx,
            "nation": nation,
            "task": task,
            "correct_answer": correct,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "response": response[:500] if len(response) > 500 else response,
        })

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EuraGovExam — HuggingFace Local Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hf.py --model Qwen/Qwen2-VL-7B-Instruct
  python run_hf.py --model llava-hf/llava-1.5-7b-hf --dtype float16
  python run_hf.py --model Qwen/Qwen2-VL-7B-Instruct --nation japan --sample-size 100
        """,
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen2-VL-7B-Instruct)")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Dataset split (default: train)")
    parser.add_argument("--nation", default=None, choices=list(NATION_MAP.keys()),
                        help="Filter by nation")
    parser.add_argument("--domain", default=None, help="Filter by domain/task")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Override sample count (default: use all)")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for JSON results (default: results)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--device", default="auto",
                        help="Torch device: auto, cuda, cuda:0, cpu (default: auto)")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Model dtype (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max new tokens to generate (default: 512)")

    args = parser.parse_args()

    # Print configuration
    print("=" * 70)
    print("EuraGovExam — HuggingFace Local Model Evaluation")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Device:      {args.device}")
    print(f"Dtype:       {args.dtype}")
    print(f"Split:       {args.split}")
    print(f"Nation:      {args.nation or 'all'}")
    print(f"Domain:      {args.domain or 'all'}")
    print(f"Sample size: {args.sample_size or 'all filtered'}")
    print(f"Seed:        {args.seed}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model, processor, model_type = load_model(args.model, args.device, args.dtype)
    print(f"  Model loaded (type={model_type})")

    # Load dataset
    print("\n[2/4] Loading dataset...")
    dataset_items = load_and_filter_dataset(args.nation, args.domain, args.split)
    if not dataset_items:
        print("\nERROR: No samples match the specified filters. Exiting.")
        return

    # Evaluate
    import torch

    print("\n[3/4] Running evaluation...")
    with torch.no_grad():
        results = evaluate(
            model, processor, model_type, dataset_items,
            args.sample_size, args.seed, args.verbose, args.max_new_tokens,
        )

    # Metrics & save
    print("\n[4/4] Calculating metrics...")
    metrics = calculate_metrics(results)
    output_file = save_results(results, metrics, args, Path(args.output_dir))
    print(f"  Results saved to: {output_file}")

    print_summary(metrics, args)


if __name__ == "__main__":
    main()
