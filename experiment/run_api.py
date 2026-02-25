#!/usr/bin/env python3
"""
EuraGovExam API Evaluation Script
==================================

Evaluate VLMs on the EuraGovExam benchmark using cloud APIs (Gemini, OpenAI).

Usage:
    # Gemini
    python run_api.py --provider gemini --model gemini-2.0-flash
    python run_api.py --provider gemini --model gemini-2.0-flash --nation japan

    # OpenAI
    python run_api.py --provider openai --model gpt-4o --api-key sk-...

    # Filters
    python run_api.py --provider gemini --model gemini-2.0-flash --nation korea --domain law
    python run_api.py --provider gemini --model gemini-2.0-flash --sample-size 100 --seed 42
"""

import os
import re
import json
import time
import random
import base64
import argparse
from io import BytesIO
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
    filename = f"eval_{args.provider}_{args.model}_{ts}.json"
    output_file = output_dir / filename

    data = {
        "metadata": {
            "model": args.model,
            "provider": args.provider,
            "split": args.split,
            "filters": {"nation": args.nation, "domain": args.domain},
            "sample_size": metrics["total_count"],
            "seed": args.seed,
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
    print(f"Provider: {args.provider}")
    print(f"Model:    {args.model}")
    print(f"Filters:  nation={args.nation or 'all'}, domain={args.domain or 'all'}")
    print(f"Samples:  {metrics['total_count']}")
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


# ─── Provider-specific functions ──────────────────────────────────────────────


def setup_gemini(model_name: str, api_key: str):
    """Initialise a Gemini GenerativeModel."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def call_gemini(model, image, prompt: str) -> str:
    """Send image + prompt to Gemini and return response text."""
    response = model.generate_content([prompt, image])
    return response.text


def setup_openai(api_key: str):
    """Initialise an OpenAI client."""
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def call_openai(client, model_name: str, image, prompt: str) -> str:
    """Send image + prompt to the OpenAI Chat Completions API."""
    # Convert PIL Image → base64 data-URL
    img = image.convert("RGB") if image.mode != "RGB" else image
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/png;base64,{b64}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content


def api_call_with_retry(call_fn, max_retries: int = 2, delay: float = 2.0) -> str:
    """Generic retry wrapper for *call_fn* (no-argument callable)."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = delay * (attempt + 1)
                print(f"    [Retry {attempt + 1}] {str(e)[:80]}... waiting {wait}s")
                time.sleep(wait)
    return f"ERROR: {str(last_error)}"


# ─── Evaluation loop ─────────────────────────────────────────────────────────


def evaluate(call_fn, dataset_items, sample_size, seed, verbose, delay, max_retries):
    """Run image-only evaluation over *dataset_items* using *call_fn(image, prompt)*."""
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

        response = api_call_with_retry(
            lambda _img=image: call_fn(_img, PROMPT),
            max_retries=max_retries,
            delay=delay,
        )
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

        time.sleep(delay)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EuraGovExam — Cloud API Evaluation (Gemini / OpenAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_api.py --provider gemini --model gemini-2.0-flash
  python run_api.py --provider openai --model gpt-4o --api-key sk-...
  python run_api.py --provider gemini --model gemini-2.0-flash --nation japan
  python run_api.py --provider gemini --model gemini-2.0-flash --sample-size 100
        """,
    )
    parser.add_argument("--provider", required=True, choices=["gemini", "openai"],
                        help="API provider")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--api-key", default=None,
                        help="API key (fallback: GEMINI_API_KEY / OPENAI_API_KEY env var)")
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
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between API calls (default: 2.0)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retry attempts per API call (default: 2)")

    args = parser.parse_args()

    # Resolve API key
    env_var = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"
    api_key = args.api_key or os.environ.get(env_var)
    if not api_key:
        parser.error(f"API key required. Use --api-key or set {env_var} env var.")

    # Print configuration
    print("=" * 70)
    print("EuraGovExam — Cloud API Evaluation")
    print("=" * 70)
    print(f"Provider:    {args.provider}")
    print(f"Model:       {args.model}")
    print(f"Split:       {args.split}")
    print(f"Nation:      {args.nation or 'all'}")
    print(f"Domain:      {args.domain or 'all'}")
    print(f"Sample size: {args.sample_size or 'all filtered'}")
    print(f"Seed:        {args.seed}")
    print("=" * 70)

    # Setup provider
    print("\n[1/4] Initialising model...")
    if args.provider == "gemini":
        model = setup_gemini(args.model, api_key)
        call_fn = lambda img, prompt: call_gemini(model, img, prompt)
    else:
        client = setup_openai(api_key)
        call_fn = lambda img, prompt: call_openai(client, args.model, img, prompt)
    print("  Model ready")

    # Load dataset
    print("\n[2/4] Loading dataset...")
    dataset_items = load_and_filter_dataset(args.nation, args.domain, args.split)
    if not dataset_items:
        print("\nERROR: No samples match the specified filters. Exiting.")
        return

    # Evaluate
    print("\n[3/4] Running evaluation...")
    results = evaluate(
        call_fn, dataset_items,
        args.sample_size, args.seed, args.verbose, args.delay, args.max_retries,
    )

    # Metrics & save
    print("\n[4/4] Calculating metrics...")
    metrics = calculate_metrics(results)
    output_file = save_results(results, metrics, args, Path(args.output_dir))
    print(f"  Results saved to: {output_file}")

    print_summary(metrics, args)


if __name__ == "__main__":
    main()
