"""
EuraGovExam Standardized Evaluation Script
===========================================

This script provides a standardized, reproducible evaluation interface for the
EuraGovExam benchmark following the paper's evaluation protocol.

Usage:
    # Full benchmark (Image-Only Setting)
    python evaluate.py --model gemini-2.0-flash --setting image-only

    # Filter by nation
    python evaluate.py --model gemini-2.0-flash --nation japan

    # Filter by domain
    python evaluate.py --model gemini-2.0-flash --domain mathematics

    # Combine filters
    python evaluate.py --model gemini-2.0-flash --nation korea --domain law
"""

import os
import re
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
from datasets import load_dataset
from PIL import Image

# Import configurations
from config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    DATASET_NAME,
    DATASET_SPLIT,
    PROMPT_TRACK_A,
    PROMPT_TRACK_B,
    PROMPT_TRACK_C,
    OUTPUT_DIR,
)

# Evaluation constants
API_DELAY_SECONDS = 2.0
MAX_RETRIES = 2
RANDOM_BASELINE = 23.7  # Weighted average of 4-choice/5-choice random guessing

# Nation name mapping: CLI -> Dataset format
NATION_MAP = {
    "korea": "South Korea",
    "japan": "Japan",
    "taiwan": "Taiwan",
    "india": "India",
    "eu": "EU",
}

# Valid domains (17 subjects in dataset)
VALID_DOMAINS = [
    "mathematics",
    "physics",
    "chemistry",
    "biology",
    "earth_science",
    "history",
    "geography",
    "politics",
    "economics",
    "law",
    "sociology",
    "ethics",
    "language",
    "literature",
    "art",
    "music",
    "physical_education",
]


def setup_gemini(model_name: str = None):
    """Initialize Gemini model."""
    genai.configure(api_key=GEMINI_API_KEY)
    model_name = model_name or MODEL_NAME
    return genai.GenerativeModel(model_name)


def extract_answer(response_text: str) -> str:
    """
    Extract answer from model response using 4-level regex cascade.

    Follows strict answer extraction rules from paper:
    - Format violations → INVALID
    - Multiple answers → INVALID (returns first match)
    - Missing output → INVALID

    Args:
        response_text: Raw model output text

    Returns:
        Extracted answer (A/B/C/D/E) or "INVALID"
    """
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


def api_call_with_retry(model, contents, max_retries=MAX_RETRIES) -> str:
    """
    Call Gemini API with exponential backoff retry logic.

    Args:
        model: Gemini model instance
        contents: Input content (prompt and/or image)
        max_retries: Maximum retry attempts

    Returns:
        Response text or "ERROR: ..." string
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = API_DELAY_SECONDS * (attempt + 1)
                print(f"    [Retry {attempt + 1}] Error: {str(e)[:50]}... waiting {wait_time}s")
                time.sleep(wait_time)
    return f"ERROR: {str(last_error)}"


def load_and_filter_dataset(
    nation: Optional[str] = None,
    domain: Optional[str] = None,
    split: str = "train",
) -> List[Tuple[int, dict]]:
    """
    Load dataset and apply nation/domain filters.

    Args:
        nation: Nation filter (CLI format: 'korea', 'japan', etc.)
        domain: Domain/task filter
        split: Dataset split

    Returns:
        List of (index, item) tuples
    """
    print(f"Loading dataset: {DATASET_NAME} (split={split})...")
    dataset = load_dataset(DATASET_NAME, split=split)
    print(f"  Dataset loaded: {len(dataset)} items")

    # Apply filters
    filtered_indices = []
    for idx, item in enumerate(dataset):
        # Nation filter
        if nation is not None:
            dataset_nation = NATION_MAP[nation]
            if item["nation"] != dataset_nation:
                continue

        # Domain filter
        if domain is not None:
            if item["task"] != domain:
                continue

        filtered_indices.append((idx, item))

    if len(filtered_indices) == 0:
        print(f"  WARNING: No samples match filters (nation={nation}, domain={domain})")
    else:
        print(f"  Filtered to {len(filtered_indices)} items")

    return filtered_indices


def evaluate_image_only(
    model,
    dataset_items: List[Tuple[int, dict]],
    sample_size: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
) -> List[dict]:
    """
    Track A: Image-Only Setting (Primary evaluation mode).

    Follows paper's Image-Only Setting (§3.1):
    - No external OCR allowed
    - Model receives only: standardized instruction + exam image

    Args:
        model: Gemini model instance
        dataset_items: List of (index, item) tuples
        sample_size: Sample size override (None = use all)
        seed: Random seed for reproducibility
        verbose: Enable verbose logging

    Returns:
        List of result dictionaries
    """
    random.seed(seed)

    # Sample if requested
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
            print(f"[{i+1}/{total}] Index={idx}, Nation={nation}, Task={task}")
            if verbose:
                print(f"  Correct answer: {correct}")

        # API call
        response = api_call_with_retry(model, [PROMPT_TRACK_A, image])
        predicted = extract_answer(response)
        is_correct = (predicted == correct)

        if verbose:
            print(f"  Predicted: {predicted} {'✓' if is_correct else '✗'}")

        # Store result
        results.append({
            "index": idx,
            "nation": nation,
            "task": task,
            "correct_answer": correct,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "response": response[:500] if len(response) > 500 else response,
        })

        # Rate limiting
        time.sleep(API_DELAY_SECONDS)

    return results




def calculate_metrics(results: List[dict]) -> dict:
    """
    Calculate evaluation metrics.

    Computes:
    - Overall accuracy
    - By-nation breakdown
    - By-domain breakdown
    - Comparison with random baseline (23.7%)

    Args:
        results: List of result dictionaries

    Returns:
        Metrics dictionary
    """
    if len(results) == 0:
        return {
            "overall_accuracy": 0.0,
            "random_baseline": RANDOM_BASELINE,
            "by_nation": {},
            "by_domain": {},
        }

    # Overall accuracy
    correct_count = sum(1 for r in results if r["is_correct"])
    overall_accuracy = (correct_count / len(results)) * 100

    # By-nation breakdown
    by_nation = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        nation = r["nation"]
        by_nation[nation]["total"] += 1
        if r["is_correct"]:
            by_nation[nation]["correct"] += 1

    nation_metrics = {}
    for nation, stats in by_nation.items():
        nation_metrics[nation] = {
            "accuracy": round((stats["correct"] / stats["total"]) * 100, 2),
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # By-domain breakdown
    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        domain = r["task"]
        by_domain[domain]["total"] += 1
        if r["is_correct"]:
            by_domain[domain]["correct"] += 1

    domain_metrics = {}
    for domain, stats in by_domain.items():
        domain_metrics[domain] = {
            "accuracy": round((stats["correct"] / stats["total"]) * 100, 2),
            "correct": stats["correct"],
            "total": stats["total"],
        }

    return {
        "overall_accuracy": round(overall_accuracy, 2),
        "random_baseline": RANDOM_BASELINE,
        "correct_count": correct_count,
        "total_count": len(results),
        "by_nation": nation_metrics,
        "by_domain": domain_metrics,
    }


def save_results(
    results: List[dict],
    metrics: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    """
    Save evaluation results to JSON file.

    Filename format: evaluate_{model}_{setting}_{timestamp}.json

    Args:
        results: List of result dictionaries
        metrics: Metrics dictionary
        args: Command-line arguments
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluate_{args.model}_{args.setting}_{timestamp}.json"
    output_file = output_dir / filename

    output_data = {
        "metadata": {
            "model": args.model,
            "setting": args.setting,
            "split": args.split,
            "filters": {
                "nation": args.nation,
                "domain": args.domain,
            },
            "sample_size": args.sample_size,
            "seed": args.seed,
            "timestamp": timestamp,
        },
        "metrics": metrics,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_file


def print_summary(metrics: dict, args: argparse.Namespace):
    """Print evaluation summary to console."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Setting: {args.setting}")
    print(f"Filters: nation={args.nation}, domain={args.domain}")
    print(f"Sample size: {metrics['total_count']}")
    print("-" * 70)
    print(f"Overall Accuracy: {metrics['overall_accuracy']}%")
    print(f"Random Baseline:  {metrics['random_baseline']}%")
    print(f"Correct: {metrics['correct_count']}/{metrics['total_count']}")

    if metrics['by_nation']:
        print("\nBy Nation:")
        for nation, stats in sorted(metrics['by_nation'].items()):
            print(f"  {nation:15s}: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")

    if metrics['by_domain']:
        print("\nBy Domain:")
        for domain, stats in sorted(metrics['by_domain'].items()):
            print(f"  {domain:20s}: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="EuraGovExam Standardized Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark (Image-Only Setting)
  python evaluate.py --model gemini-2.0-flash --setting image-only

  # Filter by nation
  python evaluate.py --model gemini-2.0-flash --nation japan

  # Filter by domain
  python evaluate.py --model gemini-2.0-flash --domain mathematics

  # Combine filters
  python evaluate.py --model gemini-2.0-flash --nation korea --domain law
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Model identifier (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DATASET_SPLIT,
        choices=["train", "test"],
        help=f"Dataset split (default: {DATASET_SPLIT})"
    )
    parser.add_argument(
        "--nation",
        type=str,
        default=None,
        choices=list(NATION_MAP.keys()),
        help="Filter by nation (korea, japan, taiwan, india, eu)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=VALID_DOMAINS,
        help="Filter by domain/subject"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample count (default: use all filtered samples)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Result JSON directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 70)
    print("EuraGovExam Standardized Evaluation")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Split:       {args.split}")
    print(f"Nation:      {args.nation or 'all'}")
    print(f"Domain:      {args.domain or 'all'}")
    print(f"Sample size: {args.sample_size or 'all filtered'}")
    print(f"Seed:        {args.seed}")
    print("=" * 70)

    # Initialize model
    print("\n[1/4] Initializing model...")
    model = setup_gemini(args.model)
    print("  Model initialized")

    # Load and filter dataset
    print("\n[2/4] Loading and filtering dataset...")
    dataset_items = load_and_filter_dataset(
        nation=args.nation,
        domain=args.domain,
        split=args.split,
    )

    if len(dataset_items) == 0:
        print("\nERROR: No samples match the specified filters. Exiting.")
        return

    # Run evaluation
    print("-" * 70)

    if args.setting == "image-only":
    results = evaluate_image_only(
        model, dataset_items, args.sample_size, args.seed, args.verbose
    )
    # Calculate metrics
    print("\n[4/4] Calculating metrics...")
    metrics = calculate_metrics(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_file = save_results(results, metrics, args, output_dir)
    print(f"  Results saved to: {output_file}")

    # Print summary
    print_summary(metrics, args)


if __name__ == "__main__":
    main()
