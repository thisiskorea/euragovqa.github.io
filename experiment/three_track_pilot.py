"""
EuraGovExam 3-Track Pilot Experiment
====================================
Track A: Image-only (VLM capability)
Track B: Text-only (OCR -> reasoning, no vision)
Track C: Image + Text (multimodal synergy)

Author: EuraGovExam Research
Date: 2025-01-20
"""

import os
import re
import json
import time
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import google.generativeai as genai
from datasets import load_dataset
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

GEMINI_API_KEY = ""
MODEL_NAME = "gemini-2.0-flash"
DATASET_NAME = "EuraGovExam/EuraGovExam"

# Experiment: 50 samples stratified by nation
PILOT_SAMPLE_SIZE = 50
NATION_DISTRIBUTION = {
    "South Korea": 15,
    "Japan": 12,
    "EU": 12,
    "India": 8,
    "Taiwan": 3,
}

# Rate limiting
API_DELAY_SECONDS = 2.5  # Delay between API calls
MAX_RETRIES = 1

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results"

# =============================================================================
# Prompts
# =============================================================================

PROMPT_TRACK_A = """You are solving a multiple-choice exam question shown in the image.
Carefully examine the image, read the question and all answer options.
Think through the problem step by step, then provide your final answer.

At the very end, provide the final answer in exactly this format:
The answer is X.

(where X is A, B, C, D, or E)"""

PROMPT_OCR = """Extract ALL text from this image exactly as written.
Preserve the structure, including:
- Question number and text
- All answer options (A, B, C, D, E) with their full text
- Any tables, diagrams descriptions, or additional context
- Maintain the original formatting and line breaks where meaningful

Output only the extracted text, nothing else."""

PROMPT_TRACK_B = """You are solving a multiple-choice exam question.
The question text has been extracted from a scanned exam document.

Here is the extracted text:
---
{ocr_text}
---

Based ONLY on the text above (no image), analyze the question and answer options.
Think through the problem step by step, then provide your final answer.

At the very end, provide the final answer in exactly this format:
The answer is X.

(where X is A, B, C, D, or E)"""

PROMPT_TRACK_C = """You are solving a multiple-choice exam question.
You have both the original image AND extracted text to help you.

Here is the text extracted from the image:
---
{ocr_text}
---

Use BOTH the image (for visual elements, diagrams, formatting) AND the extracted text 
(for precise reading) to answer the question.
Think through the problem step by step, then provide your final answer.

At the very end, provide the final answer in exactly this format:
The answer is X.

(where X is A, B, C, D, or E)"""


# =============================================================================
# Core Functions
# =============================================================================


def setup_gemini():
    """Initialize and return Gemini model."""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    return model


def extract_answer(response_text: str) -> str:
    """Extract answer letter from model response."""
    if not response_text:
        return "INVALID"

    # Pattern 1: "The answer is X" (preferred format)
    match = re.search(r"[Tt]he answer is\s*[:\s]*([A-Ea-e])[\.\s]?", response_text)
    if match:
        return match.group(1).upper()

    # Pattern 2: "Answer: X" or "Answer is X"
    match = re.search(r"[Aa]nswer\s*[:\s]+([A-Ea-e])[\.\s]?", response_text)
    if match:
        return match.group(1).upper()

    # Pattern 3: Standalone letter at end (like "B." or just "B")
    match = re.search(r"(?:^|\s)([A-E])\.?\s*$", response_text.strip(), re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 4: Last capital letter A-E in response
    letters = re.findall(r"\b([A-E])\b", response_text.upper())
    if letters:
        return letters[-1]

    return "INVALID"


def api_call_with_retry(model, contents, max_retries=MAX_RETRIES):
    """Make API call with retry logic."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"    [Retry {attempt + 1}] Error: {str(e)[:50]}...")
                time.sleep(API_DELAY_SECONDS)

    return f"ERROR: {str(last_error)}"


def stratified_sample(dataset, nation_counts: dict) -> list:
    """
    Sample items stratified by nation.

    Args:
        dataset: HuggingFace dataset
        nation_counts: Dict mapping nation -> number of samples

    Returns:
        List of dataset indices
    """
    indices_by_nation = defaultdict(list)

    # Group indices by nation
    for idx, item in enumerate(dataset):
        nation = item["nation"]
        if nation in nation_counts:
            indices_by_nation[nation].append(idx)

    # Sample from each nation
    sampled_indices = []
    for nation, count in nation_counts.items():
        available = indices_by_nation[nation]
        if len(available) >= count:
            selected = random.sample(available, count)
        else:
            selected = available
            print(
                f"  Warning: Only {len(available)} samples available for {nation}, needed {count}"
            )
        sampled_indices.extend(selected)

    random.shuffle(sampled_indices)
    return sampled_indices


# =============================================================================
# Track Functions
# =============================================================================


def run_track_a(model, image) -> tuple:
    """
    Track A: Image-only
    Input: Image only
    Output: (raw_response, extracted_answer)
    """
    response = api_call_with_retry(model, [PROMPT_TRACK_A, image])
    answer = extract_answer(response)
    return response, answer


def run_ocr(model, image) -> str:
    """
    OCR Step: Extract text from image.
    Used by Track B and Track C.
    """
    response = api_call_with_retry(model, [PROMPT_OCR, image])
    return response


def run_track_b(model, ocr_text: str) -> tuple:
    """
    Track B: Text-only (no image)
    Input: OCR-extracted text only
    Output: (raw_response, extracted_answer)
    """
    prompt = PROMPT_TRACK_B.format(ocr_text=ocr_text)
    response = api_call_with_retry(model, prompt)
    answer = extract_answer(response)
    return response, answer


def run_track_c(model, image, ocr_text: str) -> tuple:
    """
    Track C: Image + Text
    Input: Both image and OCR text
    Output: (raw_response, extracted_answer)
    """
    prompt = PROMPT_TRACK_C.format(ocr_text=ocr_text)
    response = api_call_with_retry(model, [prompt, image])
    answer = extract_answer(response)
    return response, answer


# =============================================================================
# Main Experiment
# =============================================================================


def run_experiment():
    """Run the 3-track pilot experiment."""
    print("=" * 70)
    print("EuraGovExam 3-Track Pilot Experiment")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Sample Size: {PILOT_SAMPLE_SIZE}")
    print(f"Distribution: {NATION_DISTRIBUTION}")
    print("=" * 70)

    # Setup
    model = setup_gemini()
    print("\n[1/4] Gemini model initialized")

    # Load dataset
    print("[2/4] Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"       Dataset loaded: {len(dataset)} items")

    # Stratified sampling
    sample_indices = stratified_sample(dataset, NATION_DISTRIBUTION)
    print(f"[3/4] Sampled {len(sample_indices)} items")

    # Run experiment
    print("[4/4] Running 3-track experiment...\n")
    print("-" * 70)

    results = []
    stats = {
        "track_a": {"correct": 0, "total": 0},
        "track_b": {"correct": 0, "total": 0},
        "track_c": {"correct": 0, "total": 0},
    }
    nation_stats = defaultdict(
        lambda: {
            "track_a": {"correct": 0, "total": 0},
            "track_b": {"correct": 0, "total": 0},
            "track_c": {"correct": 0, "total": 0},
        }
    )

    for i, idx in enumerate(sample_indices):
        item = dataset[idx]
        image = item["img"]
        correct = item["correct answer"].strip().upper()
        nation = item["nation"]
        task = item["task"]

        print(
            f"\n[{i+1}/{len(sample_indices)}] Index={idx}, Nation={nation}, Task={task}"
        )
        print(f"  Correct answer: {correct}")

        # --- Track A: Image-only ---
        track_a_response, track_a_answer = run_track_a(model, image)
        track_a_correct = track_a_answer == correct
        stats["track_a"]["total"] += 1
        if track_a_correct:
            stats["track_a"]["correct"] += 1
        nation_stats[nation]["track_a"]["total"] += 1
        if track_a_correct:
            nation_stats[nation]["track_a"]["correct"] += 1
        print(
            f"  Track A (Image): {track_a_answer} {'[CORRECT]' if track_a_correct else '[WRONG]'}"
        )

        time.sleep(API_DELAY_SECONDS)

        # --- OCR Step ---
        ocr_text = run_ocr(model, image)
        ocr_success = not ocr_text.startswith("ERROR")
        print(
            f"  OCR: {'Success' if ocr_success else 'Failed'} ({len(ocr_text)} chars)"
        )

        time.sleep(API_DELAY_SECONDS)

        # --- Track B: Text-only ---
        track_b_response, track_b_answer = run_track_b(model, ocr_text)
        track_b_correct = track_b_answer == correct
        stats["track_b"]["total"] += 1
        if track_b_correct:
            stats["track_b"]["correct"] += 1
        nation_stats[nation]["track_b"]["total"] += 1
        if track_b_correct:
            nation_stats[nation]["track_b"]["correct"] += 1
        print(
            f"  Track B (Text):  {track_b_answer} {'[CORRECT]' if track_b_correct else '[WRONG]'}"
        )

        time.sleep(API_DELAY_SECONDS)

        # --- Track C: Image + Text ---
        track_c_response, track_c_answer = run_track_c(model, image, ocr_text)
        track_c_correct = track_c_answer == correct
        stats["track_c"]["total"] += 1
        if track_c_correct:
            stats["track_c"]["correct"] += 1
        nation_stats[nation]["track_c"]["total"] += 1
        if track_c_correct:
            nation_stats[nation]["track_c"]["correct"] += 1
        print(
            f"  Track C (Both):  {track_c_answer} {'[CORRECT]' if track_c_correct else '[WRONG]'}"
        )

        # Store result
        result = {
            "index": idx,
            "nation": nation,
            "task": task,
            "correct_answer": correct,
            "track_a": {
                "answer": track_a_answer,
                "is_correct": track_a_correct,
                "response": track_a_response[:500]
                if len(track_a_response) > 500
                else track_a_response,
            },
            "track_b": {
                "answer": track_b_answer,
                "is_correct": track_b_correct,
                "ocr_text": ocr_text[:1000] if len(ocr_text) > 1000 else ocr_text,
                "response": track_b_response[:500]
                if len(track_b_response) > 500
                else track_b_response,
            },
            "track_c": {
                "answer": track_c_answer,
                "is_correct": track_c_correct,
                "response": track_c_response[:500]
                if len(track_c_response) > 500
                else track_c_response,
            },
        }
        results.append(result)

        time.sleep(API_DELAY_SECONDS)

    # =============================================================================
    # Compute Summary Statistics
    # =============================================================================

    n = len(sample_indices)
    track_a_acc = stats["track_a"]["correct"] / n * 100
    track_b_acc = stats["track_b"]["correct"] / n * 100
    track_c_acc = stats["track_c"]["correct"] / n * 100

    delta_ba = track_b_acc - track_a_acc
    delta_ca = track_c_acc - track_a_acc
    delta_cb = track_c_acc - track_b_acc

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nOverall Accuracy (n={n}):")
    print(
        f"  Track A (Image-only):     {track_a_acc:5.1f}%  ({stats['track_a']['correct']}/{n})"
    )
    print(
        f"  Track B (Text-only):      {track_b_acc:5.1f}%  ({stats['track_b']['correct']}/{n})"
    )
    print(
        f"  Track C (Image+Text):     {track_c_acc:5.1f}%  ({stats['track_c']['correct']}/{n})"
    )

    print(f"\nDelta Analysis:")
    print(
        f"  Delta (B - A):  {delta_ba:+5.1f}%  {'OCR > Vision' if delta_ba > 0 else 'Vision > OCR' if delta_ba < 0 else 'Equal'}"
    )
    print(
        f"  Delta (C - A):  {delta_ca:+5.1f}%  {'Synergy helps' if delta_ca > 0 else 'No synergy benefit' if delta_ca < 0 else 'Equal'}"
    )
    print(
        f"  Delta (C - B):  {delta_cb:+5.1f}%  {'Vision adds value' if delta_cb > 0 else 'Text sufficient' if delta_cb < 0 else 'Equal'}"
    )

    print(f"\nAccuracy by Nation:")
    for nation in NATION_DISTRIBUTION.keys():
        ns = nation_stats[nation]
        if ns["track_a"]["total"] > 0:
            a_acc = ns["track_a"]["correct"] / ns["track_a"]["total"] * 100
            b_acc = ns["track_b"]["correct"] / ns["track_b"]["total"] * 100
            c_acc = ns["track_c"]["correct"] / ns["track_c"]["total"] * 100
            print(
                f"  {nation:12s}: A={a_acc:5.1f}%, B={b_acc:5.1f}%, C={c_acc:5.1f}% (n={ns['track_a']['total']})"
            )

    print("-" * 70)

    # Interpretation
    print("\nInterpretation:")
    if track_c_acc >= track_a_acc and track_c_acc >= track_b_acc:
        print(
            "  -> Track C (Image+Text) performs best: Multimodal synergy is beneficial"
        )
    elif track_a_acc > track_b_acc:
        print("  -> Track A (Image-only) > Track B (Text-only): Visual grounding helps")
    elif track_b_acc > track_a_acc:
        print(
            "  -> Track B (Text-only) > Track A (Image-only): OCR/reading is the bottleneck"
        )
    else:
        print("  -> Tracks perform similarly: No clear modality advantage")

    # =============================================================================
    # Save Results
    # =============================================================================

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"3track_pilot_{timestamp}.json"

    summary = {
        "experiment": "3-track-pilot",
        "model": MODEL_NAME,
        "sample_size": n,
        "timestamp": timestamp,
        "nation_distribution": NATION_DISTRIBUTION,
        "summary": {
            "track_a_accuracy": round(track_a_acc, 2),
            "track_b_accuracy": round(track_b_acc, 2),
            "track_c_accuracy": round(track_c_acc, 2),
            "track_a_correct": stats["track_a"]["correct"],
            "track_b_correct": stats["track_b"]["correct"],
            "track_c_correct": stats["track_c"]["correct"],
            "delta_b_minus_a": round(delta_ba, 2),
            "delta_c_minus_a": round(delta_ca, 2),
            "delta_c_minus_b": round(delta_cb, 2),
        },
        "by_nation": {},
        "details": results,
    }

    # Add per-nation stats
    for nation in NATION_DISTRIBUTION.keys():
        ns = nation_stats[nation]
        if ns["track_a"]["total"] > 0:
            summary["by_nation"][nation] = {
                "n": ns["track_a"]["total"],
                "track_a_accuracy": round(
                    ns["track_a"]["correct"] / ns["track_a"]["total"] * 100, 2
                ),
                "track_b_accuracy": round(
                    ns["track_b"]["correct"] / ns["track_b"]["total"] * 100, 2
                ),
                "track_c_accuracy": round(
                    ns["track_c"]["correct"] / ns["track_c"]["total"] * 100, 2
                ),
            }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")

    return summary


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    random.seed(42)
    run_experiment()
