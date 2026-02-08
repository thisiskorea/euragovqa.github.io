"""
EuraGovExam Large-Scale 3-Track Experiment
==========================================
- 1,000 samples (200 per region)
- Intermediate saving every 50 samples
- Statistical analysis (Bootstrap CI, McNemar test)
"""

import os
import re
import json
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import google.generativeai as genai
from datasets import load_dataset
from PIL import Image

GEMINI_API_KEY = ""
MODEL_NAME = "gemini-2.0-flash"
DATASET_NAME = "EuraGovExam/EuraGovExam"

SAMPLE_SIZE = 200
NATION_DISTRIBUTION = {
    "South Korea": 40,
    "Japan": 40,
    "EU": 40,
    "India": 40,
    "Taiwan": 40,
}

API_DELAY_SECONDS = 2.0
MAX_RETRIES = 2
SAVE_INTERVAL = 50

OUTPUT_DIR = Path(__file__).parent / "results"

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


def setup_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def extract_answer(response_text: str) -> str:
    if not response_text:
        return "INVALID"

    match = re.search(r"[Tt]he answer is\s*[:\s]*([A-Ea-e])[\.\s]?", response_text)
    if match:
        return match.group(1).upper()

    match = re.search(r"[Aa]nswer\s*[:\s]+([A-Ea-e])[\.\s]?", response_text)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?:^|\s)([A-E])\.?\s*$", response_text.strip(), re.MULTILINE)
    if match:
        return match.group(1).upper()

    letters = re.findall(r"\b([A-E])\b", response_text.upper())
    if letters:
        return letters[-1]

    return "INVALID"


def api_call_with_retry(model, contents, max_retries=MAX_RETRIES):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = API_DELAY_SECONDS * (attempt + 1)
                print(
                    f"    [Retry {attempt + 1}] Error: {str(e)[:50]}... waiting {wait_time}s"
                )
                time.sleep(wait_time)
    return f"ERROR: {str(last_error)}"


def stratified_sample(dataset, nation_counts: dict) -> list:
    indices_by_nation = defaultdict(list)
    for idx, item in enumerate(dataset):
        nation = item["nation"]
        if nation in nation_counts:
            indices_by_nation[nation].append(idx)

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


def run_track_a(model, image) -> tuple:
    response = api_call_with_retry(model, [PROMPT_TRACK_A, image])
    answer = extract_answer(response)
    return response, answer


def run_ocr(model, image) -> str:
    response = api_call_with_retry(model, [PROMPT_OCR, image])
    return response


def run_track_b(model, ocr_text: str) -> tuple:
    prompt = PROMPT_TRACK_B.format(ocr_text=ocr_text)
    response = api_call_with_retry(model, prompt)
    answer = extract_answer(response)
    return response, answer


def run_track_c(model, image, ocr_text: str) -> tuple:
    prompt = PROMPT_TRACK_C.format(ocr_text=ocr_text)
    response = api_call_with_retry(model, [prompt, image])
    answer = extract_answer(response)
    return response, answer


def bootstrap_ci(
    correct_array: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
) -> dict:
    n = len(correct_array)
    accuracies = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        acc = correct_array[indices].mean()
        accuracies.append(acc)

    lower = np.percentile(accuracies, (1 - ci) / 2 * 100)
    upper = np.percentile(accuracies, (1 + ci) / 2 * 100)

    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    b = ((a_correct) & (~b_correct)).sum()
    c = ((~a_correct) & (b_correct)).sum()

    if b + c == 0:
        return {"b": int(b), "c": int(c), "p_value": 1.0, "significant": False}

    if b + c < 25:
        from scipy.stats import binom_test

        p_value = binom_test(min(b, c), b + c, 0.5) * 2
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2 as chi2_dist

        p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "a_only_correct": int(b),
        "b_only_correct": int(c),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def save_intermediate_results(
    results: list,
    stats: dict,
    nation_stats: dict,
    output_file: Path,
    sample_indices: list,
    current_idx: int,
):
    n = current_idx + 1

    track_a_acc = stats["track_a"]["correct"] / n * 100 if n > 0 else 0
    track_b_acc = stats["track_b"]["correct"] / n * 100 if n > 0 else 0
    track_c_acc = stats["track_c"]["correct"] / n * 100 if n > 0 else 0

    summary = {
        "experiment": "large-scale-3track",
        "model": MODEL_NAME,
        "sample_size": n,
        "total_planned": len(sample_indices),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": "in_progress" if n < len(sample_indices) else "completed",
        "nation_distribution": NATION_DISTRIBUTION,
        "summary": {
            "track_a_accuracy": round(track_a_acc, 2),
            "track_b_accuracy": round(track_b_acc, 2),
            "track_c_accuracy": round(track_c_acc, 2),
            "track_a_correct": stats["track_a"]["correct"],
            "track_b_correct": stats["track_b"]["correct"],
            "track_c_correct": stats["track_c"]["correct"],
            "delta_b_minus_a": round(track_b_acc - track_a_acc, 2),
            "delta_c_minus_a": round(track_c_acc - track_a_acc, 2),
            "delta_c_minus_b": round(track_c_acc - track_b_acc, 2),
        },
        "by_nation": {},
        "details": results,
    }

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


def compute_final_statistics(results: list) -> dict:
    track_a_correct = np.array([r["track_a"]["is_correct"] for r in results])
    track_b_correct = np.array([r["track_b"]["is_correct"] for r in results])
    track_c_correct = np.array([r["track_c"]["is_correct"] for r in results])

    stats = {
        "track_a": {
            "accuracy": float(track_a_correct.mean() * 100),
            "bootstrap_ci": bootstrap_ci(track_a_correct),
        },
        "track_b": {
            "accuracy": float(track_b_correct.mean() * 100),
            "bootstrap_ci": bootstrap_ci(track_b_correct),
        },
        "track_c": {
            "accuracy": float(track_c_correct.mean() * 100),
            "bootstrap_ci": bootstrap_ci(track_c_correct),
        },
        "mcnemar_b_vs_a": mcnemar_test(track_a_correct, track_b_correct),
        "mcnemar_c_vs_a": mcnemar_test(track_a_correct, track_c_correct),
        "mcnemar_c_vs_b": mcnemar_test(track_b_correct, track_c_correct),
    }

    return stats


def run_experiment():
    print("=" * 70)
    print("EuraGovExam Large-Scale 3-Track Experiment")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Target Sample Size: {SAMPLE_SIZE}")
    print(f"Distribution: {NATION_DISTRIBUTION}")
    print(f"Save Interval: Every {SAVE_INTERVAL} samples")
    print("=" * 70)

    model = setup_gemini()
    print("\n[1/4] Gemini model initialized")

    print("[2/4] Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"       Dataset loaded: {len(dataset)} items")

    sample_indices = stratified_sample(dataset, NATION_DISTRIBUTION)
    print(f"[3/4] Sampled {len(sample_indices)} items")

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"large_scale_{timestamp}.json"

    print(f"[4/4] Running 3-track experiment...")
    print(f"       Output: {output_file}")
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

        track_a_response, track_a_answer = run_track_a(model, image)
        track_a_correct = track_a_answer == correct
        stats["track_a"]["total"] += 1
        if track_a_correct:
            stats["track_a"]["correct"] += 1
        nation_stats[nation]["track_a"]["total"] += 1
        if track_a_correct:
            nation_stats[nation]["track_a"]["correct"] += 1
        print(f"  Track A (Image): {track_a_answer} {'✓' if track_a_correct else '✗'}")

        time.sleep(API_DELAY_SECONDS)

        ocr_text = run_ocr(model, image)
        ocr_success = not ocr_text.startswith("ERROR")
        print(f"  OCR: {'OK' if ocr_success else 'FAIL'} ({len(ocr_text)} chars)")

        time.sleep(API_DELAY_SECONDS)

        track_b_response, track_b_answer = run_track_b(model, ocr_text)
        track_b_correct = track_b_answer == correct
        stats["track_b"]["total"] += 1
        if track_b_correct:
            stats["track_b"]["correct"] += 1
        nation_stats[nation]["track_b"]["total"] += 1
        if track_b_correct:
            nation_stats[nation]["track_b"]["correct"] += 1
        print(f"  Track B (Text):  {track_b_answer} {'✓' if track_b_correct else '✗'}")

        time.sleep(API_DELAY_SECONDS)

        track_c_response, track_c_answer = run_track_c(model, image, ocr_text)
        track_c_correct = track_c_answer == correct
        stats["track_c"]["total"] += 1
        if track_c_correct:
            stats["track_c"]["correct"] += 1
        nation_stats[nation]["track_c"]["total"] += 1
        if track_c_correct:
            nation_stats[nation]["track_c"]["correct"] += 1
        print(f"  Track C (Both):  {track_c_answer} {'✓' if track_c_correct else '✗'}")

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

        if (i + 1) % SAVE_INTERVAL == 0:
            save_intermediate_results(
                results, stats, nation_stats, output_file, sample_indices, i
            )
            print(f"\n  [SAVED] Intermediate results at {i+1} samples")

        time.sleep(API_DELAY_SECONDS)

    save_intermediate_results(
        results,
        stats,
        nation_stats,
        output_file,
        sample_indices,
        len(sample_indices) - 1,
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_stats = compute_final_statistics(results)

    n = len(results)
    print(f"\nOverall Accuracy (n={n}):")
    for track in ["track_a", "track_b", "track_c"]:
        acc = final_stats[track]["accuracy"]
        ci = final_stats[track]["bootstrap_ci"]
        print(
            f"  {track.upper()}: {acc:.1f}% [95% CI: {ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%]"
        )

    print(f"\nStatistical Tests (McNemar):")
    for test_name in ["mcnemar_b_vs_a", "mcnemar_c_vs_a", "mcnemar_c_vs_b"]:
        test = final_stats[test_name]
        sig = "**" if test["significant"] else ""
        print(f"  {test_name}: p={test['p_value']:.4f} {sig}")

    with open(output_file, "r", encoding="utf-8") as f:
        final_data = json.load(f)
    final_data["final_statistics"] = final_stats
    final_data["status"] = "completed"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")
    return final_data


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_experiment()
