"""
Multi-Model Comparison Experiment
=================================
Compare different Gemini model variants on EuraGovExam.
Models: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
"""

import os
import json
import time
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import re

import google.generativeai as genai
from datasets import load_dataset

GEMINI_API_KEY = ""

MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

DATASET_NAME = "EuraGovExam/EuraGovExam"
SAMPLE_SIZE = 50
NATION_DISTRIBUTION = {
    "South Korea": 10,
    "Japan": 10,
    "EU": 10,
    "India": 10,
    "Taiwan": 10,
}

API_DELAY = 3.0
MAX_RETRIES = 2
OUTPUT_DIR = Path(__file__).parent / "results"

PROMPT_TRACK_A = """You are solving a multiple-choice exam question shown in the image.
Carefully examine the image, read the question and all answer options.
Think through the problem step by step, then provide your final answer.

At the very end, provide the final answer in exactly this format:
The answer is X.

(where X is A, B, C, D, or E)"""


def extract_answer(response_text: str) -> str:
    if not response_text:
        return "INVALID"

    match = re.search(r"[Tt]he answer is\s*[:\s]*([A-Ea-e])[\.\s]?", response_text)
    if match:
        return match.group(1).upper()

    match = re.search(r"[Aa]nswer\s*[:\s]+([A-Ea-e])[\.\s]?", response_text)
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
                wait_time = API_DELAY * (attempt + 1)
                print(
                    f"    [Retry {attempt + 1}] {str(e)[:50]}... waiting {wait_time}s"
                )
                time.sleep(wait_time)
    return f"ERROR: {str(last_error)}"


def stratified_sample(dataset, nation_counts: dict) -> list:
    indices_by_nation = defaultdict(list)
    for idx, item in enumerate(dataset):
        nation = item["nation"]
        if nation in nation_counts:
            indices_by_nation[nation].append(idx)

    sampled = []
    for nation, count in nation_counts.items():
        available = indices_by_nation[nation]
        selected = random.sample(available, min(count, len(available)))
        sampled.extend(selected)

    random.shuffle(sampled)
    return sampled


def run_multi_model_experiment():
    print("=" * 70)
    print("Multi-Model Comparison Experiment")
    print("=" * 70)
    print(f"Models: {MODELS}")
    print(f"Sample Size: {SAMPLE_SIZE}")
    print("=" * 70)

    genai.configure(api_key=GEMINI_API_KEY)

    print("\n[1/3] Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"       Loaded {len(dataset)} items")

    sample_indices = stratified_sample(dataset, NATION_DISTRIBUTION)
    print(f"[2/3] Sampled {len(sample_indices)} items")

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"multi_model_{timestamp}.json"

    results = {
        "experiment": "multi-model-comparison",
        "models": MODELS,
        "sample_size": len(sample_indices),
        "timestamp": timestamp,
        "by_model": {},
        "details": [],
    }

    print(f"[3/3] Running experiments...")
    print("-" * 70)

    for model_name in MODELS:
        print(f"\n>>> Testing {model_name}")
        model = genai.GenerativeModel(model_name)

        model_stats = {
            "correct": 0,
            "total": 0,
            "by_nation": defaultdict(lambda: {"correct": 0, "total": 0}),
        }

        for i, idx in enumerate(sample_indices):
            item = dataset[idx]
            image = item["img"]
            correct = item["correct answer"].strip().upper()
            nation = item["nation"]

            print(f"  [{i+1}/{len(sample_indices)}] {nation} - ", end="", flush=True)

            response = api_call_with_retry(model, [PROMPT_TRACK_A, image])
            answer = extract_answer(response)
            is_correct = answer == correct

            model_stats["total"] += 1
            if is_correct:
                model_stats["correct"] += 1

            model_stats["by_nation"][nation]["total"] += 1
            if is_correct:
                model_stats["by_nation"][nation]["correct"] += 1

            print(f"{answer} {'O' if is_correct else 'X'}")

            results["details"].append(
                {
                    "index": idx,
                    "model": model_name,
                    "nation": nation,
                    "task": item["task"],
                    "correct_answer": correct,
                    "predicted": answer,
                    "is_correct": is_correct,
                }
            )

            time.sleep(API_DELAY)

        accuracy = model_stats["correct"] / model_stats["total"] * 100
        print(f"\n  {model_name} Accuracy: {accuracy:.1f}%")

        results["by_model"][model_name] = {
            "accuracy": round(accuracy, 2),
            "correct": model_stats["correct"],
            "total": model_stats["total"],
            "by_nation": {
                nation: {
                    "accuracy": round(stats["correct"] / stats["total"] * 100, 2)
                    if stats["total"] > 0
                    else 0,
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
                for nation, stats in model_stats["by_nation"].items()
            },
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<25} {'Accuracy':>10}")
    print("-" * 40)
    for model_name in MODELS:
        acc = results["by_model"][model_name]["accuracy"]
        print(f"{model_name:<25} {acc:>9.1f}%")

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    random.seed(42)
    run_multi_model_experiment()
