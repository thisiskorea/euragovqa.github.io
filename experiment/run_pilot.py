"""
EuraGovExam Pilot Experiment
- Track A: Image-only (Gemini 2.0 Flash)
- Track B: OCR-text-only (Gemini 2.0 Flash as OCR + reasoning)
"""

import os
import re
import json
import time
import random
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from datasets import load_dataset, Dataset
from PIL import Image
import io
import base64

from config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    DATASET_NAME,
    PILOT_SAMPLE_SIZE,
    NATION_DISTRIBUTION,
    PROMPT_TRACK_A,
    PROMPT_TRACK_B,
    OUTPUT_DIR,
)


def setup_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    return model


def stratified_sample(dataset, n_samples, distribution):
    samples = []
    indices_by_nation = {nation: [] for nation in distribution.keys()}

    for idx, item in enumerate(dataset):
        nation = item["nation"]
        if nation in indices_by_nation:
            indices_by_nation[nation].append(idx)

    for nation, ratio in distribution.items():
        n_nation = max(1, int(n_samples * ratio))
        available = indices_by_nation[nation]
        if len(available) >= n_nation:
            selected = random.sample(available, n_nation)
        else:
            selected = available
        samples.extend(selected)

    random.shuffle(samples)
    return samples[:n_samples]


def extract_answer(response_text):
    patterns = [
        r"[Tt]he answer is ([A-Ea-e])",
        r"[Aa]nswer[:\s]+([A-Ea-e])",
        r"\b([A-E])\b(?:\s*$|\s*\.?\s*$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1).upper()

    last_letters = re.findall(r"\b([A-E])\b", response_text.upper())
    if last_letters:
        return last_letters[-1]

    return "INVALID"


def run_track_a(model, image):
    try:
        response = model.generate_content([PROMPT_TRACK_A, image])
        return response.text, extract_answer(response.text)
    except Exception as e:
        return f"ERROR: {str(e)}", "ERROR"


def run_track_b_ocr(model, image):
    ocr_prompt = """Extract ALL text from this image exactly as written.
Preserve the structure, line breaks, and formatting.
Include question numbers, options (A, B, C, D, E), and all content."""

    try:
        response = model.generate_content([ocr_prompt, image])
        return response.text
    except Exception as e:
        return f"OCR_ERROR: {str(e)}"


def run_track_b_reasoning(model, ocr_text):
    prompt = PROMPT_TRACK_B.format(ocr_text=ocr_text)
    try:
        response = model.generate_content(prompt)
        return response.text, extract_answer(response.text)
    except Exception as e:
        return f"ERROR: {str(e)}", "ERROR"


def run_experiment():
    print("=" * 60)
    print("EuraGovExam Pilot Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Sample Size: {PILOT_SAMPLE_SIZE}")
    print("=" * 60)

    model = setup_gemini()
    print("[1/4] Gemini model loaded")

    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"[2/4] Dataset loaded: {len(dataset)} items")

    sample_indices = stratified_sample(dataset, PILOT_SAMPLE_SIZE, NATION_DISTRIBUTION)
    print(f"[3/4] Sampled {len(sample_indices)} items")

    results = []
    track_a_correct = 0
    track_b_correct = 0

    print("[4/4] Running experiments...")
    print("-" * 60)

    for i, idx in enumerate(sample_indices):
        item = dataset[idx]
        image = item["img"]
        correct = item["correct_answer"].upper()
        nation = item["nation"]
        task = item["task"]

        print(f"\n[{i+1}/{len(sample_indices)}] Nation: {nation}, Task: {task}")

        track_a_response, track_a_answer = run_track_a(model, image)
        track_a_is_correct = track_a_answer == correct
        if track_a_is_correct:
            track_a_correct += 1
        print(
            f"  Track A: {track_a_answer} (correct: {correct}) - {'✓' if track_a_is_correct else '✗'}"
        )

        time.sleep(1)

        ocr_text = run_track_b_ocr(model, image)
        track_b_response, track_b_answer = run_track_b_reasoning(model, ocr_text)
        track_b_is_correct = track_b_answer == correct
        if track_b_is_correct:
            track_b_correct += 1
        print(
            f"  Track B: {track_b_answer} (correct: {correct}) - {'✓' if track_b_is_correct else '✗'}"
        )

        result = {
            "index": idx,
            "nation": nation,
            "task": task,
            "correct_answer": correct,
            "track_a": {
                "response": track_a_response[:500],
                "answer": track_a_answer,
                "is_correct": track_a_is_correct,
            },
            "track_b": {
                "ocr_text": ocr_text[:1000],
                "response": track_b_response[:500],
                "answer": track_b_answer,
                "is_correct": track_b_is_correct,
            },
        }
        results.append(result)

        time.sleep(2)

    track_a_acc = track_a_correct / len(sample_indices) * 100
    track_b_acc = track_b_correct / len(sample_indices) * 100
    delta = track_b_acc - track_a_acc

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"Track A (Image-only):    {track_a_acc:.1f}% ({track_a_correct}/{len(sample_indices)})"
    )
    print(
        f"Track B (OCR-text-only): {track_b_acc:.1f}% ({track_b_correct}/{len(sample_indices)})"
    )
    print(f"Delta (B - A):           {delta:+.1f}%")
    print("-" * 60)

    if delta > 0:
        print("→ Track B > Track A: OCR/읽기가 bottleneck일 가능성")
    elif delta < 0:
        print("→ Track A > Track B: Visual grounding이 도움됨")
    else:
        print("→ Track A ≈ Track B: 비슷한 성능")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"pilot_results_{timestamp}.json"

    summary = {
        "experiment": "pilot",
        "model": MODEL_NAME,
        "sample_size": len(sample_indices),
        "timestamp": timestamp,
        "results": {
            "track_a_accuracy": track_a_acc,
            "track_b_accuracy": track_b_acc,
            "delta": delta,
        },
        "by_nation": {},
        "by_task": {},
        "details": results,
    }

    for nation in NATION_DISTRIBUTION.keys():
        nation_results = [r for r in results if r["nation"] == nation]
        if nation_results:
            a_correct = sum(1 for r in nation_results if r["track_a"]["is_correct"])
            b_correct = sum(1 for r in nation_results if r["track_b"]["is_correct"])
            summary["by_nation"][nation] = {
                "n": len(nation_results),
                "track_a": a_correct / len(nation_results) * 100,
                "track_b": b_correct / len(nation_results) * 100,
            }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")

    return summary


if __name__ == "__main__":
    random.seed(42)
    run_experiment()
