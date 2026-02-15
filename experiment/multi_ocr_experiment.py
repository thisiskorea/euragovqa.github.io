"""
Multi-OCR Engine Comparison
===========================
Compare Gemini OCR vs EasyOCR on EuraGovExam samples.
"""

import json
import time
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import re
import io

import google.generativeai as genai
from datasets import load_dataset
import easyocr

GEMINI_API_KEY = ""
MODEL_NAME = "gemini-2.0-flash"
DATASET_NAME = "EuraGovExam/EuraGovExam"

SAMPLE_SIZE = 30
NATION_DISTRIBUTION = {
    "South Korea": 6,
    "Japan": 6,
    "EU": 6,
    "India": 6,
    "Taiwan": 6,
}

API_DELAY = 2.0
OUTPUT_DIR = Path(__file__).parent / "results"

PROMPT_OCR = """Extract ALL text from this image exactly as written.
Preserve the structure, including:
- Question number and text
- All answer options (A, B, C, D, E) with their full text
- Any tables, diagrams descriptions, or additional context
- Maintain the original formatting and line breaks where meaningful

Output only the extracted text, nothing else."""

PROMPT_SOLVE = """You are solving a multiple-choice exam question.
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


def run_gemini_ocr(model, image) -> str:
    try:
        response = model.generate_content([PROMPT_OCR, image])
        return response.text
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_easyocr(reader, image) -> str:
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        results = reader.readtext(img_bytes.getvalue())
        text_lines = [r[1] for r in results]
        return "\n".join(text_lines)
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_multi_ocr_experiment():
    print("=" * 70)
    print("Multi-OCR Engine Comparison Experiment")
    print("=" * 70)
    print(f"OCR Engines: Gemini, EasyOCR")
    print(f"Sample Size: {SAMPLE_SIZE}")
    print("=" * 70)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    print("\n[1/4] Loading EasyOCR (multi-language)...")
    reader = easyocr.Reader(["en", "ko", "ja", "ch_sim", "de", "fr"], gpu=False)
    print("       EasyOCR loaded")

    print("[2/4] Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"       Loaded {len(dataset)} items")

    sample_indices = stratified_sample(dataset, NATION_DISTRIBUTION)
    print(f"[3/4] Sampled {len(sample_indices)} items")

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"multi_ocr_{timestamp}.json"

    results = {
        "experiment": "multi-ocr-comparison",
        "ocr_engines": ["gemini", "easyocr"],
        "sample_size": len(sample_indices),
        "timestamp": timestamp,
        "summary": {},
        "details": [],
    }

    stats = {
        "gemini": {"correct": 0, "total": 0, "ocr_lengths": []},
        "easyocr": {"correct": 0, "total": 0, "ocr_lengths": []},
    }

    print(f"[4/4] Running OCR comparison...")
    print("-" * 70)

    for i, idx in enumerate(sample_indices):
        item = dataset[idx]
        image = item["img"]
        correct = item["correct answer"].strip().upper()
        nation = item["nation"]

        print(f"\n[{i+1}/{len(sample_indices)}] {nation} (correct: {correct})")

        gemini_ocr = run_gemini_ocr(model, image)
        stats["gemini"]["ocr_lengths"].append(len(gemini_ocr))
        print(f"  Gemini OCR: {len(gemini_ocr)} chars")

        time.sleep(API_DELAY)

        easy_ocr = run_easyocr(reader, image)
        stats["easyocr"]["ocr_lengths"].append(len(easy_ocr))
        print(f"  EasyOCR: {len(easy_ocr)} chars")

        gemini_prompt = PROMPT_SOLVE.format(ocr_text=gemini_ocr)
        gemini_response = model.generate_content(gemini_prompt)
        gemini_answer = extract_answer(gemini_response.text)
        gemini_correct = gemini_answer == correct
        stats["gemini"]["total"] += 1
        if gemini_correct:
            stats["gemini"]["correct"] += 1
        print(
            f"  Gemini-based answer: {gemini_answer} {'O' if gemini_correct else 'X'}"
        )

        time.sleep(API_DELAY)

        easy_prompt = PROMPT_SOLVE.format(ocr_text=easy_ocr)
        easy_response = model.generate_content(easy_prompt)
        easy_answer = extract_answer(easy_response.text)
        easy_correct = easy_answer == correct
        stats["easyocr"]["total"] += 1
        if easy_correct:
            stats["easyocr"]["correct"] += 1
        print(f"  EasyOCR-based answer: {easy_answer} {'O' if easy_correct else 'X'}")

        results["details"].append(
            {
                "index": idx,
                "nation": nation,
                "task": item["task"],
                "correct_answer": correct,
                "gemini_ocr": gemini_ocr[:500],
                "easyocr": easy_ocr[:500],
                "gemini_answer": gemini_answer,
                "gemini_correct": gemini_correct,
                "easyocr_answer": easy_answer,
                "easyocr_correct": easy_correct,
            }
        )

        time.sleep(API_DELAY)

    import numpy as np

    results["summary"] = {
        "gemini": {
            "accuracy": round(
                stats["gemini"]["correct"] / stats["gemini"]["total"] * 100, 2
            ),
            "avg_ocr_length": round(np.mean(stats["gemini"]["ocr_lengths"]), 1),
            "correct": stats["gemini"]["correct"],
            "total": stats["gemini"]["total"],
        },
        "easyocr": {
            "accuracy": round(
                stats["easyocr"]["correct"] / stats["easyocr"]["total"] * 100, 2
            ),
            "avg_ocr_length": round(np.mean(stats["easyocr"]["ocr_lengths"]), 1),
            "correct": stats["easyocr"]["correct"],
            "total": stats["easyocr"]["total"],
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'OCR Engine':<15} {'Accuracy':>10} {'Avg Length':>12}")
    print("-" * 40)
    for engine in ["gemini", "easyocr"]:
        s = results["summary"][engine]
        print(f"{engine:<15} {s['accuracy']:>9.1f}% {s['avg_ocr_length']:>11.1f}")

    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    random.seed(42)
    run_multi_ocr_experiment()
