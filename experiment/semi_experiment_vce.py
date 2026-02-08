"""
Semi Experiment: Visual Causal Effect with Counterfactual Images
================================================================
50 samples, gemini-3-flash-preview model
Tests: 3-Track + Image Shuffle + Image Blur
"""

import os
import re
import json
import time
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from io import BytesIO

import google.generativeai as genai
from datasets import load_dataset
from PIL import Image, ImageFilter
import numpy as np

GEMINI_API_KEY = "AIzaSyBAcnWVwzdnDvQwkM6ixIca8rpNqicOZcs"
MODEL_NAME = "gemini-3-flash-preview"
DATASET_NAME = "EuraGovExam/EuraGovExam"

SAMPLE_SIZE = 50
NATION_DISTRIBUTION = {
    "South Korea": 10,
    "Japan": 10,
    "EU": 10,
    "India": 10,
    "Taiwan": 10,
}

API_DELAY_SECONDS = 1.5
MAX_RETRIES = 3
SAVE_INTERVAL = 10

OUTPUT_DIR = Path(__file__).parent / "results"

PROMPT_IMAGE_ONLY = """You are solving a multiple-choice exam question shown in the image.
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

Output only the extracted text, nothing else."""

PROMPT_TEXT_ONLY = """You are solving a multiple-choice exam question.
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

PROMPT_MULTIMODAL = """You are solving a multiple-choice exam question.
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
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                wait_time = 30 * (attempt + 1)
                print(f"    [Rate limit] Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries:
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
            print(f"  Warning: Only {len(available)} samples for {nation}")
        sampled_indices.extend(selected)

    random.shuffle(sampled_indices)
    return sampled_indices


def blur_image(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def create_blank_image(size: tuple) -> Image.Image:
    return Image.new("RGB", size, color=(128, 128, 128))


def run_experiment():
    print("=" * 70)
    print("Semi Experiment: VCE with Counterfactual Images")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Sample Size: {SAMPLE_SIZE}")
    print(f"Distribution: {NATION_DISTRIBUTION}")
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
    output_file = OUTPUT_DIR / f"semi_vce_{timestamp}.json"

    print(f"[4/4] Running experiment...")
    print(f"       Output: {output_file}")
    print("-" * 70)

    results = []
    stats = {
        "image_only": {"correct": 0, "total": 0},
        "text_only": {"correct": 0, "total": 0},
        "multimodal": {"correct": 0, "total": 0},
        "shuffled_image": {"correct": 0, "total": 0},
        "blurred_image": {"correct": 0, "total": 0},
        "blank_image": {"correct": 0, "total": 0},
    }

    all_images = [dataset[idx]["img"] for idx in sample_indices]

    for i, idx in enumerate(sample_indices):
        item = dataset[idx]
        image = item["img"]
        correct = item["correct answer"].strip().upper()
        nation = item["nation"]
        task = item["task"]

        print(
            f"\n[{i+1}/{len(sample_indices)}] Nation={nation}, Task={task}, Correct={correct}"
        )

        result = {
            "index": idx,
            "nation": nation,
            "task": task,
            "correct_answer": correct,
        }

        # --- Track A: Image-only ---
        resp_a = api_call_with_retry(model, [PROMPT_IMAGE_ONLY, image])
        ans_a = extract_answer(resp_a)
        correct_a = ans_a == correct
        stats["image_only"]["total"] += 1
        if correct_a:
            stats["image_only"]["correct"] += 1
        result["image_only"] = {"answer": ans_a, "correct": correct_a}
        print(f"  Image-only:    {ans_a} {'✓' if correct_a else '✗'}")
        time.sleep(API_DELAY_SECONDS)

        # --- OCR ---
        ocr_text = api_call_with_retry(model, [PROMPT_OCR, image])
        ocr_success = not ocr_text.startswith("ERROR")
        result["ocr_text"] = ocr_text[:500] if len(ocr_text) > 500 else ocr_text
        print(
            f"  OCR:           {'OK' if ocr_success else 'FAIL'} ({len(ocr_text)} chars)"
        )
        time.sleep(API_DELAY_SECONDS)

        # --- Track B: Text-only ---
        prompt_b = PROMPT_TEXT_ONLY.format(ocr_text=ocr_text)
        resp_b = api_call_with_retry(model, prompt_b)
        ans_b = extract_answer(resp_b)
        correct_b = ans_b == correct
        stats["text_only"]["total"] += 1
        if correct_b:
            stats["text_only"]["correct"] += 1
        result["text_only"] = {"answer": ans_b, "correct": correct_b}
        print(f"  Text-only:     {ans_b} {'✓' if correct_b else '✗'}")
        time.sleep(API_DELAY_SECONDS)

        # --- Track C: Multimodal ---
        prompt_c = PROMPT_MULTIMODAL.format(ocr_text=ocr_text)
        resp_c = api_call_with_retry(model, [prompt_c, image])
        ans_c = extract_answer(resp_c)
        correct_c = ans_c == correct
        stats["multimodal"]["total"] += 1
        if correct_c:
            stats["multimodal"]["correct"] += 1
        result["multimodal"] = {"answer": ans_c, "correct": correct_c}
        print(f"  Multimodal:    {ans_c} {'✓' if correct_c else '✗'}")
        time.sleep(API_DELAY_SECONDS)

        # --- Counterfactual 1: Shuffled Image ---
        other_indices = [j for j in range(len(all_images)) if j != i]
        shuffle_idx = random.choice(other_indices)
        shuffled_image = all_images[shuffle_idx]

        prompt_shuffle = PROMPT_MULTIMODAL.format(ocr_text=ocr_text)
        resp_shuffle = api_call_with_retry(model, [prompt_shuffle, shuffled_image])
        ans_shuffle = extract_answer(resp_shuffle)
        correct_shuffle = ans_shuffle == correct
        stats["shuffled_image"]["total"] += 1
        if correct_shuffle:
            stats["shuffled_image"]["correct"] += 1
        result["shuffled_image"] = {
            "answer": ans_shuffle,
            "correct": correct_shuffle,
            "shuffled_from": sample_indices[shuffle_idx],
        }
        print(f"  Shuffled:      {ans_shuffle} {'✓' if correct_shuffle else '✗'}")
        time.sleep(API_DELAY_SECONDS)

        # --- Counterfactual 2: Blurred Image (sigma=5) ---
        blurred_image = blur_image(image, sigma=5)

        prompt_blur = PROMPT_MULTIMODAL.format(ocr_text=ocr_text)
        resp_blur = api_call_with_retry(model, [prompt_blur, blurred_image])
        ans_blur = extract_answer(resp_blur)
        correct_blur = ans_blur == correct
        stats["blurred_image"]["total"] += 1
        if correct_blur:
            stats["blurred_image"]["correct"] += 1
        result["blurred_image"] = {"answer": ans_blur, "correct": correct_blur}
        print(f"  Blurred:       {ans_blur} {'✓' if correct_blur else '✗'}")
        time.sleep(API_DELAY_SECONDS)

        # --- Counterfactual 3: Blank Image ---
        blank_image = create_blank_image(image.size)

        prompt_blank = PROMPT_MULTIMODAL.format(ocr_text=ocr_text)
        resp_blank = api_call_with_retry(model, [prompt_blank, blank_image])
        ans_blank = extract_answer(resp_blank)
        correct_blank = ans_blank == correct
        stats["blank_image"]["total"] += 1
        if correct_blank:
            stats["blank_image"]["correct"] += 1
        result["blank_image"] = {"answer": ans_blank, "correct": correct_blank}
        print(f"  Blank:         {ans_blank} {'✓' if correct_blank else '✗'}")

        # VCE calculations
        result["vce_text_vs_multimodal"] = int(correct_b) - int(correct_c)
        result["vce_text_vs_shuffled"] = int(correct_b) - int(correct_shuffle)
        result["vce_text_vs_blurred"] = int(correct_b) - int(correct_blur)
        result["vce_text_vs_blank"] = int(correct_b) - int(correct_blank)

        results.append(result)

        if (i + 1) % SAVE_INTERVAL == 0:
            save_results(results, stats, output_file, sample_indices, i + 1)
            print(f"\n  [SAVED] {i+1} samples")

        time.sleep(API_DELAY_SECONDS)

    save_results(results, stats, output_file, sample_indices, len(sample_indices))
    print_final_results(stats, results)

    print(f"\nResults saved to: {output_file}")
    return results


def save_results(results, stats, output_file, sample_indices, completed):
    n = completed
    summary = {
        "experiment": "semi-vce-counterfactual",
        "model": MODEL_NAME,
        "sample_size": n,
        "total_planned": len(sample_indices),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": "in_progress" if n < len(sample_indices) else "completed",
        "accuracy": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": v["correct"] / v["total"] * 100 if v["total"] > 0 else 0,
            }
            for k, v in stats.items()
        },
        "details": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_final_results(stats, results):
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Condition':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for condition, s in stats.items():
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"{condition:<20} {s['correct']:>8} {s['total']:>8} {acc:>9.1f}%")

    print("\n[VCE Analysis]")
    vce_text_multi = sum(r["vce_text_vs_multimodal"] for r in results)
    vce_text_shuffle = sum(r["vce_text_vs_shuffled"] for r in results)
    vce_text_blur = sum(r["vce_text_vs_blurred"] for r in results)
    vce_text_blank = sum(r["vce_text_vs_blank"] for r in results)

    print(f"  Sum VCE (Text vs Multimodal):  {vce_text_multi:+d}")
    print(f"  Sum VCE (Text vs Shuffled):    {vce_text_shuffle:+d}")
    print(f"  Sum VCE (Text vs Blurred):     {vce_text_blur:+d}")
    print(f"  Sum VCE (Text vs Blank):       {vce_text_blank:+d}")

    print("\n[Visual Susceptibility - Answer Changes]")
    answer_changes_shuffle = sum(
        1 for r in results if r["multimodal"]["answer"] != r["shuffled_image"]["answer"]
    )
    answer_changes_blur = sum(
        1 for r in results if r["multimodal"]["answer"] != r["blurred_image"]["answer"]
    )
    answer_changes_blank = sum(
        1 for r in results if r["multimodal"]["answer"] != r["blank_image"]["answer"]
    )

    print(
        f"  Answers changed (Original vs Shuffled): {answer_changes_shuffle}/{len(results)}"
    )
    print(
        f"  Answers changed (Original vs Blurred):  {answer_changes_blur}/{len(results)}"
    )
    print(
        f"  Answers changed (Original vs Blank):    {answer_changes_blank}/{len(results)}"
    )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_experiment()
