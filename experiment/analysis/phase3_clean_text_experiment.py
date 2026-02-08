#!/usr/bin/env python3
"""
Phase 3: Clean-Text Upper Bound Experiment
1. Sample 100 questions (stratified by nation)
2. Extract text from images using Gemini 3.0 Flash (OCR)
3. Evaluate with Gemini-2.5-Flash: Original (image) vs Clean-Text (text only)
4. Calculate Perception Gap
"""

import json
import time
import random
from pathlib import Path
from datasets import load_dataset
import google.generativeai as genai

GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

ocr_model = genai.GenerativeModel("gemini-2.0-flash")
eval_model = genai.GenerativeModel("gemini-2.5-flash")

random.seed(42)

OCR_PROMPT = """Extract ALL text from this exam question image. 
Include:
- The question text
- All answer choices (A, B, C, D, E if present)
- Any text in tables, diagrams, or figures
- Mathematical expressions (use LaTeX notation)

Output the extracted text in a clean, readable format. Preserve the structure (question, then choices).
If there are multiple languages, include all of them."""

EVAL_PROMPT_IMAGE = """You are solving a multiple choice exam question.
Look at the image carefully and select the correct answer.

Rules:
- Output ONLY a single letter (A, B, C, D, or E)
- Do not explain your reasoning
- If unsure, make your best guess

Your answer:"""

EVAL_PROMPT_TEXT = """You are solving a multiple choice exam question.
Read the question carefully and select the correct answer.

Question:
{question_text}

Rules:
- Output ONLY a single letter (A, B, C, D, or E)
- Do not explain your reasoning
- If unsure, make your best guess

Your answer:"""


def extract_text_from_image(img, model):
    """Use Gemini 3.0 Flash to OCR the image."""
    try:
        response = model.generate_content([OCR_PROMPT, img])
        return response.text.strip()
    except Exception as e:
        return f"OCR_ERROR: {str(e)}"


def evaluate_with_image(img, model):
    """Evaluate question using image input."""
    try:
        response = model.generate_content([EVAL_PROMPT_IMAGE, img])
        answer = response.text.strip().upper()
        if answer and answer[0] in "ABCDE":
            return answer[0]
        return "X"
    except Exception as e:
        return "X"


def evaluate_with_text(question_text, model):
    """Evaluate question using text-only input."""
    try:
        prompt = EVAL_PROMPT_TEXT.format(question_text=question_text)
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        if answer and answer[0] in "ABCDE":
            return answer[0]
        return "X"
    except Exception as e:
        return "X"


def sample_questions(ds, n_total=100):
    """Stratified sampling by nation."""
    nation_indices = {}
    for idx, item in enumerate(ds):
        nation = item["nation"]
        if nation not in nation_indices:
            nation_indices[nation] = []
        nation_indices[nation].append(idx)

    samples_per_nation = {
        "Japan": 30,
        "South Korea": 25,
        "India": 20,
        "EU": 15,
        "Taiwan": 10,
    }

    sampled = []
    for nation, count in samples_per_nation.items():
        if nation in nation_indices:
            indices = random.sample(
                nation_indices[nation], min(count, len(nation_indices[nation]))
            )
            for idx in indices:
                sampled.append({"index": idx, "nation": nation})

    random.shuffle(sampled)
    return sampled


def main():
    print("Loading dataset...")
    ds = load_dataset("EuraGovExam/EuraGovExam", split="train")
    print(f"Total: {len(ds)} questions")

    samples = sample_questions(ds, n_total=100)
    print(f"Sampled {len(samples)} questions")

    results = []
    checkpoint_path = Path("phase3_checkpoint.json")

    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            results = checkpoint.get("results", [])
            start_idx = len(results)
            print(f"Resuming from checkpoint: {start_idx} done")

    for i, sample in enumerate(samples[start_idx:], start=start_idx):
        idx = sample["index"]
        item = ds[idx]
        img = item["img"]
        correct = item["correct answer"].strip().upper()
        nation = item["nation"]
        task = item["task"]

        print(f"\r[{i+1}/{len(samples)}] {nation}/{task}", end="", flush=True)

        extracted_text = extract_text_from_image(img, ocr_model)
        time.sleep(0.3)

        answer_image = evaluate_with_image(img, eval_model)
        time.sleep(0.3)

        answer_text = evaluate_with_text(extracted_text, eval_model)
        time.sleep(0.3)

        result = {
            "index": idx,
            "nation": nation,
            "task": task,
            "correct_answer": correct,
            "extracted_text": extracted_text[:500]
            if not extracted_text.startswith("OCR_ERROR")
            else extracted_text,
            "answer_image": answer_image,
            "answer_text": answer_text,
            "correct_image": answer_image == correct,
            "correct_text": answer_text == correct,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"\n\nExperiment complete: {len(results)} questions")

    output_path = Path("phase3_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {"total": len(results), "results": results}, f, indent=2, ensure_ascii=False
        )

    analyze_results(results)


def analyze_results(results):
    """Analyze and print results."""

    total = len(results)
    correct_image = sum(1 for r in results if r["correct_image"])
    correct_text = sum(1 for r in results if r["correct_text"])

    acc_image = correct_image / total * 100
    acc_text = correct_text / total * 100
    perception_gap = acc_text - acc_image

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Accuracy (Image):      {acc_image:.1f}% ({correct_image}/{total})")
    print(f"Accuracy (Clean-Text): {acc_text:.1f}% ({correct_text}/{total})")
    print(f"Perception Gap:        {perception_gap:+.1f}%p")
    print()

    nation_results = {}
    for r in results:
        nation = r["nation"]
        if nation not in nation_results:
            nation_results[nation] = {"total": 0, "correct_image": 0, "correct_text": 0}
        nation_results[nation]["total"] += 1
        if r["correct_image"]:
            nation_results[nation]["correct_image"] += 1
        if r["correct_text"]:
            nation_results[nation]["correct_text"] += 1

    print("BY NATION:")
    print("-" * 60)
    print(f"{'Nation':<15} {'Image':>10} {'Text':>10} {'Gap':>10}")
    print("-" * 60)

    summary = {}
    for nation, data in sorted(nation_results.items()):
        n = data["total"]
        acc_img = data["correct_image"] / n * 100
        acc_txt = data["correct_text"] / n * 100
        gap = acc_txt - acc_img
        print(f"{nation:<15} {acc_img:>9.1f}% {acc_txt:>9.1f}% {gap:>+9.1f}%p")
        summary[nation] = {
            "n": n,
            "acc_image": round(acc_img, 1),
            "acc_text": round(acc_txt, 1),
            "perception_gap": round(gap, 1),
        }

    summary_path = Path("phase3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "overall": {
                    "n": total,
                    "acc_image": round(acc_image, 1),
                    "acc_text": round(acc_text, 1),
                    "perception_gap": round(perception_gap, 1),
                },
                "by_nation": summary,
            },
            f,
            indent=2,
        )

    print("-" * 60)
    print(f"\nResults saved to phase3_results.json and phase3_summary.json")


if __name__ == "__main__":
    main()
