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
from PIL import Image, ImageFilter

GEMINI_API_KEY = ""
MODEL_NAME = "gemini-3-flash-preview"
DATASET_NAME = "EuraGovExam/EuraGovExam"

SAMPLE_SIZE = 10
NATION_DISTRIBUTION = {"South Korea": 2, "Japan": 2, "EU": 2, "India": 2, "Taiwan": 2}

API_DELAY = 1.0
OUTPUT_DIR = Path(__file__).parent / "results"

PROMPT_IMAGE = """Solve this multiple-choice question from the image.
At the end, answer in format: The answer is X. (X = A, B, C, D, or E)"""

PROMPT_OCR = """Extract ALL text from this image exactly as written."""

PROMPT_TEXT = """Solve this multiple-choice question based only on the text below:
---
{ocr_text}
---
At the end, answer in format: The answer is X. (X = A, B, C, D, or E)"""

PROMPT_MULTI = """Solve this question using BOTH the image AND this extracted text:
---
{ocr_text}
---
At the end, answer in format: The answer is X. (X = A, B, C, D, or E)"""


def setup():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def extract_answer(text):
    if not text:
        return "INVALID"
    match = re.search(r"[Tt]he answer is\s*[:\s]*([A-Ea-e])", text)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else "INVALID"


def call_api(model, contents, retries=3):
    for attempt in range(retries):
        try:
            return model.generate_content(contents).text
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Retry {attempt+1}, waiting {wait}s: {str(e)[:40]}")
                time.sleep(wait)
    return "ERROR"


def stratified_sample(dataset, counts):
    by_nation = defaultdict(list)
    for idx, item in enumerate(dataset):
        if item["nation"] in counts:
            by_nation[item["nation"]].append(idx)

    indices = []
    for nation, count in counts.items():
        available = by_nation[nation]
        selected = random.sample(available, min(count, len(available)))
        indices.extend(selected)
    random.shuffle(indices)
    return indices


def blur_image(img, sigma=5):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def run():
    print("=" * 60)
    print(f"Quick VCE Test - {SAMPLE_SIZE} samples, {MODEL_NAME}")
    print("=" * 60)

    model = setup()
    dataset = load_dataset(DATASET_NAME, split="train")
    indices = stratified_sample(dataset, NATION_DISTRIBUTION)

    all_images = [dataset[i]["img"] for i in indices]

    results = []
    stats = {
        k: {"correct": 0, "total": 0}
        for k in ["image", "text", "multi", "shuffle", "blur"]
    }

    for i, idx in enumerate(indices):
        item = dataset[idx]
        img, correct = item["img"], item["correct answer"].strip().upper()
        nation, task = item["nation"], item["task"]

        print(f"\n[{i+1}/{len(indices)}] {nation}/{task} | Correct={correct}")

        r = {"idx": idx, "nation": nation, "task": task, "correct": correct}

        resp = call_api(model, [PROMPT_IMAGE, img])
        ans = extract_answer(resp)
        r["image"] = {"ans": ans, "ok": ans == correct}
        stats["image"]["total"] += 1
        stats["image"]["correct"] += int(ans == correct)
        print(f"  Image:   {ans} {'✓' if ans == correct else '✗'}")
        time.sleep(API_DELAY)

        ocr = call_api(model, [PROMPT_OCR, img])
        r["ocr"] = ocr[:200]
        time.sleep(API_DELAY)

        resp = call_api(model, PROMPT_TEXT.format(ocr_text=ocr))
        ans = extract_answer(resp)
        r["text"] = {"ans": ans, "ok": ans == correct}
        stats["text"]["total"] += 1
        stats["text"]["correct"] += int(ans == correct)
        print(f"  Text:    {ans} {'✓' if ans == correct else '✗'}")
        time.sleep(API_DELAY)

        resp = call_api(model, [PROMPT_MULTI.format(ocr_text=ocr), img])
        ans = extract_answer(resp)
        r["multi"] = {"ans": ans, "ok": ans == correct}
        stats["multi"]["total"] += 1
        stats["multi"]["correct"] += int(ans == correct)
        print(f"  Multi:   {ans} {'✓' if ans == correct else '✗'}")
        time.sleep(API_DELAY)

        other_idx = random.choice([j for j in range(len(all_images)) if j != i])
        resp = call_api(
            model, [PROMPT_MULTI.format(ocr_text=ocr), all_images[other_idx]]
        )
        ans = extract_answer(resp)
        r["shuffle"] = {"ans": ans, "ok": ans == correct}
        stats["shuffle"]["total"] += 1
        stats["shuffle"]["correct"] += int(ans == correct)
        print(f"  Shuffle: {ans} {'✓' if ans == correct else '✗'}")
        time.sleep(API_DELAY)

        blurred = blur_image(img)
        resp = call_api(model, [PROMPT_MULTI.format(ocr_text=ocr), blurred])
        ans = extract_answer(resp)
        r["blur"] = {"ans": ans, "ok": ans == correct}
        stats["blur"]["total"] += 1
        stats["blur"]["correct"] += int(ans == correct)
        print(f"  Blur:    {ans} {'✓' if ans == correct else '✗'}")

        r["vce_text_multi"] = int(r["text"]["ok"]) - int(r["multi"]["ok"])
        r["vce_text_shuffle"] = int(r["text"]["ok"]) - int(r["shuffle"]["ok"])
        r["vce_text_blur"] = int(r["text"]["ok"]) - int(r["blur"]["ok"])

        results.append(r)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in stats.items():
        acc = v["correct"] / v["total"] * 100 if v["total"] else 0
        print(f"  {k:10s}: {v['correct']}/{v['total']} = {acc:.1f}%")

    print("\n[VCE Summary]")
    print(f"  Text vs Multi:   {sum(r['vce_text_multi'] for r in results):+d}")
    print(f"  Text vs Shuffle: {sum(r['vce_text_shuffle'] for r in results):+d}")
    print(f"  Text vs Blur:    {sum(r['vce_text_blur'] for r in results):+d}")

    ans_change_shuffle = sum(
        1 for r in results if r["multi"]["ans"] != r["shuffle"]["ans"]
    )
    ans_change_blur = sum(1 for r in results if r["multi"]["ans"] != r["blur"]["ans"])
    print(f"\n[Answer Changes]")
    print(f"  Multi→Shuffle: {ans_change_shuffle}/{len(results)}")
    print(f"  Multi→Blur:    {ans_change_blur}/{len(results)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / f"quick_vce_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump({"stats": stats, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    random.seed(42)
    run()
