import re
import json
import time
import random
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from datasets import load_dataset
from PIL import ImageFilter

GEMINI_API_KEY = ""
MODEL_NAME = "gemini-3-flash-preview"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

print("Loading dataset...")
dataset = load_dataset("EuraGovExam/EuraGovExam", split="train")

random.seed(42)
test_indices = random.sample(range(len(dataset)), 5)


def extract_answer(text):
    if not text:
        return "X"
    m = re.search(r"[Tt]he answer is\s*[:\s]*([A-Ea-e])", text)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else "X"


def call(contents):
    try:
        return model.generate_content(contents).text
    except Exception as e:
        print(f"  Error: {e}")
        return "ERROR"


results = []
print(f"\nTesting {len(test_indices)} samples with {MODEL_NAME}")
print("=" * 50)

for i, idx in enumerate(test_indices):
    item = dataset[idx]
    img = item["img"]
    correct = item["correct answer"].strip().upper()
    nation = item["nation"]

    print(f"\n[{i+1}/5] {nation} | Correct={correct}")

    r = {"idx": idx, "nation": nation, "correct": correct}

    resp = call([img, "Solve this exam question. Answer format: The answer is X."])
    ans = extract_answer(resp)
    r["image"] = ans
    print(f"  Image-only: {ans} {'✓' if ans == correct else '✗'}")
    time.sleep(1)

    ocr = call([img, "Extract all text from this image."])
    r["ocr"] = ocr[:100] if ocr else ""
    time.sleep(1)

    resp = call(f"Solve based on text only:\n{ocr}\nAnswer format: The answer is X.")
    ans = extract_answer(resp)
    r["text"] = ans
    print(f"  Text-only:  {ans} {'✓' if ans == correct else '✗'}")
    time.sleep(1)

    resp = call(
        [img, f"Use both image and text:\n{ocr}\nAnswer format: The answer is X."]
    )
    ans = extract_answer(resp)
    r["multi"] = ans
    print(f"  Multimodal: {ans} {'✓' if ans == correct else '✗'}")
    time.sleep(1)

    blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
    resp = call(
        [blurred, f"Use image and text:\n{ocr}\nAnswer format: The answer is X."]
    )
    ans = extract_answer(resp)
    r["blur"] = ans
    print(f"  Blurred:    {ans} {'✓' if ans == correct else '✗'}")

    r["vce"] = int(r["text"] == correct) - int(r["multi"] == correct)
    results.append(r)
    time.sleep(1)

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

for cond in ["image", "text", "multi", "blur"]:
    correct_count = sum(1 for r in results if r[cond] == r["correct"])
    print(f"  {cond:10s}: {correct_count}/5 = {correct_count*20}%")

vce_sum = sum(r["vce"] for r in results)
print(f"\n  Total VCE (Text-Multi): {vce_sum:+d}")

ans_changes = sum(1 for r in results if r["multi"] != r["blur"])
print(f"  Answer changes (Multi→Blur): {ans_changes}/5")

out = (
    Path(__file__).parent
    / "results"
    / f"mini_vce_{datetime.now().strftime('%H%M%S')}.json"
)
with open(out, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out}")
