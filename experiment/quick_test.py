import google.generativeai as genai
from datasets import load_dataset
import time

GEMINI_API_KEY = "AIzaSyBAcnWVwzdnDvQwkM6ixIca8rpNqicOZcs"
MODEL_NAME = "gemini-2.0-flash"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

print("Loading dataset...")
dataset = load_dataset("EuraGovExam/EuraGovExam", split="train")
print(f"Dataset loaded: {len(dataset)} items")

PROMPT = """You are solving a multiple-choice exam question shown in the image.
At the very end, provide the final answer in exactly this format:
The answer is X. (For example: The answer is B.)"""

print("\n" + "=" * 50)
print("Quick Test: 5 samples")
print("=" * 50)

correct_count = 0
for i in range(5):
    item = dataset[i * 100]
    image = item["img"]
    correct = item["correct_answer"]
    nation = item["nation"]
    task = item["task"]

    print(f"\n[{i+1}/5] {nation} - {task}")
    print(f"  Correct answer: {correct}")

    try:
        response = model.generate_content([PROMPT, image])
        answer_text = response.text

        import re

        match = re.search(r"[Tt]he answer is ([A-Ea-e])", answer_text)
        if match:
            predicted = match.group(1).upper()
        else:
            predicted = "?"

        is_correct = predicted == correct.upper()
        if is_correct:
            correct_count += 1

        print(f"  Model answer: {predicted} {'✓' if is_correct else '✗'}")

    except Exception as e:
        print(f"  Error: {e}")

    time.sleep(2)

print("\n" + "=" * 50)
print(f"Result: {correct_count}/5 correct ({correct_count/5*100:.0f}%)")
print("=" * 50)
