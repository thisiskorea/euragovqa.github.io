"""
Prompt Routing Experiment for VCE Paper
========================================

Hypothesis: If Visual Noise is caused by Fusion Interference, then
instructing the model to "ignore the image" should recover performance.

Experiment Design:
- Take the 13 Visual Noise cases (where Text-only > Multimodal)
- Run with modified prompts that instruct image-ignoring
- Compare: Multimodal (baseline) vs Multimodal + "ignore image" prompt

Expected Results:
- If performance improves → Fusion Interference confirmed
- If no change → Something else is causing the degradation
"""

import json
import os
import time
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from datasets import load_dataset
from PIL import Image
import io
import base64

RESULTS_DIR = Path(__file__).parent / "results"
VCE_ANALYSIS_FILE = RESULTS_DIR / "vce_analysis.json"
OUTPUT_FILE = RESULTS_DIR / "prompt_routing_results.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

PROMPTS = {
    "baseline": """Answer this government exam question. 
Choose the single best answer from the given options.
Respond with ONLY the letter/number of your answer (e.g., "A" or "1").""",
    "ignore_image": """Answer this government exam question.
IMPORTANT: The image may contain irrelevant visual noise. Focus primarily on the TEXT content.
If the image doesn't add meaningful information, ignore it and answer based on text alone.
Choose the single best answer from the given options.
Respond with ONLY the letter/number of your answer (e.g., "A" or "1").""",
    "text_focus": """Answer this government exam question.
This is a TEXT-BASED question. Any image is purely supplementary.
Base your answer primarily on the written text content.
Choose the single best answer from the given options.
Respond with ONLY the letter/number of your answer (e.g., "A" or "1").""",
    "explicit_ignore": """Answer this government exam question.
INSTRUCTION: Completely IGNORE any images. Treat this as a text-only question.
Choose the single best answer from the given options.
Respond with ONLY the letter/number of your answer (e.g., "A" or "1").""",
}


def load_visual_noise_cases():
    """Load the 13 Visual Noise cases from VCE analysis."""
    with open(VCE_ANALYSIS_FILE, "r") as f:
        data = json.load(f)
    return data["noise_cases"]


def load_dataset_sample(index: int):
    """Load a specific sample from the EuraGovExam dataset."""
    dataset = load_dataset("jaesung9/EuraGovExam", split="test", streaming=True)
    for i, sample in enumerate(dataset):
        if i == index:
            return sample
    return None


def call_gemini_multimodal(
    question_text: str, image: Image.Image, prompt_type: str
) -> Optional[str]:
    """Call Gemini with multimodal input and specified prompt."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = PROMPTS[prompt_type]

    full_prompt = f"{prompt}\n\nQuestion:\n{question_text}"

    try:
        response = model.generate_content(
            [full_prompt, image],
            generation_config=genai.GenerationConfig(
                max_output_tokens=50, temperature=0.0
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return None


def extract_answer(response: str) -> str:
    """Extract the answer letter/number from response."""
    if not response:
        return ""
    response = response.upper().strip()
    for char in response:
        if char in "ABCDE12345":
            return char
    return response[:1] if response else ""


def run_experiment():
    """Run the prompt routing experiment on Visual Noise cases."""
    print("=" * 70)
    print("Prompt Routing Experiment for Visual Noise Cases")
    print("=" * 70)

    noise_cases = load_visual_noise_cases()
    print(f"\nLoaded {len(noise_cases)} Visual Noise cases")

    results = {
        "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": len(noise_cases),
        "prompt_types": list(PROMPTS.keys()),
        "cases": [],
        "summary": {},
    }

    correct_counts = {pt: 0 for pt in PROMPTS.keys()}

    for i, case in enumerate(noise_cases):
        print(
            f"\n[{i+1}/{len(noise_cases)}] Processing index {case['index']} ({case['nation']}, {case['task']})"
        )

        sample = load_dataset_sample(case["index"])
        if sample is None:
            print(f"  Failed to load sample {case['index']}")
            continue

        image = sample.get("image")
        if image is None:
            print(f"  No image found for sample {case['index']}")
            continue

        question_text = sample.get("question", "")
        if isinstance(question_text, list):
            question_text = question_text[0] if question_text else ""

        correct_answer = case["correct_answer"]

        case_result = {
            "index": case["index"],
            "nation": case["nation"],
            "task": case["task"],
            "correct_answer": correct_answer,
            "responses": {},
        }

        for prompt_type in PROMPTS.keys():
            print(f"  Testing {prompt_type}...", end=" ")

            response = call_gemini_multimodal(question_text, image, prompt_type)
            extracted = extract_answer(response or "")
            is_correct = extracted.upper() == correct_answer.upper()

            case_result["responses"][prompt_type] = {
                "raw_response": response,
                "extracted_answer": extracted,
                "is_correct": is_correct,
            }

            if is_correct:
                correct_counts[prompt_type] += 1
                print("✓")
            else:
                print(f"✗ (got {extracted}, expected {correct_answer})")

            time.sleep(1)

        results["cases"].append(case_result)

    total = len(results["cases"])
    for prompt_type in PROMPTS.keys():
        results["summary"][prompt_type] = {
            "correct": correct_counts[prompt_type],
            "total": total,
            "accuracy": correct_counts[prompt_type] / total if total > 0 else 0,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"\n{'Prompt Type':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for prompt_type, stats in results["summary"].items():
        print(
            f"{prompt_type:<20} {stats['correct']:>8} {stats['total']:>8} {stats['accuracy']*100:>9.1f}%"
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("Interpretation")
    print("=" * 70)

    baseline_acc = results["summary"]["baseline"]["accuracy"]
    best_prompt = max(PROMPTS.keys(), key=lambda x: results["summary"][x]["accuracy"])
    best_acc = results["summary"][best_prompt]["accuracy"]

    improvement = best_acc - baseline_acc

    if improvement > 0.1:
        print(f"""
STRONG EVIDENCE for Fusion Interference:
- Baseline (standard prompt): {baseline_acc*100:.1f}%
- Best prompt ({best_prompt}): {best_acc*100:.1f}%
- Improvement: +{improvement*100:.1f}%

The "ignore image" prompt recovered significant performance, confirming
that the model was being misled by visual information it should have ignored.
""")
    elif improvement > 0:
        print(f"""
MODERATE EVIDENCE for Fusion Interference:
- Baseline: {baseline_acc*100:.1f}%
- Best: {best_acc*100:.1f}% ({best_prompt})
- Improvement: +{improvement*100:.1f}%

Some improvement suggests partial Fusion Interference.
""")
    else:
        print(f"""
WEAK/NO EVIDENCE for Fusion Interference via prompt routing:
- All prompt types performed similarly
- The degradation may be due to inherent model limitations
- Alternative interventions (e.g., image removal) may be needed
""")

    return results


if __name__ == "__main__":
    run_experiment()
