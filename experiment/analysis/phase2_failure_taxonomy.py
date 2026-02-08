#!/usr/bin/env python3
"""
Phase 2: Failure Taxonomy Analysis
Analyze why VLMs fail on hard questions using Gemini Flash for image analysis.
"""

import json
import time
import base64
from pathlib import Path
from io import BytesIO
from datasets import load_dataset
import google.generativeai as genai
from PIL import Image

GEMINI_API_KEY = "AIzaSyBAcnWVwzdnDvQwkM6ixIca8rpNqicOZcs"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Failure taxonomy categories
FAILURE_TAXONOMY = {
    "ocr_text_recognition": "OCR/text recognition failure - model cannot read text in image",
    "table_structure": "Table/chart structure comprehension failure - complex tables, graphs",
    "multi_column_layout": "Multi-column or complex layout confusion",
    "vertical_nonlatin_script": "Vertical text or non-Latin script issues (CJK, Devanagari)",
    "math_symbol_interpretation": "Mathematical formula or symbol interpretation error",
    "figure_text_alignment": "Figure-text alignment or reference problem",
    "code_switching": "Code-switching (multilingual mixing) in content",
    "pure_reasoning_knowledge": "Pure reasoning or domain knowledge gap (not vision-related)",
    "image_quality": "Low image quality, blur, or resolution issues",
    "diagram_understanding": "Complex diagram or flowchart understanding failure",
}

ANALYSIS_PROMPT = """You are an expert analyzing why Vision-Language Models (VLMs) might fail on this exam question.

This is a civil service exam question from {nation}, in the {task} domain.

Analyze the image and determine the PRIMARY visual challenge that would cause VLMs to fail.

Choose exactly ONE primary category from:
1. ocr_text_recognition - Text is hard to read (small, stylized, handwritten)
2. table_structure - Complex tables, charts, or graphs that need structural understanding
3. multi_column_layout - Multiple columns, boxes, or complex spatial arrangement
4. vertical_nonlatin_script - Vertical text direction or non-Latin scripts (Japanese, Korean, Hindi)
5. math_symbol_interpretation - Mathematical formulas, equations, or technical symbols
6. figure_text_alignment - Need to match figures/diagrams with text references
7. code_switching - Multiple languages mixed in the content
8. pure_reasoning_knowledge - No significant visual challenge; requires domain knowledge
9. image_quality - Poor image quality, blur, low resolution
10. diagram_understanding - Complex diagrams, flowcharts, circuit diagrams, etc.

Also identify up to 2 SECONDARY challenges if present.

Respond in JSON format:
{{
    "primary_category": "<category_name>",
    "secondary_categories": ["<category1>", "<category2>"] or [],
    "visual_elements": ["<element1>", "<element2>", ...],
    "difficulty_factors": "<brief explanation of what makes this hard for VLMs>",
    "language_detected": "<primary language in image>",
    "has_handwriting": true/false,
    "has_diagram": true/false,
    "has_table": true/false,
    "has_math": true/false
}}
"""


def image_to_base64(img):
    """Convert PIL Image to base64."""
    if img is None:
        return None
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def analyze_question(item, question_info, model):
    """Analyze a single question image."""
    img = item["img"]
    if img is None:
        return {
            "index": question_info["index"],
            "error": "No image",
            "primary_category": "pure_reasoning_knowledge",
            "secondary_categories": [],
            "visual_elements": [],
            "difficulty_factors": "Text-only question",
            "has_handwriting": False,
            "has_diagram": False,
            "has_table": False,
            "has_math": False,
        }

    prompt = ANALYSIS_PROMPT.format(
        nation=question_info["nation"], task=question_info["task"]
    )

    try:
        response = model.generate_content([prompt, img])
        response_text = response.text

        # Parse JSON from response
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text.strip())
        result["index"] = question_info["index"]
        result["nation"] = question_info["nation"]
        result["task"] = question_info["task"]
        result["correct_answer"] = question_info["correct_answer"]
        return result

    except json.JSONDecodeError as e:
        return {
            "index": question_info["index"],
            "nation": question_info["nation"],
            "task": question_info["task"],
            "error": f"JSON parse error: {str(e)}",
            "raw_response": response_text
            if "response_text" in dir()
            else "No response",
            "primary_category": "unknown",
            "secondary_categories": [],
        }
    except Exception as e:
        return {
            "index": question_info["index"],
            "nation": question_info["nation"],
            "task": question_info["task"],
            "error": str(e),
            "primary_category": "unknown",
            "secondary_categories": [],
        }


def main():
    # Load questions to analyze
    with open("hard_questions_for_analysis.json", "r") as f:
        analysis_data = json.load(f)

    questions = analysis_data["questions"]
    print(f"Loaded {len(questions)} questions for analysis")

    # Load dataset
    print("Loading EuraGovExam dataset...")
    ds = load_dataset("EuraGovExam/EuraGovExam", split="train")
    print(f"Dataset loaded: {len(ds)} samples")

    # Analyze each question
    results = []
    checkpoint_path = Path("failure_taxonomy_checkpoint.json")

    # Resume from checkpoint if exists
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            results = checkpoint.get("results", [])
            start_idx = len(results)
            print(f"Resuming from checkpoint: {start_idx} already done")

    for i, q in enumerate(questions[start_idx:], start=start_idx):
        print(
            f"\rAnalyzing {i+1}/{len(questions)}: index={q['index']}, {q['nation']}/{q['task']}",
            end="",
            flush=True,
        )

        item = ds[q["index"]]
        result = analyze_question(item, q, model)
        results.append(result)

        # Save checkpoint every 10 questions
        if (i + 1) % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)

        # Rate limiting - Gemini Flash has generous limits but be safe
        time.sleep(0.5)

    print(f"\n\nAnalysis complete: {len(results)} questions")

    # Save final results
    output_path = Path("failure_taxonomy_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "total_analyzed": len(results),
                "taxonomy_categories": FAILURE_TAXONOMY,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved to {output_path}")

    # Generate summary statistics
    generate_summary(results)


def generate_summary(results):
    """Generate summary statistics from results."""
    # Count primary categories
    primary_counts = {}
    secondary_counts = {}
    nation_category = {}
    task_category = {}

    valid_results = [r for r in results if r.get("primary_category") != "unknown"]

    for r in valid_results:
        primary = r.get("primary_category", "unknown")
        primary_counts[primary] = primary_counts.get(primary, 0) + 1

        for sec in r.get("secondary_categories", []):
            secondary_counts[sec] = secondary_counts.get(sec, 0) + 1

        nation = r.get("nation", "unknown")
        if nation not in nation_category:
            nation_category[nation] = {}
        nation_category[nation][primary] = nation_category[nation].get(primary, 0) + 1

        task = r.get("task", "unknown")
        if task not in task_category:
            task_category[task] = {}
        task_category[task][primary] = task_category[task].get(primary, 0) + 1

    # Count visual elements
    has_handwriting = sum(1 for r in valid_results if r.get("has_handwriting", False))
    has_diagram = sum(1 for r in valid_results if r.get("has_diagram", False))
    has_table = sum(1 for r in valid_results if r.get("has_table", False))
    has_math = sum(1 for r in valid_results if r.get("has_math", False))

    summary = {
        "total_valid": len(valid_results),
        "total_errors": len(results) - len(valid_results),
        "primary_category_distribution": dict(
            sorted(primary_counts.items(), key=lambda x: -x[1])
        ),
        "secondary_category_distribution": dict(
            sorted(secondary_counts.items(), key=lambda x: -x[1])
        ),
        "nation_category_breakdown": nation_category,
        "task_category_breakdown": task_category,
        "visual_element_counts": {
            "has_handwriting": has_handwriting,
            "has_diagram": has_diagram,
            "has_table": has_table,
            "has_math": has_math,
        },
    }

    with open("failure_taxonomy_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Failure Taxonomy Summary ===")
    print(f"Valid results: {len(valid_results)}")
    print(f"\nPrimary Category Distribution:")
    for cat, count in sorted(primary_counts.items(), key=lambda x: -x[1]):
        pct = count / len(valid_results) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print(f"\nVisual Elements:")
    print(
        f"  Has handwriting: {has_handwriting} ({has_handwriting/len(valid_results)*100:.1f}%)"
    )
    print(f"  Has diagram: {has_diagram} ({has_diagram/len(valid_results)*100:.1f}%)")
    print(f"  Has table: {has_table} ({has_table/len(valid_results)*100:.1f}%)")
    print(f"  Has math: {has_math} ({has_math/len(valid_results)*100:.1f}%)")


if __name__ == "__main__":
    main()
