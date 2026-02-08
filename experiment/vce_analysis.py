"""
VCE (Visual Causal Effect) Analysis
====================================
Analyze existing 200-sample data to compute Visual Causal Effect metrics.

VCE = Acc(Text-only) - Acc(Image+Text)
- VCE > 0: Visual Noise (images hurt performance)
- VCE < 0: Visual Benefit (images help performance)
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

RESULTS_FILE = (
    Path(__file__).parent / "results" / "large_scale_20260120_170800_fixed.json"
)


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_vce_per_sample(details):
    """Compute VCE for each sample."""
    vce_data = []
    for item in details:
        # Track A: Image-only
        # Track B: Text-only (OCR)
        # Track C: Image + Text (Multimodal)

        track_a_correct = item["track_a"]["is_correct"]
        track_b_correct = item["track_b"]["is_correct"]
        track_c_correct = item["track_c"]["is_correct"]

        # VCE = Text-only - Multimodal (Track B - Track C)
        # Positive VCE = Text-only better = Visual Noise
        vce_b_vs_c = int(track_b_correct) - int(track_c_correct)

        # Also compute: Image-only vs Multimodal (Track A - Track C)
        vce_a_vs_c = int(track_a_correct) - int(track_c_correct)

        # Text-only vs Image-only (Track B - Track A)
        vce_b_vs_a = int(track_b_correct) - int(track_a_correct)

        vce_data.append(
            {
                "index": item["index"],
                "nation": item["nation"],
                "task": item["task"],
                "correct_answer": item["correct_answer"],
                "track_a": track_a_correct,
                "track_b": track_b_correct,
                "track_c": track_c_correct,
                "vce_text_vs_multimodal": vce_b_vs_c,  # B - C
                "vce_image_vs_multimodal": vce_a_vs_c,  # A - C
                "vce_text_vs_image": vce_b_vs_a,  # B - A
            }
        )

    return vce_data


def analyze_by_nation(vce_data):
    """Analyze VCE patterns by nation."""
    nation_stats = defaultdict(
        lambda: {
            "n": 0,
            "track_a_correct": 0,
            "track_b_correct": 0,
            "track_c_correct": 0,
            "vce_text_vs_multimodal": [],
            "vce_text_vs_image": [],
            "visual_noise_cases": 0,  # B correct, C wrong
            "visual_benefit_cases": 0,  # C correct, B wrong
        }
    )

    for item in vce_data:
        nation = item["nation"]
        nation_stats[nation]["n"] += 1
        nation_stats[nation]["track_a_correct"] += int(item["track_a"])
        nation_stats[nation]["track_b_correct"] += int(item["track_b"])
        nation_stats[nation]["track_c_correct"] += int(item["track_c"])
        nation_stats[nation]["vce_text_vs_multimodal"].append(
            item["vce_text_vs_multimodal"]
        )
        nation_stats[nation]["vce_text_vs_image"].append(item["vce_text_vs_image"])

        # Visual Noise: Text correct, Multimodal wrong
        if item["track_b"] and not item["track_c"]:
            nation_stats[nation]["visual_noise_cases"] += 1
        # Visual Benefit: Multimodal correct, Text wrong
        if item["track_c"] and not item["track_b"]:
            nation_stats[nation]["visual_benefit_cases"] += 1

    return nation_stats


def analyze_by_task(vce_data):
    """Analyze VCE patterns by task/domain."""
    task_stats = defaultdict(
        lambda: {
            "n": 0,
            "track_a_correct": 0,
            "track_b_correct": 0,
            "track_c_correct": 0,
            "visual_noise_cases": 0,
            "visual_benefit_cases": 0,
        }
    )

    for item in vce_data:
        task = item["task"]
        task_stats[task]["n"] += 1
        task_stats[task]["track_a_correct"] += int(item["track_a"])
        task_stats[task]["track_b_correct"] += int(item["track_b"])
        task_stats[task]["track_c_correct"] += int(item["track_c"])

        if item["track_b"] and not item["track_c"]:
            task_stats[task]["visual_noise_cases"] += 1
        if item["track_c"] and not item["track_b"]:
            task_stats[task]["visual_benefit_cases"] += 1

    return task_stats


def find_visual_noise_cases(vce_data):
    """Find specific cases where adding image hurt performance."""
    noise_cases = []
    for item in vce_data:
        # Text correct but Multimodal wrong
        if item["track_b"] and not item["track_c"]:
            noise_cases.append(item)
    return noise_cases


def find_visual_benefit_cases(vce_data):
    """Find specific cases where adding image helped performance."""
    benefit_cases = []
    for item in vce_data:
        # Multimodal correct but Text wrong
        if item["track_c"] and not item["track_b"]:
            benefit_cases.append(item)
    return benefit_cases


def print_analysis(data, nation_stats, task_stats, noise_cases, benefit_cases):
    """Print comprehensive VCE analysis."""

    print("=" * 70)
    print("VCE (Visual Causal Effect) Analysis")
    print("=" * 70)
    print(f"\nTotal samples: {len(data)}")

    # Overall statistics
    total_a = sum(int(d["track_a"]) for d in data)
    total_b = sum(int(d["track_b"]) for d in data)
    total_c = sum(int(d["track_c"]) for d in data)

    print(f"\n[Overall Accuracy]")
    print(
        f"  Track A (Image-only):  {total_a}/{len(data)} = {total_a/len(data)*100:.1f}%"
    )
    print(
        f"  Track B (Text-only):   {total_b}/{len(data)} = {total_b/len(data)*100:.1f}%"
    )
    print(
        f"  Track C (Multimodal):  {total_c}/{len(data)} = {total_c/len(data)*100:.1f}%"
    )

    print(f"\n[Visual Noise vs Benefit]")
    print(
        f"  Visual Noise cases (Text correct, Multimodal wrong):   {len(noise_cases)}"
    )
    print(
        f"  Visual Benefit cases (Multimodal correct, Text wrong): {len(benefit_cases)}"
    )
    print(
        f"  Net VCE: {len(noise_cases) - len(benefit_cases):+d} (positive = noise dominates)"
    )

    # By Nation
    print("\n" + "=" * 70)
    print("VCE by NATION")
    print("=" * 70)
    print(
        f"{'Nation':<15} {'N':>5} {'Acc_A':>7} {'Acc_B':>7} {'Acc_C':>7} {'VCE':>7} {'Noise':>6} {'Benefit':>7}"
    )
    print("-" * 70)

    for nation in sorted(nation_stats.keys()):
        stats = nation_stats[nation]
        n = stats["n"]
        acc_a = stats["track_a_correct"] / n * 100
        acc_b = stats["track_b_correct"] / n * 100
        acc_c = stats["track_c_correct"] / n * 100
        vce = acc_b - acc_c  # Text - Multimodal

        print(
            f"{nation:<15} {n:>5} {acc_a:>6.1f}% {acc_b:>6.1f}% {acc_c:>6.1f}% {vce:>+6.1f}% {stats['visual_noise_cases']:>6} {stats['visual_benefit_cases']:>7}"
        )

    # Highlight Visual Noise phenomenon
    print("\n[Visual Noise Phenomenon - Japan Focus]")
    japan_stats = nation_stats.get("Japan", {})
    if japan_stats:
        j_acc_b = japan_stats["track_b_correct"] / japan_stats["n"] * 100
        j_acc_c = japan_stats["track_c_correct"] / japan_stats["n"] * 100
        print(f"  Japan Text-only:   {j_acc_b:.1f}%")
        print(f"  Japan Multimodal:  {j_acc_c:.1f}%")
        print(
            f"  Japan VCE:         {j_acc_b - j_acc_c:+.1f}% (adding image HURTS by {j_acc_b - j_acc_c:.1f}pp)"
        )

    # By Task
    print("\n" + "=" * 70)
    print("VCE by TASK (Top Visual Noise domains)")
    print("=" * 70)

    task_vce = []
    for task, stats in task_stats.items():
        n = stats["n"]
        if n >= 5:  # Only tasks with enough samples
            acc_b = stats["track_b_correct"] / n * 100
            acc_c = stats["track_c_correct"] / n * 100
            vce = acc_b - acc_c
            task_vce.append(
                (
                    task,
                    n,
                    acc_b,
                    acc_c,
                    vce,
                    stats["visual_noise_cases"],
                    stats["visual_benefit_cases"],
                )
            )

    # Sort by VCE (highest = most visual noise)
    task_vce.sort(key=lambda x: x[4], reverse=True)

    print(
        f"{'Task':<20} {'N':>5} {'Acc_B':>7} {'Acc_C':>7} {'VCE':>7} {'Noise':>6} {'Benefit':>7}"
    )
    print("-" * 70)
    for task, n, acc_b, acc_c, vce, noise, benefit in task_vce[:10]:
        print(
            f"{task:<20} {n:>5} {acc_b:>6.1f}% {acc_c:>6.1f}% {vce:>+6.1f}% {noise:>6} {benefit:>7}"
        )

    # Detailed Visual Noise cases
    print("\n" + "=" * 70)
    print("Sample Visual Noise Cases (Text correct, Multimodal wrong)")
    print("=" * 70)
    for i, case in enumerate(noise_cases[:5]):
        print(
            f"\n[Case {i+1}] Index={case['index']}, Nation={case['nation']}, Task={case['task']}"
        )
        print(f"  Correct answer: {case['correct_answer']}")
        print(f"  Track A (Image): {'✓' if case['track_a'] else '✗'}")
        print(f"  Track B (Text):  ✓")
        print(f"  Track C (Both):  ✗")

    return {
        "total_samples": len(data),
        "overall_acc": {
            "A": total_a / len(data),
            "B": total_b / len(data),
            "C": total_c / len(data),
        },
        "visual_noise_count": len(noise_cases),
        "visual_benefit_count": len(benefit_cases),
        "net_vce": len(noise_cases) - len(benefit_cases),
        "nation_stats": {
            k: {
                "n": v["n"],
                "acc_a": v["track_a_correct"] / v["n"],
                "acc_b": v["track_b_correct"] / v["n"],
                "acc_c": v["track_c_correct"] / v["n"],
                "vce": (v["track_b_correct"] - v["track_c_correct"]) / v["n"],
                "noise": v["visual_noise_cases"],
                "benefit": v["visual_benefit_cases"],
            }
            for k, v in nation_stats.items()
        },
        "noise_cases": noise_cases,
        "benefit_cases": benefit_cases,
    }


def main():
    print("Loading results...")
    results = load_results()
    details = results["details"]

    print("Computing VCE for each sample...")
    vce_data = compute_vce_per_sample(details)

    print("Analyzing by nation...")
    nation_stats = analyze_by_nation(vce_data)

    print("Analyzing by task...")
    task_stats = analyze_by_task(vce_data)

    print("Finding Visual Noise cases...")
    noise_cases = find_visual_noise_cases(vce_data)

    print("Finding Visual Benefit cases...")
    benefit_cases = find_visual_benefit_cases(vce_data)

    analysis = print_analysis(
        vce_data, nation_stats, task_stats, noise_cases, benefit_cases
    )

    # Save analysis
    output_file = Path(__file__).parent / "results" / "vce_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"\nAnalysis saved to: {output_file}")

    return analysis


if __name__ == "__main__":
    main()
