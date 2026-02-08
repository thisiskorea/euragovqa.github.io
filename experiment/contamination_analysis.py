import json
from pathlib import Path
from collections import defaultdict
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def load_results():
    fixed_file = RESULTS_DIR / "large_scale_20260120_170800_fixed.json"
    with open(fixed_file, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_task_difficulty(results: dict):
    print("=" * 70)
    print("Task Difficulty Analysis (Potential Contamination Indicators)")
    print("=" * 70)

    task_stats = defaultdict(
        lambda: {
            "track_a": {"correct": 0, "total": 0},
            "track_b": {"correct": 0, "total": 0},
            "track_c": {"correct": 0, "total": 0},
        }
    )

    for item in results["details"]:
        task = item["task"]
        for track in ["track_a", "track_b", "track_c"]:
            task_stats[task][track]["total"] += 1
            if item[track]["is_correct"]:
                task_stats[task][track]["correct"] += 1

    print("\nHigh accuracy tasks (potential memorization):")
    print(f"{'Task':<20} {'N':>5} {'Avg Acc':>10} {'Flag':>10}")
    print("-" * 50)

    flagged = []
    for task in sorted(task_stats.keys()):
        stats = task_stats[task]
        n = stats["track_a"]["total"]
        if n < 3:
            continue

        accs = []
        for track in ["track_a", "track_b", "track_c"]:
            if stats[track]["total"] > 0:
                acc = stats[track]["correct"] / stats[track]["total"] * 100
                accs.append(acc)

        avg_acc = np.mean(accs)
        flag = "HIGH" if avg_acc > 85 else ("MEDIUM" if avg_acc > 75 else "")

        if flag:
            flagged.append((task, n, avg_acc, flag))

        print(f"{task:<20} {n:>5} {avg_acc:>9.1f}% {flag:>10}")

    return flagged


def analyze_answer_distribution(results: dict):
    print("\n" + "=" * 70)
    print("Answer Distribution Analysis")
    print("=" * 70)

    answer_counts = defaultdict(int)
    for item in results["details"]:
        answer_counts[item["correct_answer"]] += 1

    print("\nCorrect Answer Distribution:")
    total = sum(answer_counts.values())
    expected = total / len(answer_counts)

    chi_sq = 0
    for answer in sorted(answer_counts.keys()):
        count = answer_counts[answer]
        pct = count / total * 100
        deviation = (count - expected) ** 2 / expected
        chi_sq += deviation
        print(f"  {answer}: {count} ({pct:.1f}%)")

    print(f"\nChi-square statistic: {chi_sq:.2f}")
    print(f"Expected uniform: {expected:.1f} per answer")

    if chi_sq > 9.49:
        print("WARNING: Answer distribution significantly non-uniform (p<0.05)")
    else:
        print("OK: Answer distribution reasonably uniform")


def analyze_nation_task_interaction(results: dict):
    print("\n" + "=" * 70)
    print("Nation-Task Interaction Analysis")
    print("=" * 70)

    nation_task = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for item in results["details"]:
        nation = item["nation"]
        task = item["task"]
        is_correct = item["track_c"]["is_correct"]

        nation_task[nation][task]["total"] += 1
        if is_correct:
            nation_task[nation][task]["correct"] += 1

    print("\nTrack C Accuracy by Nation-Task (cells with n>=3):")

    all_tasks = sorted(set(item["task"] for item in results["details"]))
    all_nations = sorted(set(item["nation"] for item in results["details"]))

    header = f"{'Task':<15}" + "".join(f"{n[:8]:>10}" for n in all_nations)
    print(header)
    print("-" * len(header))

    for task in all_tasks:
        row = f"{task:<15}"
        for nation in all_nations:
            stats = nation_task[nation][task]
            if stats["total"] >= 3:
                acc = stats["correct"] / stats["total"] * 100
                row += f"{acc:>9.0f}%"
            else:
                row += f"{'n<3':>10}"
        print(row)


def main():
    results = load_results()

    print(f"Analyzing {len(results['details'])} samples\n")

    analyze_task_difficulty(results)
    analyze_answer_distribution(results)
    analyze_nation_task_interaction(results)

    print("\n" + "=" * 70)
    print("CONTAMINATION ASSESSMENT SUMMARY")
    print("=" * 70)
    print("""
Limitations of this analysis:
- No temporal data (exam years) available in dataset
- Cannot directly verify against model training data
- Small sample size (n=200) limits statistical power

Indirect indicators checked:
1. Task difficulty variance - Some tasks show unusually high accuracy
2. Answer distribution - Checked for bias
3. Nation-task patterns - Regional performance differences

Recommendation for paper:
- Acknowledge limitation: cannot definitively rule out contamination
- Note: Real civil service exams from government sources
- Propose: Future work should include temporal analysis
""")


if __name__ == "__main__":
    main()
