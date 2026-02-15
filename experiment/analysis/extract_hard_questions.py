#!/usr/bin/env python3
"""
Extract hardest questions for failure analysis.
Based on Phase 1 findings: Japan and Earth Science are hardest.
"""

from datasets import load_dataset
import json
import random
from pathlib import Path

random.seed(42)

print("Loading EuraGovExam dataset...")
ds = load_dataset("EuraGovExam/EuraGovExam", split="train")
print(f"Total samples: {len(ds)}")

with open("difficulty_ranking_results.json", "r") as f:
    difficulty = json.load(f)

hard_nations = ["japan", "india"]
hard_tasks = [
    "earth_science",
    "geography",
    "history",
    "economics",
    "engineering",
    "physics",
    "mathematics",
]

hard_questions = []
medium_questions = []
easy_questions = []

for idx, item in enumerate(ds):
    nation = item["nation"].lower().replace(" ", "_")
    task = item["task"].lower()

    question_info = {
        "index": idx,
        "nation": item["nation"],
        "task": item["task"],
        "correct_answer": item["correct answer"],
        "has_image": item["img"] is not None,
    }

    if nation in hard_nations and task in hard_tasks:
        hard_questions.append(question_info)
    elif nation in hard_nations or task in hard_tasks:
        medium_questions.append(question_info)
    else:
        easy_questions.append(question_info)

print(f"\nQuestion categorization:")
print(f"  Hard (hard nation + hard task): {len(hard_questions)}")
print(f"  Medium (hard nation OR hard task): {len(medium_questions)}")
print(f"  Easy (neither): {len(easy_questions)}")

sampled_hard = random.sample(hard_questions, min(60, len(hard_questions)))
sampled_medium = random.sample(medium_questions, min(80, len(medium_questions)))
sampled_easy = random.sample(easy_questions, min(37, len(easy_questions)))

analysis_set = sampled_hard + sampled_medium + sampled_easy
random.shuffle(analysis_set)

print(f"\nSampled for analysis: {len(analysis_set)} questions")
print(f"  From hard: {len(sampled_hard)}")
print(f"  From medium: {len(sampled_medium)}")
print(f"  From easy: {len(sampled_easy)}")

nation_dist = {}
task_dist = {}
for q in analysis_set:
    nation_dist[q["nation"]] = nation_dist.get(q["nation"], 0) + 1
    task_dist[q["task"]] = task_dist.get(q["task"], 0) + 1

print(f"\nNation distribution in sample:")
for n, c in sorted(nation_dist.items(), key=lambda x: -x[1]):
    print(f"  {n}: {c}")

print(f"\nTask distribution in sample (top 10):")
for t, c in sorted(task_dist.items(), key=lambda x: -x[1])[:10]:
    print(f"  {t}: {c}")

output_path = Path("hard_questions_for_analysis.json")
with open(output_path, "w") as f:
    json.dump(
        {
            "total_sampled": len(analysis_set),
            "sampling_strategy": "Stratified by difficulty (hard nation + hard task)",
            "hard_nations": hard_nations,
            "hard_tasks": hard_tasks,
            "questions": analysis_set,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"\nSaved to {output_path}")

indices_for_image_analysis = [q["index"] for q in analysis_set]
with open("question_indices_for_analysis.json", "w") as f:
    json.dump(indices_for_image_analysis, f)

print(f"Saved indices to question_indices_for_analysis.json")
