#!/usr/bin/env python3
"""
EuraGovExam Phase 1: Statistical Rigor Analysis
==============================================
- Task 2: Bootstrap CI (approximate)
- Task 3: Pairwise significance tests
- Task 4: Variance decomposition & interaction analysis

Author: EuraGovExam Team
Date: 2026-01-21
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TOTAL_QUESTIONS = 8000  # Total dataset size
ALPHA = 0.05  # Significance level
N_BOOTSTRAP = 10000  # Bootstrap iterations

NATIONS = ["india", "eu", "taiwan", "japan", "south_korea"]
NATION_SIZES = {
    "india": 1699,
    "eu": 1289,
    "taiwan": 557,
    "japan": 2557,
    "south_korea": 1898,
}  # From dataset distribution

DOMAIN_SIZES = {
    "mathematics": 1039,
    "administration": 928,
    "biology": 899,
    "law": 818,
    "language": 740,
    "engineering": 688,
    "physics": 540,
    "economics": 463,
    "computer_science": 441,
    "history": 256,
    "medicine": 238,
    "politics": 209,
    "chemistry": 207,
    "geography": 181,
    "philosophy": 127,
    "psychology": 119,
    "earth_science": 107,
}

# ==============================================================================
# TASK 2: Bootstrap CI Calculation (Approximate)
# ==============================================================================


def wilson_ci(p, n, z=1.96):
    """
    Wilson score interval for binomial proportion.
    More accurate than normal approximation, especially for extreme p.
    """
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0, center - margin), min(1, center + margin)


def compute_ci_for_model(model_data, total_n=TOTAL_QUESTIONS):
    """Compute 95% CI for a single model's overall and breakdown accuracies."""
    results = {
        "model": model_data["model"],
        "overall": {"accuracy": model_data["overall"], "n": total_n},
        "by_nation": {},
        "by_task": {},
    }

    # Overall CI
    p = model_data["overall"] / 100
    ci_low, ci_high = wilson_ci(p, total_n)
    results["overall"]["ci_lower"] = round(ci_low * 100, 2)
    results["overall"]["ci_upper"] = round(ci_high * 100, 2)
    results["overall"]["ci_width"] = round((ci_high - ci_low) * 100, 2)

    # By nation CI
    for nation, acc in model_data["nation"].items():
        n = NATION_SIZES.get(nation, 1000)  # fallback
        p = acc / 100
        ci_low, ci_high = wilson_ci(p, n)
        results["by_nation"][nation] = {
            "accuracy": acc,
            "n": n,
            "ci_lower": round(ci_low * 100, 2),
            "ci_upper": round(ci_high * 100, 2),
            "ci_width": round((ci_high - ci_low) * 100, 2),
        }

    # By task/domain CI
    for task, acc in model_data["tasks"].items():
        n = DOMAIN_SIZES.get(task, 500)  # fallback
        p = acc / 100
        ci_low, ci_high = wilson_ci(p, n)
        results["by_task"][task] = {
            "accuracy": acc,
            "n": n,
            "ci_lower": round(ci_low * 100, 2),
            "ci_upper": round(ci_high * 100, 2),
            "ci_width": round((ci_high - ci_low) * 100, 2),
        }

    return results


def compute_all_ci(leaderboard_data):
    """Compute CI for all models."""
    return [compute_ci_for_model(m) for m in leaderboard_data]


# ==============================================================================
# TASK 3: Pairwise Significance Tests
# ==============================================================================


def two_proportion_z_test(p1, n1, p2, n2):
    """
    Two-proportion z-test for comparing two independent proportions.
    H0: p1 = p2
    """
    # Pooled proportion
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        return 0, 1.0

    # Z-statistic
    z = (p1 - p2) / se

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


def cohens_h(p1, p2):
    """
    Cohen's h effect size for two proportions.
    Small: 0.2, Medium: 0.5, Large: 0.8
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def pairwise_significance_tests(leaderboard_data, top_k=10):
    """
    Perform pairwise significance tests between top-k models.
    """
    # Sort by overall accuracy
    sorted_models = sorted(leaderboard_data, key=lambda x: x["overall"], reverse=True)[
        :top_k
    ]

    results = {
        "models_compared": [m["model"] for m in sorted_models],
        "pairwise_tests": [],
        "bonferroni_alpha": ALPHA / (top_k * (top_k - 1) / 2),
    }

    n = TOTAL_QUESTIONS

    for i, model1 in enumerate(sorted_models):
        for j, model2 in enumerate(sorted_models[i + 1 :], i + 1):
            p1 = model1["overall"] / 100
            p2 = model2["overall"] / 100

            z_stat, p_value = two_proportion_z_test(p1, n, p2, n)
            h = cohens_h(p1, p2)

            test_result = {
                "model_1": model1["model"],
                "model_2": model2["model"],
                "acc_1": model1["overall"],
                "acc_2": model2["overall"],
                "diff": round(model1["overall"] - model2["overall"], 2),
                "z_statistic": round(float(z_stat), 3),
                "p_value": float(p_value),
                "p_value_formatted": f"{p_value:.2e}"
                if p_value < 0.001
                else f"{p_value:.4f}",
                "cohens_h": round(float(h), 3),
                "effect_size": "large"
                if abs(h) >= 0.8
                else ("medium" if abs(h) >= 0.5 else "small"),
                "significant_bonferroni": bool(p_value < results["bonferroni_alpha"]),
                "significant_uncorrected": bool(p_value < ALPHA),
            }
            results["pairwise_tests"].append(test_result)

    # Summary statistics
    sig_count = sum(1 for t in results["pairwise_tests"] if t["significant_bonferroni"])
    results["summary"] = {
        "total_comparisons": len(results["pairwise_tests"]),
        "significant_after_bonferroni": sig_count,
        "bonferroni_alpha": round(results["bonferroni_alpha"], 6),
    }

    return results


# ==============================================================================
# TASK 4: Variance Decomposition & Interaction Analysis
# ==============================================================================


def variance_decomposition(leaderboard_data):
    """
    Decompose variance in accuracy across regions and domains.
    """
    results = {"by_model": [], "aggregate": {}}

    all_nation_vars = []
    all_task_vars = []

    for model in leaderboard_data:
        nation_accs = list(model["nation"].values())
        task_accs = list(model["tasks"].values())

        nation_var = np.var(nation_accs)
        task_var = np.var(task_accs)
        nation_range = max(nation_accs) - min(nation_accs)
        task_range = max(task_accs) - min(task_accs)

        model_result = {
            "model": model["model"],
            "overall": model["overall"],
            "nation_variance": round(nation_var, 2),
            "nation_std": round(np.std(nation_accs), 2),
            "nation_range": round(nation_range, 2),
            "nation_min": round(min(nation_accs), 2),
            "nation_max": round(max(nation_accs), 2),
            "task_variance": round(task_var, 2),
            "task_std": round(np.std(task_accs), 2),
            "task_range": round(task_range, 2),
            "task_min": round(min(task_accs), 2),
            "task_max": round(max(task_accs), 2),
        }

        # Find hardest/easiest
        nation_dict = model["nation"]
        task_dict = model["tasks"]
        model_result["hardest_nation"] = min(nation_dict, key=nation_dict.get)
        model_result["easiest_nation"] = max(nation_dict, key=nation_dict.get)
        model_result["hardest_task"] = min(task_dict, key=task_dict.get)
        model_result["easiest_task"] = max(task_dict, key=task_dict.get)

        results["by_model"].append(model_result)
        all_nation_vars.append(nation_var)
        all_task_vars.append(task_var)

    # Aggregate statistics
    results["aggregate"] = {
        "mean_nation_variance": round(np.mean(all_nation_vars), 2),
        "mean_task_variance": round(np.mean(all_task_vars), 2),
        "nation_vs_task_ratio": round(
            np.mean(all_nation_vars) / np.mean(all_task_vars), 3
        )
        if np.mean(all_task_vars) > 0
        else 0,
        "interpretation": "Region causes MORE variance"
        if np.mean(all_nation_vars) > np.mean(all_task_vars)
        else "Domain causes MORE variance",
    }

    return results


def compute_difficulty_ranking(leaderboard_data):
    """
    Compute difficulty ranking for nations and domains across all models.
    """
    # Aggregate accuracies
    nation_accs = {n: [] for n in NATIONS}
    task_accs = {t: [] for t in DOMAIN_SIZES.keys()}

    for model in leaderboard_data:
        for nation, acc in model["nation"].items():
            if nation in nation_accs:
                nation_accs[nation].append(acc)
        for task, acc in model["tasks"].items():
            if task in task_accs:
                task_accs[task].append(acc)

    # Compute mean accuracy per nation/task
    nation_difficulty = {
        n: {
            "mean_accuracy": round(np.mean(accs), 2),
            "std": round(np.std(accs), 2),
            "min": round(min(accs), 2),
            "max": round(max(accs), 2),
        }
        for n, accs in nation_accs.items()
        if accs
    }

    task_difficulty = {
        t: {
            "mean_accuracy": round(np.mean(accs), 2),
            "std": round(np.std(accs), 2),
            "min": round(min(accs), 2),
            "max": round(max(accs), 2),
        }
        for t, accs in task_accs.items()
        if accs
    }

    # Sort by difficulty (lower accuracy = harder)
    nation_ranking = sorted(
        nation_difficulty.items(), key=lambda x: x[1]["mean_accuracy"]
    )
    task_ranking = sorted(task_difficulty.items(), key=lambda x: x[1]["mean_accuracy"])

    return {
        "nation_difficulty": dict(nation_ranking),
        "task_difficulty": dict(task_ranking),
        "hardest_nation": nation_ranking[0][0] if nation_ranking else None,
        "easiest_nation": nation_ranking[-1][0] if nation_ranking else None,
        "hardest_task": task_ranking[0][0] if task_ranking else None,
        "easiest_task": task_ranking[-1][0] if task_ranking else None,
    }


def interaction_analysis(leaderboard_data):
    """
    Analyze Model x Nation and Model x Domain interactions.
    Find where rankings reverse or models have unusual strengths/weaknesses.
    """
    results = {
        "model_nation_interaction": [],
        "model_task_interaction": [],
        "rank_reversals": [],
    }

    # Get top 5 models
    sorted_models = sorted(leaderboard_data, key=lambda x: x["overall"], reverse=True)[
        :5
    ]

    # Model x Nation interaction
    for model in sorted_models:
        overall = model["overall"]
        for nation, acc in model["nation"].items():
            delta = acc - overall
            results["model_nation_interaction"].append(
                {
                    "model": model["model"],
                    "nation": nation,
                    "nation_accuracy": acc,
                    "overall_accuracy": overall,
                    "delta": round(delta, 2),
                    "relative_strength": "strong"
                    if delta > 10
                    else ("weak" if delta < -10 else "neutral"),
                }
            )

    # Model x Task interaction
    for model in sorted_models:
        overall = model["overall"]
        for task, acc in model["tasks"].items():
            delta = acc - overall
            results["model_task_interaction"].append(
                {
                    "model": model["model"],
                    "task": task,
                    "task_accuracy": acc,
                    "overall_accuracy": overall,
                    "delta": round(delta, 2),
                    "relative_strength": "strong"
                    if delta > 10
                    else ("weak" if delta < -10 else "neutral"),
                }
            )

    # Find rank reversals (where model A > B overall but B > A in specific nation/task)
    for i, model1 in enumerate(sorted_models):
        for model2 in sorted_models[i + 1 :]:
            # Check nations
            for nation in NATIONS:
                if nation in model1["nation"] and nation in model2["nation"]:
                    if model1["nation"][nation] < model2["nation"][nation]:
                        results["rank_reversals"].append(
                            {
                                "type": "nation",
                                "condition": nation,
                                "model_higher_overall": model1["model"],
                                "model_lower_overall": model2["model"],
                                "higher_model_score": model1["nation"][nation],
                                "lower_model_score": model2["nation"][nation],
                                "reversal_magnitude": round(
                                    model2["nation"][nation] - model1["nation"][nation],
                                    2,
                                ),
                            }
                        )

            # Check tasks (top 5 tasks only to limit output)
            for task in list(DOMAIN_SIZES.keys())[:5]:
                if task in model1["tasks"] and task in model2["tasks"]:
                    if model1["tasks"][task] < model2["tasks"][task]:
                        results["rank_reversals"].append(
                            {
                                "type": "task",
                                "condition": task,
                                "model_higher_overall": model1["model"],
                                "model_lower_overall": model2["model"],
                                "higher_model_score": model1["tasks"][task],
                                "lower_model_score": model2["tasks"][task],
                                "reversal_magnitude": round(
                                    model2["tasks"][task] - model1["tasks"][task], 2
                                ),
                            }
                        )

    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    print("=" * 60)
    print("EuraGovExam Phase 1: Statistical Rigor Analysis")
    print("=" * 60)

    # Load data
    with open("leaderboard.json", "r") as f:
        leaderboard = json.load(f)

    print(f"\nLoaded {len(leaderboard)} models from leaderboard.json")

    # Task 2: Bootstrap CI
    print("\n[Task 2] Computing Confidence Intervals...")
    ci_results = compute_all_ci(leaderboard)

    with open("bootstrap_ci_results.json", "w") as f:
        json.dump(ci_results, f, indent=2)
    print(f"  -> Saved to bootstrap_ci_results.json")

    # Print sample
    top_model = ci_results[0]
    print(f"  Example: {top_model['model']}")
    print(
        f"    Overall: {top_model['overall']['accuracy']}% "
        f"[{top_model['overall']['ci_lower']}%, {top_model['overall']['ci_upper']}%]"
    )

    # Task 3: Pairwise Tests
    print("\n[Task 3] Performing Pairwise Significance Tests...")
    pairwise_results = pairwise_significance_tests(leaderboard, top_k=10)

    with open("pairwise_test_results.json", "w") as f:
        json.dump(pairwise_results, f, indent=2)
    print(f"  -> Saved to pairwise_test_results.json")

    # Print summary
    print(f"  Total comparisons: {pairwise_results['summary']['total_comparisons']}")
    print(
        f"  Significant (Bonferroni): {pairwise_results['summary']['significant_after_bonferroni']}"
    )
    print(f"  Bonferroni alpha: {pairwise_results['summary']['bonferroni_alpha']:.6f}")

    # Task 4: Variance Decomposition
    print("\n[Task 4] Computing Variance Decomposition...")
    variance_results = variance_decomposition(leaderboard)

    with open("variance_decomposition_results.json", "w") as f:
        json.dump(variance_results, f, indent=2)
    print(f"  -> Saved to variance_decomposition_results.json")

    # Print aggregate
    agg = variance_results["aggregate"]
    print(f"  Mean nation variance: {agg['mean_nation_variance']}")
    print(f"  Mean task variance: {agg['mean_task_variance']}")
    print(f"  Interpretation: {agg['interpretation']}")

    # Difficulty Ranking
    print("\n[Task 4b] Computing Difficulty Rankings...")
    difficulty_results = compute_difficulty_ranking(leaderboard)

    with open("difficulty_ranking_results.json", "w") as f:
        json.dump(difficulty_results, f, indent=2)
    print(f"  -> Saved to difficulty_ranking_results.json")

    print(f"  Hardest nation: {difficulty_results['hardest_nation']}")
    print(f"  Easiest nation: {difficulty_results['easiest_nation']}")
    print(f"  Hardest task: {difficulty_results['hardest_task']}")
    print(f"  Easiest task: {difficulty_results['easiest_task']}")

    # Interaction Analysis
    print("\n[Task 4c] Computing Interaction Analysis...")
    interaction_results = interaction_analysis(leaderboard)

    with open("interaction_analysis_results.json", "w") as f:
        json.dump(interaction_results, f, indent=2)
    print(f"  -> Saved to interaction_analysis_results.json")

    print(f"  Found {len(interaction_results['rank_reversals'])} rank reversals")

    # Final Summary
    print("\n" + "=" * 60)
    print("Phase 1 Statistical Analysis Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. bootstrap_ci_results.json")
    print("  2. pairwise_test_results.json")
    print("  3. variance_decomposition_results.json")
    print("  4. difficulty_ranking_results.json")
    print("  5. interaction_analysis_results.json")

    return {
        "ci": ci_results,
        "pairwise": pairwise_results,
        "variance": variance_results,
        "difficulty": difficulty_results,
        "interaction": interaction_results,
    }


if __name__ == "__main__":
    results = main()
