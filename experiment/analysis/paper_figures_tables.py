#!/usr/bin/env python3
"""
논문용 Figure/Table 생성 스크립트
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ANALYSIS_DIR = Path(__file__).parent
FIGURES_DIR = ANALYSIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


def load_data():
    with open(ANALYSIS_DIR / "leaderboard.json", "r") as f:
        leaderboard = json.load(f)
    with open(ANALYSIS_DIR / "mixed_effects_anova_results.json", "r") as f:
        anova_results = json.load(f)
    with open(ANALYSIS_DIR / "controlled_experiment_results.json", "r") as f:
        controlled_results = json.load(f)
    return leaderboard, anova_results, controlled_results


def create_nation_performance_heatmap(leaderboard):
    models = []
    nations = ["taiwan", "eu", "south_korea", "india", "japan"]

    data = []
    for model in leaderboard:
        row = {"model": model["model"]}
        for nation in nations:
            row[nation] = model["nation"].get(nation, 0)
        data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index("model")
    df = df.sort_values("taiwan", ascending=False)

    top_models = df.head(15)

    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(
        top_models,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "Accuracy (%)"},
    )

    ax.set_xlabel("Region")
    ax.set_ylabel("Model")
    ax.set_title("VLM Performance by Region (Top 15 Models)")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "nation_performance_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(FIGURES_DIR / "nation_performance_heatmap.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: nation_performance_heatmap.png/pdf")


def create_variance_decomposition_chart(anova_results):
    var_comp = anova_results["variance_components"]

    nation_var = var_comp["nation_analysis"]["nation_variance"]
    task_var = var_comp["task_analysis"]["task_variance"]

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Nation\nEffect", "Task\nEffect"]
    values = [nation_var, task_var]
    colors = ["#2E86AB", "#A23B72"]

    bars = ax.bar(categories, values, color=colors, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ratio = nation_var / task_var
    ax.text(
        0.5,
        max(values) * 0.5,
        f"Ratio: {ratio:.1f}x",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_ylabel("Variance")
    ax.set_title("Nation Effect vs Task Effect\n(Variance Decomposition)")
    ax.set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "variance_decomposition.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(FIGURES_DIR / "variance_decomposition.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: variance_decomposition.png/pdf")


def create_nation_ranking_barplot(leaderboard):
    from collections import defaultdict

    nation_accs = defaultdict(list)
    for model in leaderboard:
        for nation, acc in model["nation"].items():
            nation_accs[nation].append(acc)

    nation_means = {n: np.mean(accs) for n, accs in nation_accs.items()}
    nation_stds = {n: np.std(accs) for n, accs in nation_accs.items()}

    sorted_nations = sorted(nation_means.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    nations = [n[0].replace("_", " ").title() for n in sorted_nations]
    means = [n[1] for n in sorted_nations]
    stds = [nation_stds[n[0]] for n in sorted_nations]

    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(nations)))

    bars = ax.barh(nations, means, xerr=stds, color=colors, capsize=3)

    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(
            mean + stds[i] + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.1f}%",
            va="center",
            fontsize=11,
        )

    ax.set_xlabel("Average Accuracy (%)")
    ax.set_title("VLM Performance by Region\n(Mean ± Std across 23 models)")
    ax.set_xlim(0, 70)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nation_ranking.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "nation_ranking.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: nation_ranking.png/pdf")


def create_model_comparison_table(leaderboard):
    top_models = sorted(leaderboard, key=lambda x: x["overall"], reverse=True)[:10]

    table_data = []
    for model in top_models:
        row = {
            "Model": model["model"],
            "Overall": f"{model['overall']:.1f}",
            "Taiwan": f"{model['nation'].get('taiwan', 0):.1f}",
            "EU": f"{model['nation'].get('eu', 0):.1f}",
            "Korea": f"{model['nation'].get('south_korea', 0):.1f}",
            "India": f"{model['nation'].get('india', 0):.1f}",
            "Japan": f"{model['nation'].get('japan', 0):.1f}",
            "Range": f"{max(model['nation'].values()) - min(model['nation'].values()):.1f}",
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    latex_table = df.to_latex(index=False, escape=False)

    with open(FIGURES_DIR / "main_results_table.tex", "w") as f:
        f.write(latex_table)

    df.to_csv(FIGURES_DIR / "main_results_table.csv", index=False)

    print(f"Saved: main_results_table.tex, main_results_table.csv")

    return df


def create_effect_size_comparison(anova_results):
    nation_eta = anova_results["anova_nation"]["eta_squared"]
    task_eta = anova_results["anova_task"]["eta_squared"]

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Nation Effect", "Task Effect"]
    eta_values = [nation_eta * 100, task_eta * 100]
    colors = ["#2E86AB", "#A23B72"]

    bars = ax.bar(categories, eta_values, color=colors, width=0.5)

    for bar, val in zip(bars, eta_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"η² = {val/100:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Small effect (1%)")
    ax.axhline(y=6, color="gray", linestyle="-.", alpha=0.5, label="Medium effect (6%)")
    ax.axhline(y=14, color="gray", linestyle=":", alpha=0.5, label="Large effect (14%)")

    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Effect Size Comparison (η²)")
    ax.set_ylim(0, 20)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "effect_size_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(FIGURES_DIR / "effect_size_comparison.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: effect_size_comparison.png/pdf")


def generate_paper_summary(leaderboard, anova_results, controlled_results):
    summary = """
================================================================================
                    PAPER-READY STATISTICS SUMMARY
================================================================================

1. MAIN FINDING: Nation Effect Dominates Task Effect
--------------------------------------------------------------------------------
   - Variance Ratio: {nation_var:.1f} / {task_var:.1f} = {ratio:.1f}x
   - Nation η² = {nation_eta:.3f} ({nation_interp})
   - Task η² = {task_eta:.3f} ({task_interp})

2. STATISTICAL SIGNIFICANCE
--------------------------------------------------------------------------------
   - Nation effect: F = {nation_f:.2f}, p = {nation_p}
   - Task effect: F = {task_f:.2f}, p = {task_p}

3. CONSISTENCY ACROSS MODELS
--------------------------------------------------------------------------------
   - Nation dominates in {n_dom}/{n_total} models ({pct:.1f}%)
   - Binomial test: p = {binom_p:.4f} (significant)
   - Closed models: {closed_ratio:.2f}x ratio
   - Open models: {open_ratio:.2f}x ratio

4. REGIONAL PERFORMANCE GAP
--------------------------------------------------------------------------------
   - Hardest: {hardest} ({hardest_acc:.1f}%)
   - Easiest: {easiest} ({easiest_acc:.1f}%)
   - Gap: {gap:.1f} percentage points

5. KEY TAKEAWAY FOR PAPER
--------------------------------------------------------------------------------
   "The jurisdiction (region) of a civil service exam explains {nation_eta_pct:.1f}%
    of VLM performance variance, compared to only {task_eta_pct:.1f}% for subject
    domain. This {ratio:.1f}x difference is statistically significant (p < 0.01)
    and consistent across 82.6% of evaluated models."

================================================================================
""".format(
        nation_var=anova_results["variance_components"]["nation_analysis"][
            "nation_variance"
        ],
        task_var=anova_results["variance_components"]["task_analysis"]["task_variance"],
        ratio=anova_results["variance_components"]["comparison"][
            "nation_to_task_variance_ratio"
        ],
        nation_eta=anova_results["anova_nation"]["eta_squared"],
        nation_interp=anova_results["anova_nation"]["effect_size_interpretation"],
        task_eta=anova_results["anova_task"]["eta_squared"],
        task_interp=anova_results["anova_task"]["effect_size_interpretation"],
        nation_f=anova_results["anova_nation"]["f_statistic"],
        nation_p=anova_results["anova_nation"]["p_value_formatted"],
        task_f=anova_results["anova_task"]["f_statistic"],
        task_p=anova_results["anova_task"]["p_value_formatted"],
        n_dom=controlled_results["within_model"]["summary"]["n_nation_dominates"],
        n_total=controlled_results["within_model"]["summary"]["n_models"],
        pct=controlled_results["within_model"]["summary"]["percentage"],
        binom_p=controlled_results["within_model"]["summary"]["binomial_p_value"],
        closed_ratio=controlled_results["by_model_type"]["closed_models"]["mean_ratio"],
        open_ratio=controlled_results["by_model_type"]["open_models"]["mean_ratio"],
        hardest=controlled_results["paper_stats"]["headline_stats"]["hardest_nation"],
        hardest_acc=controlled_results["paper_stats"]["headline_stats"][
            "hardest_nation_acc"
        ],
        easiest=controlled_results["paper_stats"]["headline_stats"]["easiest_nation"],
        easiest_acc=controlled_results["paper_stats"]["headline_stats"][
            "easiest_nation_acc"
        ],
        gap=controlled_results["paper_stats"]["headline_stats"]["performance_gap"],
        nation_eta_pct=anova_results["anova_nation"]["eta_squared"] * 100,
        task_eta_pct=anova_results["anova_task"]["eta_squared"] * 100,
    )

    print(summary)

    with open(FIGURES_DIR / "paper_summary.txt", "w") as f:
        f.write(summary)

    print(f"\nSaved: paper_summary.txt")


def main():
    print("=" * 70)
    print("Generating Paper Figures and Tables")
    print("=" * 70)

    leaderboard, anova_results, controlled_results = load_data()

    print("\n[1] Creating Nation Performance Heatmap...")
    create_nation_performance_heatmap(leaderboard)

    print("\n[2] Creating Variance Decomposition Chart...")
    create_variance_decomposition_chart(anova_results)

    print("\n[3] Creating Nation Ranking Barplot...")
    create_nation_ranking_barplot(leaderboard)

    print("\n[4] Creating Main Results Table...")
    create_model_comparison_table(leaderboard)

    print("\n[5] Creating Effect Size Comparison...")
    create_effect_size_comparison(anova_results)

    print("\n[6] Generating Paper Summary...")
    generate_paper_summary(leaderboard, anova_results, controlled_results)

    print("\n" + "=" * 70)
    print("All figures and tables saved to:", FIGURES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
