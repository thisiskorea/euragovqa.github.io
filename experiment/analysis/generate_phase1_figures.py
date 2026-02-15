#!/usr/bin/env python3
"""
Phase 1 Figure Generation for EuraGovExam Statistical Analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 150

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_results():
    with open("bootstrap_ci_results.json") as f:
        ci_results = json.load(f)
    with open("pairwise_test_results.json") as f:
        pairwise_results = json.load(f)
    with open("variance_decomposition_results.json") as f:
        variance_results = json.load(f)
    with open("difficulty_ranking_results.json") as f:
        difficulty_results = json.load(f)
    with open("interaction_analysis_results.json") as f:
        interaction_results = json.load(f)
    return (
        ci_results,
        pairwise_results,
        variance_results,
        difficulty_results,
        interaction_results,
    )


def fig1_overall_accuracy_with_ci(ci_results):
    """Overall accuracy with 95% CI for all models."""
    sorted_models = sorted(
        ci_results, key=lambda x: x["overall"]["accuracy"], reverse=True
    )

    models = [m["model"] for m in sorted_models]
    accs = [m["overall"]["accuracy"] for m in sorted_models]
    ci_lows = [m["overall"]["ci_lower"] for m in sorted_models]
    ci_highs = [m["overall"]["ci_upper"] for m in sorted_models]

    errors_low = [a - l for a, l in zip(accs, ci_lows)]
    errors_high = [h - a for a, h in zip(accs, ci_highs)]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = []
    for acc in accs:
        if acc >= 80:
            colors.append("#2ecc71")
        elif acc >= 50:
            colors.append("#3498db")
        elif acc >= 25:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    y_pos = np.arange(len(models))
    bars = ax.barh(
        y_pos,
        accs,
        xerr=[errors_low, errors_high],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )

    ax.axvline(
        x=25, color="red", linestyle="--", linewidth=1.5, label="Random Guess (25%)"
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("EuraGovExam: Model Performance with 95% Confidence Intervals")
    ax.set_xlim(0, 100)

    for i, (acc, ci_l, ci_h) in enumerate(zip(accs, ci_lows, ci_highs)):
        ax.text(
            acc + 2, i, f"{acc:.1f}% [{ci_l:.1f}, {ci_h:.1f}]", va="center", fontsize=8
        )

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="High (>=80%)"),
        mpatches.Patch(color="#3498db", label="Medium (50-80%)"),
        mpatches.Patch(color="#f39c12", label="Low (25-50%)"),
        mpatches.Patch(color="#e74c3c", label="Very Low (<25%)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Random Guess"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "fig1_overall_accuracy_ci.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(OUTPUT_DIR / "fig1_overall_accuracy_ci.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig1_overall_accuracy_ci.png/pdf saved")


def fig2_nation_difficulty(difficulty_results):
    """Nation difficulty ranking with mean accuracy across all models."""
    nations = list(difficulty_results["nation_difficulty"].keys())
    means = [
        difficulty_results["nation_difficulty"][n]["mean_accuracy"] for n in nations
    ]
    stds = [difficulty_results["nation_difficulty"][n]["std"] for n in nations]

    nation_labels = {
        "japan": "Japan",
        "india": "India",
        "south_korea": "South Korea",
        "eu": "EU",
        "taiwan": "Taiwan",
    }
    labels = [nation_labels.get(n, n) for n in nations]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "#e74c3c" if m < 35 else "#f39c12" if m < 45 else "#2ecc71" for m in means
    ]

    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        color=colors,
        edgecolor="black",
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    ax.axhline(y=25, color="red", linestyle="--", linewidth=1.5, label="Random Guess")

    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_xlabel("Nation/Region")
    ax.set_title("EuraGovExam: Difficulty by Nation (Mean ± Std across all models)")
    ax.set_ylim(0, 80)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 2,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_nation_difficulty.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig2_nation_difficulty.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig2_nation_difficulty.png/pdf saved")


def fig3_domain_difficulty(difficulty_results):
    """Domain difficulty ranking."""
    tasks = list(difficulty_results["task_difficulty"].keys())
    means = [difficulty_results["task_difficulty"][t]["mean_accuracy"] for t in tasks]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [
        "#e74c3c" if m < 35 else "#f39c12" if m < 42 else "#2ecc71" for m in means
    ]

    y_pos = np.arange(len(tasks))
    bars = ax.barh(y_pos, means, color=colors, edgecolor="black", linewidth=0.5)

    ax.axvline(x=25, color="red", linestyle="--", linewidth=1.5, label="Random Guess")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.replace("_", " ").title() for t in tasks])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Accuracy (%)")
    ax.set_title("EuraGovExam: Difficulty by Domain (Mean across all models)")
    ax.set_xlim(0, 60)

    for i, mean in enumerate(means):
        ax.text(mean + 1, i, f"{mean:.1f}%", va="center", fontsize=9)

    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_domain_difficulty.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig3_domain_difficulty.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig3_domain_difficulty.png/pdf saved")


def fig4_pairwise_significance_heatmap(pairwise_results):
    """Heatmap of pairwise significance tests."""
    models = pairwise_results["models_compared"]
    n = len(models)

    p_matrix = np.ones((n, n))
    diff_matrix = np.zeros((n, n))

    model_idx = {m: i for i, m in enumerate(models)}

    for test in pairwise_results["pairwise_tests"]:
        i = model_idx[test["model_1"]]
        j = model_idx[test["model_2"]]
        p_matrix[i, j] = test["p_value"]
        p_matrix[j, i] = test["p_value"]
        diff_matrix[i, j] = test["diff"]
        diff_matrix[j, i] = -test["diff"]

    fig, ax = plt.subplots(figsize=(12, 10))

    sig_matrix = -np.log10(p_matrix + 1e-300)
    mask = np.triu(np.ones_like(sig_matrix, dtype=bool))

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        sig_matrix,
        mask=mask,
        annot=False,
        cmap=cmap,
        xticklabels=models,
        yticklabels=models,
        ax=ax,
        vmin=0,
        vmax=50,
        cbar_kws={"label": "-log10(p-value)"},
    )

    bonf_alpha = pairwise_results["bonferroni_alpha"]
    bonf_threshold = -np.log10(bonf_alpha)

    for test in pairwise_results["pairwise_tests"]:
        i = model_idx[test["model_1"]]
        j = model_idx[test["model_2"]]
        if i < j:
            sig_marker = "***" if test["significant_bonferroni"] else ""
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{test['diff']:.1f}\n{sig_marker}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax.set_title(
        f"Pairwise Significance Tests (Bonferroni α={bonf_alpha:.4f})\n*** = significant after correction"
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "fig4_pairwise_significance.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(OUTPUT_DIR / "fig4_pairwise_significance.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig4_pairwise_significance.png/pdf saved")


def fig5_variance_decomposition(variance_results):
    """Variance decomposition: Nation vs Domain."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sorted_models = sorted(
        variance_results["by_model"], key=lambda x: x["overall"], reverse=True
    )[:10]

    models = [m["model"][:15] for m in sorted_models]
    nation_vars = [m["nation_variance"] for m in sorted_models]
    task_vars = [m["task_variance"] for m in sorted_models]

    x = np.arange(len(models))
    width = 0.35

    ax1 = axes[0]
    bars1 = ax1.bar(
        x - width / 2, nation_vars, width, label="Nation Variance", color="#3498db"
    )
    bars2 = ax1.bar(
        x + width / 2, task_vars, width, label="Domain Variance", color="#e74c3c"
    )

    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Model")
    ax1.set_title("Variance in Accuracy: Nation vs Domain")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()

    ax2 = axes[1]
    agg = variance_results["aggregate"]
    categories = ["Nation", "Domain"]
    values = [agg["mean_nation_variance"], agg["mean_task_variance"]]
    colors = ["#3498db", "#e74c3c"]

    bars = ax2.bar(categories, values, color=colors, edgecolor="black")
    ax2.set_ylabel("Mean Variance (across all models)")
    ax2.set_title(f'Aggregate Variance Comparison\n{agg["interpretation"]}')

    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "fig5_variance_decomposition.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(OUTPUT_DIR / "fig5_variance_decomposition.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig5_variance_decomposition.png/pdf saved")


def fig6_model_nation_heatmap(ci_results):
    """Heatmap of model performance by nation."""
    sorted_models = sorted(
        ci_results, key=lambda x: x["overall"]["accuracy"], reverse=True
    )[:12]

    models = [m["model"] for m in sorted_models]
    nations = ["taiwan", "eu", "south_korea", "india", "japan"]
    nation_labels = ["Taiwan", "EU", "S.Korea", "India", "Japan"]

    data = []
    for model in sorted_models:
        row = [model["by_nation"].get(n, {}).get("accuracy", 0) for n in nations]
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=nation_labels,
        yticklabels=models,
        ax=ax,
        vmin=10,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
    )

    ax.set_title("Model Performance by Nation\n(Sorted by Overall Accuracy)")
    ax.set_xlabel("Nation")
    ax.set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "fig6_model_nation_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(OUTPUT_DIR / "fig6_model_nation_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("  -> fig6_model_nation_heatmap.png/pdf saved")


def main():
    print("=" * 60)
    print("Generating Phase 1 Figures")
    print("=" * 60)

    (
        ci_results,
        pairwise_results,
        variance_results,
        difficulty_results,
        interaction_results,
    ) = load_results()

    print("\n[Figure 1] Overall Accuracy with CI...")
    fig1_overall_accuracy_with_ci(ci_results)

    print("\n[Figure 2] Nation Difficulty...")
    fig2_nation_difficulty(difficulty_results)

    print("\n[Figure 3] Domain Difficulty...")
    fig3_domain_difficulty(difficulty_results)

    print("\n[Figure 4] Pairwise Significance Heatmap...")
    fig4_pairwise_significance_heatmap(pairwise_results)

    print("\n[Figure 5] Variance Decomposition...")
    fig5_variance_decomposition(variance_results)

    print("\n[Figure 6] Model x Nation Heatmap...")
    fig6_model_nation_heatmap(ci_results)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
