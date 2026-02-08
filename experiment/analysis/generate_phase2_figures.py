#!/usr/bin/env python3
"""Generate Phase 2 Failure Taxonomy visualization figures."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

with open("failure_taxonomy_summary.json", "r") as f:
    summary = json.load(f)

CATEGORY_LABELS = {
    "vertical_nonlatin_script": "Non-Latin Script",
    "ocr_text_recognition": "OCR/Text Recognition",
    "math_symbol_interpretation": "Math/Symbol",
    "pure_reasoning_knowledge": "Pure Reasoning",
    "diagram_understanding": "Diagram Understanding",
    "table_structure": "Table/Chart Structure",
    "figure_text_alignment": "Figure-Text Alignment",
    "code_switching": "Code-Switching",
    "multi_column_layout": "Multi-Column Layout",
    "image_quality": "Image Quality",
}

COLORS = {
    "vertical_nonlatin_script": "#e74c3c",
    "ocr_text_recognition": "#3498db",
    "math_symbol_interpretation": "#9b59b6",
    "pure_reasoning_knowledge": "#2ecc71",
    "diagram_understanding": "#f39c12",
    "table_structure": "#1abc9c",
    "figure_text_alignment": "#e67e22",
    "code_switching": "#95a5a6",
    "multi_column_layout": "#34495e",
    "image_quality": "#7f8c8d",
}


def fig7_primary_category_distribution():
    """Primary failure category distribution (pie + bar)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    dist = summary["primary_category_distribution"]
    categories = list(dist.keys())
    counts = list(dist.values())
    total = sum(counts)

    labels = [CATEGORY_LABELS.get(c, c) for c in categories]
    colors = [COLORS.get(c, "#cccccc") for c in categories]

    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )
    ax1.set_title("Primary Failure Categories\n(n=176)", fontsize=14, fontweight="bold")

    ax1.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.9, 0.5), fontsize=9)

    y_pos = np.arange(len(categories))
    bars = ax2.barh(y_pos, counts, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Number of Questions", fontsize=12)
    ax2.set_title("Failure Category Counts", fontsize=14, fontweight="bold")

    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / total * 100
        ax2.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax2.set_xlim(0, max(counts) * 1.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_failure_taxonomy_distribution.png")
    plt.savefig(FIGURES_DIR / "fig7_failure_taxonomy_distribution.pdf")
    plt.close()
    print("Generated fig7_failure_taxonomy_distribution")


def fig8_nation_failure_heatmap():
    """Failure category by nation heatmap."""
    nation_data = summary["nation_category_breakdown"]

    all_categories = list(summary["primary_category_distribution"].keys())
    nations = list(nation_data.keys())

    matrix = np.zeros((len(nations), len(all_categories)))
    for i, nation in enumerate(nations):
        for j, cat in enumerate(all_categories):
            matrix[i, j] = nation_data[nation].get(cat, 0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_pct = matrix / row_sums * 100

    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(matrix_pct, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(all_categories)))
    ax.set_yticks(np.arange(len(nations)))
    ax.set_xticklabels(
        [CATEGORY_LABELS.get(c, c) for c in all_categories], rotation=45, ha="right"
    )
    ax.set_yticklabels(nations)

    for i in range(len(nations)):
        for j in range(len(all_categories)):
            val = matrix_pct[i, j]
            count = int(matrix[i, j])
            if count > 0:
                text_color = "white" if val > 30 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}%\n({count})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Percentage within Nation (%)", fontsize=10)

    ax.set_title(
        "Failure Category Distribution by Nation", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Failure Category", fontsize=12)
    ax.set_ylabel("Nation", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_nation_failure_heatmap.png")
    plt.savefig(FIGURES_DIR / "fig8_nation_failure_heatmap.pdf")
    plt.close()
    print("Generated fig8_nation_failure_heatmap")


def fig9_visual_elements():
    """Visual element presence in analyzed questions."""
    elements = summary["visual_element_counts"]
    total = summary["total_valid"]

    labels = ["Has Math", "Has Table", "Has Diagram", "Has Handwriting"]
    counts = [
        elements["has_math"],
        elements["has_table"],
        elements["has_diagram"],
        elements["has_handwriting"],
    ]
    pcts = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#9b59b6", "#1abc9c", "#f39c12", "#e74c3c"]
    bars = ax.bar(labels, pcts, color=colors, edgecolor="black", linewidth=1.2)

    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Percentage of Questions (%)", fontsize=12)
    ax.set_title(
        f"Visual Elements in Analyzed Questions (n={total})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig9_visual_elements.png")
    plt.savefig(FIGURES_DIR / "fig9_visual_elements.pdf")
    plt.close()
    print("Generated fig9_visual_elements")


def fig10_task_failure_stacked():
    """Stacked bar chart of failure categories by task domain."""
    task_data = summary["task_category_breakdown"]

    tasks_with_counts = []
    for task, cats in task_data.items():
        total = sum(cats.values())
        if total >= 3:
            tasks_with_counts.append((task, total, cats))

    tasks_with_counts.sort(key=lambda x: -x[1])
    tasks_with_counts = tasks_with_counts[:10]

    tasks = [t[0] for t in tasks_with_counts]

    all_categories = list(summary["primary_category_distribution"].keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(tasks))
    width = 0.7
    bottom = np.zeros(len(tasks))

    for cat in all_categories:
        heights = []
        for task, total, cats in tasks_with_counts:
            heights.append(cats.get(cat, 0))

        if sum(heights) > 0:
            bars = ax.bar(
                x,
                heights,
                width,
                bottom=bottom,
                label=CATEGORY_LABELS.get(cat, cat),
                color=COLORS.get(cat, "#cccccc"),
            )
            bottom += np.array(heights)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("_", " ").title() for t in tasks], rotation=45, ha="right"
    )
    ax.set_ylabel("Number of Questions", fontsize=12)
    ax.set_title(
        "Failure Categories by Task Domain (Top 10 Domains)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig10_task_failure_stacked.png")
    plt.savefig(FIGURES_DIR / "fig10_task_failure_stacked.pdf")
    plt.close()
    print("Generated fig10_task_failure_stacked")


def fig11_secondary_categories():
    """Secondary failure categories comparison."""
    primary = summary["primary_category_distribution"]
    secondary = summary["secondary_category_distribution"]

    all_cats = set(primary.keys()) | set(secondary.keys())
    all_cats = sorted(
        all_cats, key=lambda x: primary.get(x, 0) + secondary.get(x, 0), reverse=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_cats))
    width = 0.35

    primary_vals = [primary.get(c, 0) for c in all_cats]
    secondary_vals = [secondary.get(c, 0) for c in all_cats]

    bars1 = ax.bar(x - width / 2, primary_vals, width, label="Primary", color="#3498db")
    bars2 = ax.bar(
        x + width / 2,
        secondary_vals,
        width,
        label="Secondary",
        color="#e74c3c",
        alpha=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CATEGORY_LABELS.get(c, c) for c in all_cats], rotation=45, ha="right"
    )
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Primary vs Secondary Failure Categories", fontsize=14, fontweight="bold"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig11_primary_vs_secondary.png")
    plt.savefig(FIGURES_DIR / "fig11_primary_vs_secondary.pdf")
    plt.close()
    print("Generated fig11_primary_vs_secondary")


def main():
    print("Generating Phase 2 figures...")
    fig7_primary_category_distribution()
    fig8_nation_failure_heatmap()
    fig9_visual_elements()
    fig10_task_failure_stacked()
    fig11_secondary_categories()
    print("\nAll Phase 2 figures generated in figures/")


if __name__ == "__main__":
    main()
