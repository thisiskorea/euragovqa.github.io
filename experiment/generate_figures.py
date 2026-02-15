import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / "vce_analysis.json", "r") as f:
    vce_data = json.load(f)

with open(RESULTS_DIR / "statistical_analysis_results.json", "r") as f:
    stats_data = json.load(f)

with open(RESULTS_DIR / "visual_noise_image_analysis.json", "r") as f:
    noise_analysis = json.load(f)


def figure1_vce_by_region():
    nations = ["Japan", "Taiwan", "India", "South Korea", "EU"]
    vce_values = [stats_data["vce_by_nation"][n]["vce"] * 100 for n in nations]
    ci_lower = [stats_data["vce_by_nation"][n]["ci"][0] * 100 for n in nations]
    ci_upper = [stats_data["vce_by_nation"][n]["ci"][1] * 100 for n in nations]

    errors = [
        [vce - low for vce, low in zip(vce_values, ci_lower)],
        [high - vce for vce, high in zip(vce_values, ci_upper)],
    ]

    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vce_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(nations, vce_values, color=colors, edgecolor="black", linewidth=1.2)
    ax.errorbar(
        nations,
        vce_values,
        yerr=errors,
        fmt="none",
        color="black",
        capsize=5,
        capthick=2,
    )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_ylabel("VCE (Text-only − Multimodal) %", fontsize=12)
    ax.set_xlabel("Region", fontsize=12)
    ax.set_title(
        "Visual Causal Effect by Region\n(Positive = Images Hurt, Negative = Images Help)",
        fontsize=14,
    )

    ax.set_ylim(-25, 30)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    noise_patch = mpatches.Patch(color="#d62728", label="Visual Noise (images hurt)")
    benefit_patch = mpatches.Patch(
        color="#2ca02c", label="Visual Benefit (images help)"
    )
    ax.legend(handles=[noise_patch, benefit_patch], loc="upper right")

    for bar, val in zip(bars, vce_values):
        height = bar.get_height()
        ax.annotate(
            f"{val:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -15),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_vce_by_region.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig1_vce_by_region.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig1_vce_by_region.png/pdf")


def figure2_visual_noise_patterns():
    labels = ["Text-Heavy\n(No visual content)", "Simple Diagram", "Complex Visual"]
    sizes = [
        noise_analysis["pattern_classification"]["Pattern_A_Text_Heavy"]["count"],
        noise_analysis["pattern_classification"]["Pattern_B_Simple_Diagram"]["count"],
        noise_analysis["pattern_classification"]["Pattern_C_Complex_Visual"]["count"],
    ]
    colors = ["#ff9999", "#66b3ff", "#99ff99"]
    explode = (0.05, 0, 0)

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.0f}%\n({int(pct/100*sum(sizes))})",
        shadow=True,
        startangle=90,
        textprops={"fontsize": 12},
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    ax.set_title(
        "Visual Noise Cases by Document Type\n(n=13 cases where images hurt performance)",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        0,
        -1.3,
        "69% of Visual Noise cases have NO meaningful visual content\n→ Fusion Interference from processing text-as-image",
        ha="center",
        fontsize=11,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "fig2_visual_noise_patterns.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(FIGURES_DIR / "fig2_visual_noise_patterns.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig2_visual_noise_patterns.png/pdf")


def figure3_track_accuracy():
    nations = ["Japan", "Taiwan", "India", "South Korea", "EU"]

    track_a = [vce_data["nation_stats"][n]["acc_a"] * 100 for n in nations]
    track_b = [vce_data["nation_stats"][n]["acc_b"] * 100 for n in nations]
    track_c = [vce_data["nation_stats"][n]["acc_c"] * 100 for n in nations]

    x = np.arange(len(nations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(
        x - width,
        track_a,
        width,
        label="Track A (Image-only)",
        color="#ff7f0e",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x,
        track_b,
        width,
        label="Track B (Text-only)",
        color="#1f77b4",
        edgecolor="black",
    )
    bars3 = ax.bar(
        x + width,
        track_c,
        width,
        label="Track C (Multimodal)",
        color="#2ca02c",
        edgecolor="black",
    )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Region", fontsize=12)
    ax.set_title(
        "Model Accuracy by Modality and Region\n(Gemini-2.0-Flash, n=40 per region)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(nations)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    for i, nation in enumerate(nations):
        if track_b[i] > track_c[i]:
            ax.annotate(
                "",
                xy=(i + width, track_c[i] + 2),
                xytext=(i, track_b[i] + 2),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
            )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_track_accuracy.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig3_track_accuracy.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig3_track_accuracy.png/pdf")


def figure4_noise_benefit_comparison():
    nations = ["Japan", "Taiwan", "South Korea", "India", "EU"]

    noise_counts = [vce_data["nation_stats"][n]["noise"] for n in nations]
    benefit_counts = [vce_data["nation_stats"][n]["benefit"] for n in nations]

    y = np.arange(len(nations))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.barh(
        y - height / 2,
        noise_counts,
        height,
        label="Visual Noise Cases",
        color="#d62728",
        edgecolor="black",
    )
    bars2 = ax.barh(
        y + height / 2,
        benefit_counts,
        height,
        label="Visual Benefit Cases",
        color="#2ca02c",
        edgecolor="black",
    )

    ax.set_xlabel("Number of Cases", fontsize=12)
    ax.set_ylabel("Region", fontsize=12)
    ax.set_title(
        "Visual Noise vs Benefit Cases by Region\n(out of n=40 samples per region)",
        fontsize=14,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(nations)
    ax.legend(loc="lower right")
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 10)

    for bar, val in zip(bars1, noise_counts):
        if val > 0:
            ax.text(
                val + 0.2,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    for bar, val in zip(bars2, benefit_counts):
        if val > 0:
            ax.text(
                val + 0.2,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    ax.text(
        7,
        4.5,
        "Japan: 7 noise vs 5 benefit\n→ Net Visual Noise",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="#ffcccc", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "fig4_noise_benefit_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(FIGURES_DIR / "fig4_noise_benefit_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig4_noise_benefit_comparison.png/pdf")


def figure5_regional_distribution_noise_cases():
    regions = list(noise_analysis["regional_distribution"].keys())
    counts = [noise_analysis["regional_distribution"][r]["count"] for r in regions]

    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4"]

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=regions,
        colors=colors,
        autopct=lambda pct: f"{pct:.0f}%\n({int(pct/100*sum(counts))})",
        shadow=True,
        startangle=140,
        textprops={"fontsize": 12},
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    ax.set_title(
        "Regional Distribution of Visual Noise Cases\n(n=13 total)",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        0,
        -1.3,
        "Japan accounts for 54% of all Visual Noise cases\n→ Dense Japanese text may cause more Fusion Interference",
        ha="center",
        fontsize=11,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "fig5_regional_noise_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        FIGURES_DIR / "fig5_regional_noise_distribution.pdf", bbox_inches="tight"
    )
    plt.close()
    print("Saved: fig5_regional_noise_distribution.png/pdf")


if __name__ == "__main__":
    print("Generating figures for VCE Paper...")
    print("=" * 50)

    figure1_vce_by_region()
    figure2_visual_noise_patterns()
    figure3_track_accuracy()
    figure4_noise_benefit_comparison()
    figure5_regional_distribution_noise_cases()

    print("=" * 50)
    print(f"All figures saved to: {FIGURES_DIR}")
