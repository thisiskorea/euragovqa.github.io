#!/usr/bin/env python3
"""
Integrated Publication Figures for EuraGovExam
Target: NeurIPS Datasets & Benchmarks Track 2025

Main Text Figures:
- Figure 1: Model Performance Overview (Closed vs Open + Family breakdown)
- Figure 2: Cross-Regional Analysis (Heatmap + Difficulty rankings)

Appendix Figures:
- Figure A1: VCE Analysis
- Figure A2: Error Fingerprints
- Figure A3: Pairwise Effect Sizes
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

SINGLE_COL = 3.25
DOUBLE_COL = 6.75

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman"],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors
COLORS = {
    'blue': '#648FFF',
    'purple': '#785EF0',
    'magenta': '#DC267F',
    'orange': '#FE6100',
    'yellow': '#FFB000',
    'gray': '#6C757D',
}

MODEL_COLORS = {'closed': '#648FFF', 'open': '#FE6100'}
TRACK_COLORS = {'A': '#648FFF', 'B': '#FFB000', 'C': '#DC267F'}

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'custom_diverging', ['#DC267F', '#FFFFFF', '#648FFF']
)

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Model classification
CLOSED_MODELS = {
    "o3", "o4-mini", "GPT-4o", "GPT-4.1", "GPT-4.1-mini",
    "GPT-5", "GPT-5.2", "GPT-5-nano",
    "Gemini-2.5-pro", "Gemini-2.5-flash", "Gemini-2.5-flash-lite",
    "Gemini-3-pro", "Gemini-3-flash",
    "Claude-Sonnet-4"
}

MODEL_FAMILIES = {
    "Google": ["Gemini-2.5-pro", "Gemini-2.5-flash", "Gemini-2.5-flash-lite", "Gemini-3-pro", "Gemini-3-flash"],
    "OpenAI": ["o3", "o4-mini", "GPT-4o", "GPT-4.1", "GPT-4.1-mini", "GPT-5", "GPT-5.2", "GPT-5-nano"],
    "Anthropic": ["Claude-Sonnet-4"],
    "Qwen": ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2-VL-72B-Instruct"],
    "LLaVA": ["llama3-llava-next-8b", "llava-1.5-13b", "llava-1.5-7b", "LLaVA-NeXT-Video-7B-DPO-hf"],
    "Ovis": ["Ovis2-8B", "Ovis2-16B", "Ovis2-32B"],
    "Other": ["Phi-3.5-vision-instruct", "InternVL2.5-38B-MPO", "Llama-3.2-11B-Vision"]
}

NATION_DISPLAY = {"india": "India", "eu": "EU", "taiwan": "Taiwan", "japan": "Japan", "south_korea": "S. Korea"}
TASK_DISPLAY = {
    "chemistry": "Chem.", "philosophy": "Phil.", "earth_science": "Earth Sci.",
    "psychology": "Psych.", "economics": "Econ.", "biology": "Bio.",
    "geography": "Geo.", "physics": "Phys.", "politics": "Poli.",
    "history": "Hist.", "administration": "Admin.", "language": "Lang.",
    "medicine": "Med.", "engineering": "Eng.", "computer_science": "CS",
    "law": "Law", "mathematics": "Math."
}


def load_data():
    with open(ANALYSIS_DIR / "leaderboard.json") as f:
        leaderboard = json.load(f)
    with open(RESULTS_DIR / "vce_analysis.json") as f:
        vce_data = json.load(f)
    with open(ANALYSIS_DIR / "pairwise_test_results.json") as f:
        pairwise_data = json.load(f)
    return leaderboard, vce_data, pairwise_data


# =============================================================================
# FIGURE 1: Model Performance Overview (Main Text)
# =============================================================================

def create_figure1(leaderboard):
    """
    Figure 1: Model Performance Overview
    (a) Closed vs Open box plot
    (b) Performance by model family
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.2))

    # Data preparation
    closed_vals = [m["overall"] for m in leaderboard if m["model"] in CLOSED_MODELS]
    open_vals = [m["overall"] for m in leaderboard if m["model"] not in CLOSED_MODELS]

    # Statistics
    pooled_std = np.sqrt(((len(closed_vals)-1)*np.var(closed_vals, ddof=1) +
                          (len(open_vals)-1)*np.var(open_vals, ddof=1)) /
                         (len(closed_vals) + len(open_vals) - 2))
    cohens_d = (np.mean(closed_vals) - np.mean(open_vals)) / pooled_std

    # Panel (a): Box plot
    ax1 = axes[0]
    bp = ax1.boxplot([closed_vals, open_vals], positions=[0, 1], widths=0.5,
                     patch_artist=True, showfliers=False)

    bp['boxes'][0].set_facecolor(MODEL_COLORS['closed'])
    bp['boxes'][1].set_facecolor(MODEL_COLORS['open'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        box.set_linewidth(0.5)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(0.5)
    for cap in bp['caps']:
        cap.set_linewidth(0.5)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1)

    # Scatter
    np.random.seed(42)
    ax1.scatter(np.random.normal(0, 0.06, len(closed_vals)), closed_vals,
               c=MODEL_COLORS['closed'], s=18, alpha=0.8, edgecolor='white', linewidth=0.3, zorder=5)
    ax1.scatter(np.random.normal(1, 0.06, len(open_vals)), open_vals,
               c=MODEL_COLORS['open'], s=18, alpha=0.8, edgecolor='white', linewidth=0.3, zorder=5)

    # Baseline
    ax1.axhline(y=25, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.text(1.55, 25, 'Random', fontsize=6, color=COLORS['gray'], va='center')

    # Significance bracket
    max_y = max(max(closed_vals), max(open_vals))
    ax1.plot([0, 0, 1, 1], [max_y+2, max_y+4, max_y+4, max_y+2], lw=0.5, c='black')
    ax1.text(0.5, max_y+4.5, f'$p < 0.001$, $d = {cohens_d:.1f}$', ha='center', va='bottom', fontsize=6)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f'Closed\n($n$={len(closed_vals)})', f'Open\n($n$={len(open_vals)})'])
    ax1.set_ylabel(r'Overall Accuracy (\%)')
    ax1.set_ylim(0, 102)
    ax1.set_title(r'\textbf{(a)} Closed vs. Open Source', loc='left', fontsize=8)

    # Panel (b): Family breakdown
    ax2 = axes[1]

    family_data = []
    for family, models in MODEL_FAMILIES.items():
        scores = [m["overall"] for m in leaderboard if m["model"] in models]
        if scores:
            family_data.append({
                'family': family,
                'mean': np.mean(scores),
                'std': np.std(scores, ddof=1) if len(scores) > 1 else 0,
                'n': len(scores),
                'type': 'closed' if family in ['OpenAI', 'Google', 'Anthropic'] else 'open'
            })

    family_data.sort(key=lambda x: x['mean'], reverse=True)

    x = np.arange(len(family_data))
    colors = [MODEL_COLORS[f['type']] for f in family_data]

    bars = ax2.bar(x, [f['mean'] for f in family_data],
                   yerr=[f['std'] for f in family_data],
                   capsize=2, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.3,
                   error_kw={'linewidth': 0.5, 'capthick': 0.5})

    ax2.axhline(y=25, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f['family'] for f in family_data], rotation=35, ha='right')
    ax2.set_ylabel(r'Overall Accuracy (\%)')
    ax2.set_ylim(0, 102)
    ax2.set_title(r'\textbf{(b)} Performance by Model Family', loc='left', fontsize=8)

    # n labels
    for i, f in enumerate(family_data):
        ax2.text(i, f['mean'] + f['std'] + 2, f"$n$={f['n']}",
                ha='center', fontsize=5, color=COLORS['gray'])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=MODEL_COLORS['closed'], alpha=0.8, label='Closed', edgecolor='black', linewidth=0.3),
        mpatches.Patch(facecolor=MODEL_COLORS['open'], alpha=0.8, label='Open', edgecolor='black', linewidth=0.3),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=6)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_main_performance.pdf")
    fig.savefig(FIGURES_DIR / "fig1_main_performance.png", dpi=300)
    plt.close(fig)
    print("Saved: fig1_main_performance.pdf/png")


# =============================================================================
# FIGURE 2: Cross-Regional Analysis (Main Text)
# =============================================================================

def create_figure2(leaderboard):
    """
    Figure 2: Cross-Regional Analysis
    (a) Model×Region interaction heatmap (left, larger)
    (b) Region difficulty (top-right)
    (c) Task difficulty - top 10 hardest (bottom-right)
    """
    fig = plt.figure(figsize=(DOUBLE_COL, 3.8))

    # GridSpec: 2 rows, 2 cols with width ratios
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1],
                           wspace=0.28, hspace=0.55)

    # Panel (a): Heatmap - spans both rows on left
    ax_heatmap = fig.add_subplot(gs[:, 0])

    # Panel (b): Region difficulty - top right
    ax_region = fig.add_subplot(gs[0, 1])

    # Panel (c): Task difficulty - bottom right
    ax_task = fig.add_subplot(gs[1, 1])

    # === Panel (a): Heatmap ===
    models_sorted = sorted(leaderboard, key=lambda x: x["overall"], reverse=True)
    model_names = [m["model"] for m in models_sorted]
    nations = ["india", "japan", "south_korea", "taiwan", "eu"]

    delta_matrix = np.zeros((len(model_names), len(nations)))
    for i, model in enumerate(models_sorted):
        overall = model["overall"]
        for j, nation in enumerate(nations):
            nation_acc = model["nation"].get(nation, 0)
            delta_matrix[i, j] = nation_acc - overall

    im = ax_heatmap.imshow(delta_matrix, cmap=DIVERGING_CMAP, aspect='auto', vmin=-25, vmax=25)

    # Grid
    ax_heatmap.set_xticks(np.arange(len(nations)+1)-0.5, minor=True)
    ax_heatmap.set_yticks(np.arange(len(model_names)+1)-0.5, minor=True)
    ax_heatmap.grid(which='minor', color='white', linewidth=0.5)
    ax_heatmap.tick_params(which='minor', size=0)

    ax_heatmap.set_xticks(range(len(nations)))
    ax_heatmap.set_xticklabels([NATION_DISPLAY.get(n, n) for n in nations], fontsize=6)
    ax_heatmap.set_yticks(range(len(model_names)))
    ax_heatmap.set_yticklabels(model_names, fontsize=5)

    # Color y-labels
    for i, label in enumerate(ax_heatmap.get_yticklabels()):
        if model_names[i] in CLOSED_MODELS:
            label.set_color(MODEL_COLORS['closed'])
        else:
            label.set_color(MODEL_COLORS['open'])

    # Annotations for extreme values
    for i in range(len(model_names)):
        for j in range(len(nations)):
            val = delta_matrix[i, j]
            if abs(val) >= 14:
                color = 'white' if abs(val) > 18 else 'black'
                ax_heatmap.text(j, i, f'{val:+.0f}', ha='center', va='center',
                               fontsize=4, color=color, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heatmap, shrink=0.6, pad=0.02)
    cbar.set_label(r'$\Delta$ (Region $-$ Overall) \%p', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax_heatmap.set_xlabel('Region', fontsize=7)
    ax_heatmap.set_title(r'\textbf{(a)} Model $\times$ Region Interaction', loc='left', fontsize=8)

    # === Panel (b): Region difficulty ===
    nation_means = {}
    for model in leaderboard:
        for nation in model["nation"]:
            if nation not in nation_means:
                nation_means[nation] = []
            nation_means[nation].append(model["nation"][nation])

    nation_avg = {n: (np.mean(v), np.std(v)) for n, v in nation_means.items()}
    sorted_nations = sorted(nation_avg.items(), key=lambda x: x[1][0])

    y_pos = np.arange(len(sorted_nations))
    means = [x[1][0] for x in sorted_nations]
    stds = [x[1][1] for x in sorted_nations]
    labels = [NATION_DISPLAY.get(x[0], x[0]) for x in sorted_nations]

    norm = plt.Normalize(min(means), max(means))
    colors = plt.cm.RdYlGn(norm(means))

    bars = ax_region.barh(y_pos, means, xerr=stds, height=0.6,
                          color=colors, edgecolor='black', linewidth=0.3,
                          capsize=2, error_kw={'linewidth': 0.4, 'capthick': 0.4})

    ax_region.axvline(x=25, color=COLORS['gray'], linestyle='--', linewidth=0.4, alpha=0.7)
    ax_region.set_yticks(y_pos)
    ax_region.set_yticklabels(labels, fontsize=6)
    ax_region.set_xlabel(r'Avg. Accuracy (\%)', fontsize=6)
    ax_region.set_xlim(0, 70)
    ax_region.set_title(r'\textbf{(b)} By Region', loc='left', fontsize=8)
    ax_region.tick_params(axis='x', labelsize=6)

    for bar, mean, std in zip(bars, means, stds):
        ax_region.text(mean + std + 1, bar.get_y() + bar.get_height()/2,
                      f'{mean:.0f}', va='center', fontsize=5)

    # === Panel (c): Task difficulty (top 10) ===
    task_means = {}
    for model in leaderboard:
        for task in model["tasks"]:
            if task not in task_means:
                task_means[task] = []
            task_means[task].append(model["tasks"][task])

    task_avg = {t: (np.mean(v), np.std(v)) for t, v in task_means.items()}
    sorted_tasks = sorted(task_avg.items(), key=lambda x: x[1][0])[:10]

    y_pos = np.arange(len(sorted_tasks))
    means = [x[1][0] for x in sorted_tasks]
    stds = [x[1][1] for x in sorted_tasks]
    labels = [TASK_DISPLAY.get(x[0], x[0]) for x in sorted_tasks]

    norm = plt.Normalize(min(means), max(means))
    colors = plt.cm.RdYlGn(norm(means))

    bars = ax_task.barh(y_pos, means, xerr=stds, height=0.6,
                        color=colors, edgecolor='black', linewidth=0.3,
                        capsize=2, error_kw={'linewidth': 0.4, 'capthick': 0.4})

    ax_task.axvline(x=25, color=COLORS['gray'], linestyle='--', linewidth=0.4, alpha=0.7)
    ax_task.set_yticks(y_pos)
    ax_task.set_yticklabels(labels, fontsize=6)
    ax_task.set_xlabel(r'Avg. Accuracy (\%)', fontsize=6)
    ax_task.set_xlim(0, 70)
    ax_task.set_title(r'\textbf{(c)} By Task (10 Hardest)', loc='left', fontsize=8)
    ax_task.tick_params(axis='x', labelsize=6)

    for bar, mean, std in zip(bars, means, stds):
        ax_task.text(mean + std + 1, bar.get_y() + bar.get_height()/2,
                    f'{mean:.0f}', va='center', fontsize=5)

    fig.savefig(FIGURES_DIR / "fig2_main_regional.pdf")
    fig.savefig(FIGURES_DIR / "fig2_main_regional.png", dpi=300)
    plt.close(fig)
    print("Saved: fig2_main_regional.pdf/png")


# =============================================================================
# APPENDIX FIGURES
# =============================================================================

def create_appendix_vce(vce_data):
    """Figure A1: VCE Analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0))

    nations = ["Japan", "India", "South Korea", "Taiwan", "EU"]
    nation_stats = vce_data["nation_stats"]

    # Panel (a): 3-Track accuracy
    ax1 = axes[0]
    x = np.arange(len(nations))
    width = 0.22

    acc_a = [nation_stats[n]["acc_a"] * 100 for n in nations]
    acc_b = [nation_stats[n]["acc_b"] * 100 for n in nations]
    acc_c = [nation_stats[n]["acc_c"] * 100 for n in nations]

    ax1.bar(x - width, acc_a, width, label='Track A (Image)', color=TRACK_COLORS['A'], edgecolor='black', linewidth=0.3)
    ax1.bar(x, acc_b, width, label='Track B (Text)', color=TRACK_COLORS['B'], edgecolor='black', linewidth=0.3)
    ax1.bar(x + width, acc_c, width, label='Track C (Both)', color=TRACK_COLORS['C'], edgecolor='black', linewidth=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("South Korea", "S. Korea") for n in nations], fontsize=6)
    ax1.set_ylabel(r'Accuracy (\%)')
    ax1.set_ylim(0, 95)
    ax1.legend(loc='upper left', fontsize=5, ncol=1)
    ax1.set_title(r'\textbf{(a)} Three-Track Accuracy', loc='left', fontsize=8)
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax1.set_axisbelow(True)

    # Panel (b): Noise vs Benefit
    ax2 = axes[1]
    noise = [nation_stats[n]["noise"] for n in nations]
    benefit = [nation_stats[n]["benefit"] for n in nations]

    width = 0.35
    ax2.bar(x - width/2, noise, width, label='Visual Noise', color=COLORS['magenta'], edgecolor='black', linewidth=0.3)
    ax2.bar(x + width/2, benefit, width, label='Visual Benefit', color=COLORS['blue'], edgecolor='black', linewidth=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace("South Korea", "S. Korea") for n in nations], fontsize=6)
    ax2.set_ylabel(r'Count ($n$=40/region)')
    ax2.legend(loc='upper right', fontsize=5)
    ax2.set_title(r'\textbf{(b)} Visual Noise vs. Benefit', loc='left', fontsize=8)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figA1_vce_analysis.pdf")
    fig.savefig(FIGURES_DIR / "figA1_vce_analysis.png", dpi=300)
    plt.close(fig)
    print("Saved: figA1_vce_analysis.pdf/png")


def create_appendix_fingerprint(leaderboard):
    """Figure A2: Error Fingerprints"""
    representative = ["Gemini-2.5-pro", "o3", "Claude-Sonnet-4",
                     "Qwen2-VL-72B-Instruct", "GPT-4o", "InternVL2.5-38B-MPO"]

    tasks = sorted(leaderboard[0]["tasks"].keys())

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 3.2))
    axes = axes.flatten()

    for idx, model_name in enumerate(representative):
        model_data = next((m for m in leaderboard if m["model"] == model_name), None)
        if model_data is None:
            continue

        ax = axes[idx]
        task_accs = [(TASK_DISPLAY.get(t, t), model_data["tasks"][t]) for t in tasks]
        task_accs.sort(key=lambda x: x[1])

        y_pos = np.arange(len(task_accs))
        values = [x[1] for x in task_accs]
        labels = [x[0] for x in task_accs]

        norm = plt.Normalize(0, 100)
        colors = plt.cm.RdYlGn(norm(values))

        ax.barh(y_pos, values, height=0.7, color=colors, edgecolor='black', linewidth=0.2)
        ax.axvline(x=25, color=COLORS['gray'], linestyle='--', linewidth=0.4, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=4)
        ax.set_xlabel(r'Accuracy (\%)', fontsize=5)
        ax.tick_params(axis='x', labelsize=5)

        overall = model_data["overall"]
        ax.set_title(f'{model_name} ({overall:.1f}\\%)', fontsize=6, fontweight='bold')

        color = MODEL_COLORS['closed'] if model_name in CLOSED_MODELS else MODEL_COLORS['open']
        ax.spines['left'].set_color(color)
        ax.spines['left'].set_linewidth(2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figA2_error_fingerprint.pdf")
    fig.savefig(FIGURES_DIR / "figA2_error_fingerprint.png", dpi=300)
    plt.close(fig)
    print("Saved: figA2_error_fingerprint.pdf/png")


def create_appendix_pairwise(pairwise_data):
    """Figure A3: Pairwise Effect Sizes"""
    models = pairwise_data["models_compared"]
    tests = pairwise_data["pairwise_tests"]

    n = len(models)
    effect_matrix = np.full((n, n), np.nan)
    sig_matrix = np.zeros((n, n), dtype=bool)

    for test in tests:
        m1, m2 = test["model_1"], test["model_2"]
        if m1 in models and m2 in models:
            i, j = models.index(m1), models.index(m2)
            effect_matrix[i, j] = test["cohens_h"]
            effect_matrix[j, i] = test["cohens_h"]
            sig_matrix[i, j] = test["significant_bonferroni"]
            sig_matrix[j, i] = test["significant_bonferroni"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.0, SINGLE_COL + 0.6))

    mask = np.triu(np.ones_like(effect_matrix, dtype=bool))
    masked_data = np.ma.array(effect_matrix, mask=mask)

    im = ax.imshow(masked_data, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1.1)

    for i in range(n):
        for j in range(i):
            h = effect_matrix[i, j]
            sig = sig_matrix[i, j]
            if not np.isnan(h):
                color = 'white' if h > 0.6 else 'black'
                text = f'{h:.2f}' if sig else f'({h:.2f})'
                ax.text(j, i, text, ha='center', va='center', fontsize=5, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(models, fontsize=5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(r"Cohen's $h$", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    for val, label in [(0.2, 'S'), (0.5, 'M'), (0.8, 'L')]:
        cbar.ax.axhline(y=val, color='black', linestyle='-', linewidth=0.3)

    ax.set_title(r"Pairwise Effect Sizes (Top 10)", fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figA3_pairwise_effects.pdf")
    fig.savefig(FIGURES_DIR / "figA3_pairwise_effects.png", dpi=300)
    plt.close(fig)
    print("Saved: figA3_pairwise_effects.pdf/png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("EuraGovExam Integrated Figures")
    print("=" * 60)

    FIGURES_DIR.mkdir(exist_ok=True)

    print("\nLoading data...")
    leaderboard, vce_data, pairwise_data = load_data()

    print("\n--- Main Text Figures ---")
    print("\n[Figure 1] Model Performance Overview...")
    create_figure1(leaderboard)

    print("\n[Figure 2] Cross-Regional Analysis...")
    create_figure2(leaderboard)

    print("\n--- Appendix Figures ---")
    print("\n[Figure A1] VCE Analysis...")
    create_appendix_vce(vce_data)

    print("\n[Figure A2] Error Fingerprints...")
    create_appendix_fingerprint(leaderboard)

    print("\n[Figure A3] Pairwise Effects...")
    create_appendix_pairwise(pairwise_data)

    print("\n" + "=" * 60)
    print("Done! Generated:")
    print("  Main:     fig1_main_performance.pdf, fig2_main_regional.pdf")
    print("  Appendix: figA1_vce_analysis.pdf, figA2_error_fingerprint.pdf, figA3_pairwise_effects.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
